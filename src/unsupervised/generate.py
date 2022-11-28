import sys, os
import json
from typing import Tuple, List

import hydra
import dotenv
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from hydra.utils import to_absolute_path

# add the working directory to $PYTHONPATH
# needed to make local imports work
sys.path.append(os.getenv("PWD", "."))
# load the `.env` file 
dotenv.load_dotenv()

from src.unsupervised.model import UnsupervisedTranslation
from src.data import get_dataset


def compute_ppl(logits, labels, pad_id=0) -> Tuple[List[float], List[float]]:
    entropy = F.cross_entropy(logits.transpose(1, 2), labels[:, 1:], ignore_index=pad_id, reduction="none")
    loss = F.cross_entropy(logits.transpose(1, 2), labels[:, 1:], ignore_index=pad_id)
    # normalize for sequence length
    t = (labels[:, 1:] != pad_id).sum(dim=1)
    ppl = (entropy.sum(dim=1) / t).exp()
    return ppl.tolist(), [loss]


@hydra.main(config_path="../config", config_name="generate_unsupervised", version_base="1.1")
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UnsupervisedTranslation.load_from_checkpoint(to_absolute_path(config.model_path), map_location=device)
    tokenizer_a, tokenizer_b = model.tokenizer_a, model.tokenizer_b

    bleu = evaluate.load("bleu")

    ds = get_dataset(
        dataset_name=config.data.dataset_name,
        language_pair=config.data.language_pair,
        stream=False
    )

    dl = torch.utils.data.DataLoader(
        ds[config.data.split],
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
    )

    a_key, b_key = config.data.language_pair.split("-")
    translations_a_to_b = []
    ppls_a_to_b = []
    losses_a_to_b = []

    translations_b_to_a = []
    ppls_b_to_a = []
    losses_b_to_a = []

    metrics = []

    max_batches = config.data.max_batches
    if max_batches is None or max_batches <= 0:
        max_batches = len(dl)

    # model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(dl, desc="Generating translations", total=min(max_batches, len(dl)))):
            if i == max_batches:
                break
            xs = batch["translation"][a_key]
            ys = batch["translation"][b_key]

            tokens_a = tokenizer_a(xs, max_length=config.max_seq_len, padding=True, truncation=True, return_tensors="pt")
            tokens_b = tokenizer_b(ys, max_length=config.max_seq_len, padding=True, truncation=True, return_tensors="pt")

            input_ids_a = tokens_a["input_ids"].to(device)
            attention_mask_a = tokens_a["attention_mask"].to(device)
            input_ids_b = tokens_b["input_ids"].to(device)
            attention_mask_b = tokens_b["attention_mask"].to(device)

            batch = {
                "input_ids": input_ids_a,
                "labels": input_ids_b,
                "attention_mask_src": attention_mask_a,
                "attention_mask_tgt": attention_mask_b,
            }
            ms = model.get_loss(batch)
            metrics.append(ms)

            # generate translations
            pred_tokens_ab = model.translate_from_a_to_b(
                input_ids=input_ids_a,
                attention_mask=attention_mask_a,
                decoder_input_ids=input_ids_b[:, :1],
                max_new_tokens=config.max_new_tokens,
                eos_token_id=tokenizer_b.eos_token_id,
            )
            translation_ab = tokenizer_b.batch_decode(pred_tokens_ab, skip_special_tokens=True)
            for x, y, t in zip(xs, ys, translation_ab):
                translations_a_to_b.append({"src": x, "ref": y, "pred": t})

            pred_tokens_ba = model.translate_from_b_to_a(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                decoder_input_ids=input_ids_a[:, :1],
                max_new_tokens=config.max_new_tokens,
                eos_token_id=tokenizer_a.eos_token_id,
            )
            translation_ba = tokenizer_a.batch_decode(pred_tokens_ba, skip_special_tokens=True)
            for x, y, t in zip(xs, ys, translation_ba):
                translations_b_to_a.append({"src": y, "ref": x, "pred": t})

            # compute perplexity
            enc_a = model.encode_a(
                input_ids=input_ids_a,
                attention_mask=attention_mask_a,
            )
            logits_ab = model.decode_b(input_ids_b[:, :-1], enc_a["z"])
            ppl_ab, loss_ab = compute_ppl(logits_ab, input_ids_b, pad_id=tokenizer_b.pad_token_id)
            ppls_a_to_b.extend(ppl_ab)
            losses_a_to_b.extend(loss_ab)

            enc_b = model.encode_b(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
            )
            logits_ba = model.decode_a(input_ids_a[:, :-1], enc_b["z"])
            ppl_ba, loss_ba = compute_ppl(logits_ba, input_ids_a, pad_id=tokenizer_a.pad_token_id)
            ppls_b_to_a.extend(ppl_ba)
            losses_b_to_a.extend(loss_ba)


    predictions = [t["pred"] for t in translations_a_to_b]
    references = [[t["ref"]] for t in translations_a_to_b]
    scores_ab = bleu.compute(predictions=predictions, references=references)

    predictions = [t["pred"] for t in translations_b_to_a]
    references = [[t["ref"]] for t in translations_b_to_a]
    scores_ba = bleu.compute(predictions=predictions, references=references)

    # print metrics
    print("Metrics:")
    print(pd.DataFrame(metrics).mean())

    print("BLEU (a -> b): ", scores_ab["bleu"])
    print("Prec. (a -> b): ", scores_ab["precisions"])
    print("PPL (a -> b): ", np.mean(ppls_a_to_b))
    print("Loss (a -> b): ", np.mean(losses_a_to_b))

    print("BLEU (b -> a): ", scores_ba["bleu"])
    print("Prec. (b -> a): ", scores_ba["precisions"])
    print("PPL (b -> a): ", np.mean(ppls_b_to_a))
    print("Loss (b -> a): ", np.mean(losses_b_to_a))

    print(f"Saving translations to {os.getcwd()}")
    with open("translations_a_to_b.json", "w") as f:
        json.dump(translations_a_to_b, f, ensure_ascii=False, indent=2)

    with open("translations_b_to_a.json", "w") as f:
        json.dump(translations_b_to_a, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
