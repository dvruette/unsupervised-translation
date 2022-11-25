import sys, os
import json
from typing import Tuple, List

import hydra
import dotenv
import evaluate
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from hydra.utils import to_absolute_path

# add the working directory to $PYTHONPATH
# needed to make local imports work
sys.path.append(os.getenv("PWD", "."))
# load the `.env` file 
dotenv.load_dotenv()

from src.unsupervised.model import UnsupervisedTranslation, get_tokenizers
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

    tokenizer_a, tokenizer_b = get_tokenizers()
    model = UnsupervisedTranslation.load_from_checkpoint(
        to_absolute_path(config.model_path),
        vocab_size_a=tokenizer_a.vocab_size,
        vocab_size_b=tokenizer_b.vocab_size,
    )

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

    model.eval()
    model.to(device)
    with torch.no_grad():
        max_batches = 16
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

            # generate translations
            pred_tokens = model.translate_from_a_to_b(
                input_ids=input_ids_a,
                attention_mask=attention_mask_a,
                decoder_input_ids=input_ids_b[:, :1],
                max_new_tokens=config.max_new_tokens,
                eos_token_id=tokenizer_b.eos_token_id,
            )
            translation = tokenizer_b.batch_decode(pred_tokens, skip_special_tokens=True)
            for x, y, t in zip(xs, ys, translation):
                translations_a_to_b.append({"src": x, "ref": y, "pred": t})

            pred_tokens = model.translate_from_b_to_a(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                decoder_input_ids=input_ids_a[:, :1],
                max_new_tokens=config.max_new_tokens,
                eos_token_id=tokenizer_a.eos_token_id,
            )
            translation = tokenizer_a.batch_decode(pred_tokens, skip_special_tokens=True)
            for x, y, t in zip(xs, ys, translation):
                translations_b_to_a.append({"src": x, "ref": y, "pred": t})

            # compute perplexity
            enc = model.encode_a(
                input_ids=input_ids_a,
                attention_mask=attention_mask_a,
            )
            logits = model.decode_b(input_ids_b[:, :-1], enc["z"])
            ppl, loss = compute_ppl(logits, input_ids_b, pad_id=tokenizer_b.pad_token_id)
            ppls_a_to_b.extend(ppl)
            losses_a_to_b.extend(loss)

            enc = model.encode_b(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
            )
            logits = model.decode_a(input_ids_a[:, :-1], enc["z"])
            ppl, loss = compute_ppl(logits, input_ids_a, pad_id=tokenizer_a.pad_token_id)
            ppls_b_to_a.extend(ppl)
            losses_b_to_a.extend(loss)


    predictions = [t["pred"] for t in translations_a_to_b]
    references = [[t["ref"]] for t in translations_a_to_b]
    scores = bleu.compute(predictions=predictions, references=references)

    # print metrics
    print("BLEU (a -> b): ", scores["bleu"])
    print("PPL (a -> b): ", np.mean(ppls_a_to_b))
    print("Loss (a -> b): ", np.mean(losses_a_to_b))

    predictions = [t["pred"] for t in translations_b_to_a]
    references = [[t["ref"]] for t in translations_b_to_a]
    scores = bleu.compute(predictions=predictions, references=references)

    print("BLEU (b -> a): ", scores["bleu"])
    print("PPL (b -> a): ", np.mean(ppls_b_to_a))
    print("Loss (b -> a): ", np.mean(losses_b_to_a))


    with open("translations_a_to_b.json", "w") as f:
        json.dump(translations_a_to_b, f)

    with open("translations_b_to_a.json", "w") as f:
        json.dump(translations_b_to_a, f)


if __name__ == "__main__":
    main()
