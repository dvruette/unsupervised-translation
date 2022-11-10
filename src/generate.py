import sys, os
import json

import hydra
import torch
import dotenv
import evaluate
from hydra.utils import to_absolute_path
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
)

# add the working directory to $PYTHONPATH
# needed to make local imports work
sys.path.append(os.getenv("PWD", "."))
# load the `.env` file 
dotenv.load_dotenv()


@hydra.main(config_path="./config", config_name="generate_baseline", version_base="1.1")
def main(config):
    if config.is_huggingface:
        src_tokenizer = AutoTokenizer.from_pretrained(config.src_tokenizer_path)
        tgt_tokenizer = src_tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path)
    else:
        src_tokenizer = AutoTokenizer.from_pretrained(config.src_tokenizer_path)
        tgt_tokenizer = AutoTokenizer.from_pretrained(config.tgt_tokenizer_path)
        model = torch.load(to_absolute_path(config.model_path))

    bleu = evaluate.load("bleu")

    src_sentences = [["Translate from English to German: The cat eats the mouse."]]
    tgt_sentences = [["Die Katze isst die Maus."]]

    translations = []
    metrics = []
    for xs, ys in zip(src_sentences, tgt_sentences):
        src_tokens = src_tokenizer(xs, max_length=config.max_seq_len, padding=True, truncation=True, return_tensors="pt")
        tgt_tokens = tgt_tokenizer(ys, max_length=config.max_seq_len, padding=True, truncation=True, return_tensors="pt")

        pred_tokens = model.generate(
            input_ids=src_tokens["input_ids"],
            attention_mask=src_tokens["attention_mask"],
            decoder_input_ids=tgt_tokens["input_ids"][:, :1],
            max_new_tokens=config.max_new_tokens,
        )
        translation = tgt_tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        translations.append({
            "src": xs,
            "tgt": ys,
            "pred": translation,
        })
        print(translation)

        score = bleu.compute(predictions=translation, references=[[y] for y in ys])
        metrics.append(score)
        print(score)

    with open("translations.json", "w") as f:
        json.dump(translations, f)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
