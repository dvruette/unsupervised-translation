import sys, os
import json

import hydra
import torch
import dotenv
from hydra.utils import to_absolute_path
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
)

# add the working directory to $PYTHONPATH
# needed to make local imports work
sys.path.append(os.getenv("PWD", "."))
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


    src_sentences = [["Translate from English to German: The cat eats the mouse."]]
    tgt_sentences = [["Die Katze isst die Maus."]]

    translations = []
    metrics = []
    for x, y in zip(src_sentences, tgt_sentences):
        src_tokens = src_tokenizer(x, max_length=config.max_seq_len, padding=True, truncation=True, return_tensors="pt")
        # tgt_tokens = tgt_tokenizer(y, max_length=config.max_seq_len, padding=True, truncation=True, return_tensors="pt")

        pred_tokens = model.generate(
            input_ids=src_tokens["input_ids"],
            attention_mask=src_tokens["attention_mask"],
            # decoder_input_ids=tgt_tokens["input_ids"][:1],
            max_new_tokens=config.max_new_tokens,
        )
        translation = tgt_tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        translations.append({
            "src": x,
            "tgt": y,
            "pred": translation,
        })
        print(translation)

    with open("translations.json", "w") as f:
        json.dump(translations, f)

    


if __name__ == "__main__":
    main()