import sys, os
import json

import hydra
import dotenv
import evaluate
from hydra.utils import to_absolute_path

# add the working directory to $PYTHONPATH
# needed to make local imports work
sys.path.append(os.getenv("PWD", "."))
# load the `.env` file 
dotenv.load_dotenv()

from src.unsupervised.model import UnsupervisedTranslation, get_tokenizers, get_model


@hydra.main(config_path="../config", config_name="generate_unsupervised", version_base="1.1")
def main(config):
    tokenizer_a, tokenizer_b = get_tokenizers()
    autoencoder_a = get_model(vocab_size=tokenizer_a.vocab_size)
    autoencoder_b = get_model(vocab_size=tokenizer_b.vocab_size)
    model = UnsupervisedTranslation.load_from_checkpoint(
        to_absolute_path(config.model_path),
        autoencoder_a=autoencoder_a,
        autoencoder_b=autoencoder_b,
    )

    bleu = evaluate.load("bleu")

    sentences_a = [["The cat eats the mouse."]]
    sentences_b = [["Die Katze isst die Maus."]]

    translations = []
    metrics = []
    for xs, ys in zip(sentences_a, sentences_b):
        tokens_a = tokenizer_a(xs, max_length=config.max_seq_len, padding=True, truncation=True, return_tensors="pt")
        tokens_b = tokenizer_b(ys, max_length=config.max_seq_len, padding=True, truncation=True, return_tensors="pt")

        pred_tokens = model.translate_from_a_to_b(
            input_ids=tokens_a["input_ids"],
            attention_mask=tokens_a["attention_mask"],
            decoder_input_ids=tokens_b["input_ids"][:, :1],
            max_new_tokens=config.max_new_tokens,
        )
        translation = tokenizer_b.batch_decode(pred_tokens, skip_special_tokens=True)
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
