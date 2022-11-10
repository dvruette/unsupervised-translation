import os

from datasets import inspect_dataset, load_dataset_builder, Split
from transformers import PreTrainedTokenizer

def get_dataset(dataset_name: bool = "wmt19", stream: bool = True):
    folder = f"{dataset_name}_scripts"
    inspect_dataset("wmt19", folder)
    builder = load_dataset_builder(
        os.path.join(folder, "wmt_utils.py"),
        language_pair=("de", "en"),
        subsets={
            Split.TRAIN: ["commoncrawl"],
            Split.VALIDATION: ["newstest2018"],
        },
    )

    if stream:
        ds = builder.as_streaming_dataset()
    else:
        builder.download_and_prepare()
        ds = builder.as_dataset()

    return ds.with_format("torch")


class DataCollatorForSupervisedMT:
    def __init__(self, src_tokenizer: PreTrainedTokenizer, tgt_tokenizer: PreTrainedTokenizer, src_key="de", tgt_key="en"):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_key = src_key
        self.tgt_key = tgt_key

    def __call__(self, features):
        features = [x["translation"] if "translation" in x else x for x in features]
        src_text = [x[self.src_key] for x in features]
        tgt_text = [x[self.tgt_key] for x in features]

        src_tokens = self.src_tokenizer(src_text, padding=True, max_length=512, truncation=True, return_tensors="pt")
        tgt_tokens = self.tgt_tokenizer(tgt_text, padding=True, max_length=512, truncation=True, return_tensors="pt")
        return {
            "input_ids": src_tokens["input_ids"],
            "attention_mask": src_tokens["attention_mask"],
            "labels": tgt_tokens["input_ids"],
        }


class DataCollatorForUnsupervisedMT:
    def __init__(self, src_tokenizer: PreTrainedTokenizer, tgt_tokenizer: PreTrainedTokenizer, src_key="de", tgt_key="en"):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_key = src_key
        self.tgt_key = tgt_key

    def __call__(self, features):
        features = [x["translation"] if "translation" in x else x for x in features]
        src_text = [x[self.src_key] for x in features]
        tgt_text = [x[self.tgt_key] for x in features]

        src_tokens = self.src_tokenizer(src_text, padding=True, max_length=512, truncation=True, return_tensors="pt")
        tgt_tokens = self.tgt_tokenizer(tgt_text, padding=True, max_length=512, truncation=True, return_tensors="pt")
        return {
            "input_ids": src_tokens["input_ids"],
            "attention_mask_src": src_tokens["attention_mask"],
            "attention_mask_tgt": tgt_tokens["attention_mask"],
            "labels": tgt_tokens["input_ids"],
        }
