import os

from datasets import inspect_dataset, load_dataset_builder, Split, load_dataset
from datasets.iterable_dataset import IterableDataset
from transformers import PreTrainedTokenizer
from hydra.utils import to_absolute_path

def get_dataset(dataset_name: str = "wmt14", language_pair: str = "de-en", stream: bool = True) -> IterableDataset:
    ds = load_dataset(dataset_name, language_pair, streaming=stream)
    return ds.with_format("torch")


class DataCollatorForSupervisedMT:
    def __init__(self, src_tokenizer: PreTrainedTokenizer, tgt_tokenizer: PreTrainedTokenizer, max_seq_len=256, src_key="de", tgt_key="en"):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len
        self.src_key = src_key
        self.tgt_key = tgt_key

    def __call__(self, features):
        features = [x["translation"] if "translation" in x else x for x in features]
        src_text = [x[self.src_key] for x in features]
        tgt_text = [x[self.tgt_key] for x in features]

        src_tokens = self.src_tokenizer(src_text, padding=True, max_length=self.max_seq_len, truncation=True, return_tensors="pt")
        tgt_tokens = self.tgt_tokenizer(tgt_text, padding=True, max_length=self.max_seq_len, truncation=True, return_tensors="pt")
        return {
            "input_ids": src_tokens["input_ids"],
            "attention_mask": src_tokens["attention_mask"],
            "labels": tgt_tokens["input_ids"],
        }


class DataCollatorForUnsupervisedMT:
    def __init__(
        self,
        src_tokenizer: PreTrainedTokenizer,
        tgt_tokenizer: PreTrainedTokenizer,
        src_key: str = "de",
        tgt_key: str = "en",
        max_seq_len: int = 512,
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.max_seq_len = max_seq_len

    def __call__(self, features):
        features = [x["translation"] if "translation" in x else x for x in features]
        src_text = [x[self.src_key] for x in features]
        tgt_text = [x[self.tgt_key] for x in features]

        src_tokens = self.src_tokenizer(src_text, padding=True, max_length=self.max_seq_len, truncation=True, return_tensors="pt")
        tgt_tokens = self.tgt_tokenizer(tgt_text, padding=True, max_length=self.max_seq_len, truncation=True, return_tensors="pt")
        return {
            "input_ids": src_tokens["input_ids"],
            "attention_mask_src": src_tokens["attention_mask"],
            "attention_mask_tgt": tgt_tokens["attention_mask"],
            "labels": tgt_tokens["input_ids"],
        }
