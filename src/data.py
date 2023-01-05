import random

import transformers
import torch
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from transformers import PreTrainedTokenizer

def get_dataset(dataset_name: str = "wmt14", language_pair: str = "de-en", stream: bool = False) -> IterableDataset:
    ds = load_dataset(dataset_name, language_pair, streaming=stream)
    return ds.with_format("torch")

def get_unsupervised_dataset(dataset_name: str = "wmt14", language_pair: str = "de-en") -> IterableDataset:
    lang_a, lang_b = language_pair.split("-")
    ds_a = load_dataset(dataset_name, language_pair)
    ds_b = load_dataset(dataset_name, language_pair)
    ds_a, ds_b = ds_a.with_format("torch"), ds_b.with_format("torch")
    return {
        key: UnsupervisedDataset(ds_a[key], ds_b[key], key_a=lang_a, key_b=lang_b) for key in ds_a.keys()
    }

class UnsupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, ds_a, ds_b, key_a, key_b):
        assert len(ds_a) == len(ds_b)
        self.ds_a = ds_a.shuffle()
        self.ds_b = ds_b.shuffle()
        self.key_a = key_a
        self.key_b = key_b

    def __getitem__(self, idx):
        return {
            "translation": {
                self.key_a: self.ds_a[idx]["translation"][self.key_a],
                self.key_b: self.ds_b[idx]["translation"][self.key_b],
            }
        }
    
    def __len__(self):
        return len(self.ds_a)

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
        p_del: float = 0.15,
        p_perm: float = 0.1,
        k_perm: int = 2,
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.max_seq_len = max_seq_len
        self.p_del = p_del
        self.p_perm = p_perm
        self.k_perm = k_perm

    def _augment(self, text: str):
        words = text.split(" ")
        if random.random() < 0.9:
            # each word has a 10% chance of being deleted
            for k in range(self.k_perm):
                # each bigram has a 10% chance of being permuted
                for i, j in zip(range(k % 2, len(words) - 1, 2), range(k % 2 + 1, len(words), 2)):
                    if random.random() < self.p_perm:
                        words[i], words[j] = words[j], words[i]
            words = [word for word in words if random.random() > self.p_del]
        text = " ".join(words)
        return text

    def _tokenize(self, tokenizer: transformers.BertTokenizer, text: str):
        return tokenizer(text, padding=True, max_length=self.max_seq_len, truncation=True, return_tensors="pt")

    def __call__(self, features):
        features = [x["translation"] if "translation" in x else x for x in features]
        src_text = [x[self.src_key] for x in features]
        tgt_text = [x[self.tgt_key] for x in features]

        src_aug_text = [self._augment(x) for x in src_text]
        tgt_aug_text = [self._augment(x) for x in tgt_text]

        src_aug_tokens = self._tokenize(self.src_tokenizer, src_aug_text)
        tgt_aug_tokens = self._tokenize(self.tgt_tokenizer, tgt_aug_text)
        src_tokens = self._tokenize(self.src_tokenizer, src_text)
        tgt_tokens = self._tokenize(self.tgt_tokenizer, tgt_text)

        return {
            "src_input_ids": src_aug_tokens["input_ids"],
            "tgt_input_ids": tgt_aug_tokens["input_ids"],
            "src_attention_mask": src_aug_tokens["attention_mask"],
            "tgt_attention_mask": tgt_aug_tokens["attention_mask"],
            "src_labels": src_tokens["input_ids"],
            "tgt_labels": tgt_tokens["input_ids"],
            "src_label_mask": src_tokens["attention_mask"],
            "tgt_label_mask": tgt_tokens["attention_mask"],
        }
