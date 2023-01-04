from typing import Dict, Optional, Tuple
import evaluate

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.text import bleu
import transformers
from transformers import BertConfig, BertModel, BertLMHeadModel, BertTokenizer

import wandb
import random

class CustomBertLMHeadModel(BertLMHeadModel):
    def prepare_inputs_for_generation(self, *args, encoder_hidden_states=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        inputs["encoder_hidden_states"] = encoder_hidden_states
        return inputs

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
    ):
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.decoder(
            decoder_input_ids,
            encoder_hidden_states=encoder_out.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        encoder_out = self.encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_hidden_states = encoder_out.last_hidden_state

        return self.decoder.generate(decoder_input_ids, encoder_hidden_states=encoder_hidden_states, **kwargs)


class SupervisedTranslation(pl.LightningModule):
    def __init__(self, autoencoder: AutoEncoder, bleu_eval_freq: int, num_beams: int = 4, lr: float = 1e-4 , encoder_tokenizer_path: str = "bert-base-cased", decoder_tokenizer_path: str = "deepset/gbert-base"):
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = autoencoder
        self.lr = lr
        self.bleu_eval_freq = bleu_eval_freq
        self.num_beams = num_beams
        self.src_tokenizer = BertTokenizer.from_pretrained(encoder_tokenizer_path)
        self.tgt_tokenizer = BertTokenizer.from_pretrained(decoder_tokenizer_path)

    def get_loss(self, batch):
        tgt_ids = batch["labels"]  # shape: (batch_size, tgt_seq_len + 1)
        decoder_input_ids = tgt_ids[:, :-1]  # don't include the last one
        labels = tgt_ids[:, 1:]  # shift decoder input by 1 to prevent cheating
        labels = -100*(labels == 0) + labels  # mask labels where there is padding

        output = self.autoencoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )

        logits = output.logits  # shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        # cross_entropy requires shapes (batch_size, vocab_size, seq_len) and (batch_size, seq_len)
        loss = F.cross_entropy(logits.transpose(1, 2), labels)
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log("train", {"loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = {}
        with torch.no_grad():
            if (self.global_step % self.bleu_eval_freq == 0 and self.global_step > 0):
                input_ids = torch.repeat_interleave(batch["input_ids"], self.num_beams, dim=0)
                attention_mask = torch.repeat_interleave(batch["attention_mask"], self.num_beams, dim=0)

                pred_tokens = self.autoencoder.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=batch["labels"][:,:1],
                    max_new_tokens=256,
                    num_beams=self.num_beams,
                    do_sample=False
                )

                inputs = self.src_tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                translations= self.tgt_tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

                filtered = [(x.strip(), y.strip()) for x, y in zip(inputs, translations) if x.strip() and y.strip()]

                bleu_score = 0.0
                if len(filtered) > 0:
                    xs, ys = tuple(zip(*filtered))
                    bleu = evaluate.load("bleu")
                    bleu_score = bleu.compute(predictions=xs, references=ys)["bleu"]
                      
                self.log("val", {"bleu": bleu_score}, prog_bar=False, sync_dist=True)

        loss = self.get_loss(batch)
        self.log("val", {"loss": loss}, prog_bar=False, sync_dist=True)
        return {"loss": loss} 

    def test_step(self, batch, batch_idx):
        print("Hello, test step")
        with torch.no_grad():
            input_ids = torch.repeat_interleave(batch["input_ids"], self.num_beams, dim=0)
            attention_mask = torch.repeat_interleave(batch["attention_mask"], self.num_beams, dim=0)

            print("Test Step", batch_idx)
            pred_tokens = self.autoencoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=batch["labels"][:,:1],
                max_new_tokens=256,
                num_beams=self.num_beams,
                do_sample=False
            )

            inputs = self.src_tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            translations= self.tgt_tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

            pred = [(x.strip(), y.strip()) for x, y in zip(inputs, translations) if x.strip() and y.strip()]

            self.log("test_step", batch_idx, on_step=True, prog_bar=True, logger=False)
            return pred
    

    def test_epoch_end(self, test_step_outputs):
        pred = [item for sublist in l for item in sublist]
        
        xs, ys = tuple(zip(*pred))
        bleu = evaluate.load("bleu")
        
        scores = bleu.compute(predictions=xs, references=ys)
        self.log(scores)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr, weight_decay=0.01)
        return optimizer


def get_model_and_tokenizer():
    encoder_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    decoder_tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
    encoder_config = BertConfig(
        vocab_size=encoder_tokenizer.vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
    )

    decoder_config = BertConfig(
        vocab_size=decoder_tokenizer.vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
    )
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True

    encoder = BertModel(encoder_config, add_pooling_layer=False)
    decoder = CustomBertLMHeadModel(decoder_config)
    autoencoder = AutoEncoder(encoder, decoder)
    return autoencoder, encoder_tokenizer, decoder_tokenizer
