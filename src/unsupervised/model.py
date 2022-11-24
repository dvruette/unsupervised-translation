from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertLMHeadModel, BertTokenizer

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
            input_ids,
            attention_mask=attention_mask,
        )
        return self.decoder(
            decoder_input_ids,
            encoder_hidden_states=encoder_out.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )


class UnsupervisedTranslation(pl.LightningModule):
    def __init__(
        self,
        autoencoder_a: AutoEncoder,
        autoencoder_b: AutoEncoder,
        lr: float = 1e-4,
        beta_cycle: float = 0.1,
    ):
        super().__init__()
        self.autoencoder_a = autoencoder_a
        self.autoencoder_b = autoencoder_b
        self.lr = lr
        self.beta_cycle = beta_cycle

        self.save_hyperparameters(ignore=["autoencoder_a", "autoencoder_b"])

    def get_loss(self, batch):
        # get compute hidden states
        encoder_a_out = self.autoencoder_a.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask_src"],
        )
        encoder_b_out = self.autoencoder_b.encoder(
            batch["labels"],
            attention_mask=batch["attention_mask_tgt"],
        )

        # pool hidden states to get sentence embeddings
        z_a = torch.max(encoder_a_out.last_hidden_state, dim=1, keepdim=True)[0]
        z_b = torch.max(encoder_b_out.last_hidden_state, dim=1, keepdim=True)[0]

        # get decoder outputs
        decoder_a_out = self.autoencoder_a.decoder(
            batch["input_ids"][:, :-1],
            encoder_hidden_states=z_a,
        )
        decoder_b_out = self.autoencoder_b.decoder(
            batch["labels"][:, :-1],
            encoder_hidden_states=z_b,
        )
        logits_a = decoder_a_out.logits
        logits_b = decoder_b_out.logits

        # compute reconstruction loss
        l_rec_a = F.cross_entropy(logits_a.transpose(1, 2), batch["input_ids"][:, 1:], ignore_index=0)
        l_rec_b = F.cross_entropy(logits_b.transpose(1, 2), batch["labels"][:, 1:], ignore_index=0)
        l_rec = l_rec_a + l_rec_b
        # compute cycle consistency loss
        l_cycle = 2*F.mse_loss(z_a, z_b) # is equal to F.mse_loss(z_a, z_b) + F.mse_loss(z_b, z_a)
        # compute total loss
        loss = l_rec + self.beta_cycle*l_cycle
        return {
            "loss": loss,
            "l_rec": l_rec,
            "l_cycle": l_cycle,
        }


    def training_step(self, batch, batch_idx):
        metrics = self.get_loss(batch)
        self.log("train", metrics, prog_bar=False)
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.get_loss(batch)
        self.log("val", metrics, prog_bar=False)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr)
        return optimizer

    def translate_from_a_to_b(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self._translate(
            self.autoencoder_a,
            self.autoencoder_b,
            input_ids,
            attention_mask,
            decoder_input_ids,
            **kwargs,
        )

    def translate_from_b_to_a(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self._translate(
            self.autoencoder_b,
            self.autoencoder_a,
            input_ids,
            attention_mask,
            decoder_input_ids,
            **kwargs,
        )

    def _translate(
        self,
        autoencoder_src: AutoEncoder,
        autoencoder_tgt: AutoEncoder,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        encoder_hidden_states = autoencoder_src.encoder(
            input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        return autoencoder_tgt.decoder.generate(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            **kwargs,
        )


def get_tokenizers():
    encoder_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    decoder_tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

    return encoder_tokenizer, decoder_tokenizer

def get_model(vocab_size):
    encoder_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
    )

    decoder_config = BertConfig(
        vocab_size=vocab_size,
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
    return autoencoder
