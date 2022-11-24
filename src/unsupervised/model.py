from typing import Optional, Literal

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
        pooling: Literal["mean", "max"] = "max",
        latent_regularizer: Literal["lnorm", "vq", "none"] = "vq",
        d_model: int = 512,
        n_codes: int = 1024,
        n_groups: int = 2,
        lr: float = 1e-4,
        beta_cycle: float = 0.1,
    ):
        super().__init__()
        self.autoencoder_a = autoencoder_a
        self.autoencoder_b = autoencoder_b
        self.pooling = pooling
        self.latent_regularizer = latent_regularizer
        self.lr = lr
        self.beta_cycle = beta_cycle

        if self.latent_regularizer == "lnorm":
            self.lnorm = nn.LayerNorm(d_model)
        elif self.latent_regularizer == "vq":
            self.vq_embed = VectorQuantizeEMA(d_model, self.n_codes, self.n_groups)
        elif self.latent_regularizer != "none":
            raise ValueError(f"Unknown regularizer: {self.latent_regularizer}")

        self.save_hyperparameters(ignore=["autoencoder_a", "autoencoder_b"])

    def _encode(self, encoder, *args, **kwargs):
        # get compute hidden states
        encoder_out = encoder(*args, **kwargs)

        # pool hidden states to get sentence embeddings
        if self.pooling == "max":
            z = torch.max(encoder_out.last_hidden_state, dim=1, keepdim=True)[0]
        elif self.pooling == "mean":
            z = torch.mean(encoder_out.last_hidden_state, dim=1, keepdim=True)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        if self.latent_regularizer == "lnorm":
            # normalize sentence embeddings
            z = self.lnorm(z)
        elif self.latent_regularizer == "vq":
            # vector-quantize embeddings
            vq_z = self.vq_embed(z)
            self.log("entropy", vq_z["entropy"])
            self.log("usage", vq_z["avg_usage"])
            z = vq_z["z"]
        return z

    def _decode(self, decoder, input_ids, z, **kwargs):
        decoder_out = decoder(input_ids, encoder_hidden_states=z_a, **kwargs)
        return decoder_out.logits

    def encode_a(self, *args, **kwargs):
        return self._encode(self.autoencoder_a.encoder, *args, **kwargs)

    def encode_b(self, *args, **kwargs):
        return self._encode(self.autoencoder_b.encoder, *args, **kwargs)

    def decode_a(self, *args, **kwargs):
        return self._decode(self.autoencoder_a.decoder, *args, **kwargs)

    def decode_b(self, *args, **kwargs):
        return self._decode(self.autoencoder_b.decoder, *args, **kwargs)

    def get_loss(self, batch):
        z_a = self.encode_a(batch["input_ids"], attention_mask=batch["attention_mask_src"])
        z_b = self.encode_b(batch["labels"], attention_mask=batch["attention_mask_tgt"])

        logits_a = self.decode_a(batch["input_ids"][:, :-1], z_a)
        logits_b = self.decode_b(batch["labels"][:, :-1], z_b)

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
