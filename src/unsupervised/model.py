from typing import Optional, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertLMHeadModel, BertTokenizer

from src.unsupervised.vq_vae import VectorQuantizeEMA


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

def get_tokenizers(path_a="bert-base-cased", path_b="deepset/gbert-base"):
    tokenizer_a = BertTokenizer.from_pretrained(path_a)
    tokenizer_b = BertTokenizer.from_pretrained(path_b)

    return tokenizer_a, tokenizer_b

def build_autoencoder(
    vocab_size: int,
    hidden_size: int = 512,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    num_attention_heads: int = 8,
    intermediate_size: int = 2048,
    max_position_embeddings: int = 512,
):
    encoder_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_encoder_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )

    decoder_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_decoder_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True

    encoder = BertModel(encoder_config, add_pooling_layer=False)
    decoder = CustomBertLMHeadModel(decoder_config)
    autoencoder = AutoEncoder(encoder, decoder)
    return autoencoder

class UnsupervisedTranslation(pl.LightningModule):
    def __init__(
        self,
        vocab_size_a: int,
        vocab_size_b: int,
        use_oracle: bool = False,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        pooling: Literal["mean", "max"] = "max",
        latent_regularizer: Literal["lnorm", "vq", "lnorm+vq", "none"] = "vq",
        d_model: int = 512,
        n_codes: int = 1024,
        n_groups: int = 2,
        lr: float = 1e-4,
        beta_cycle: float = 0.1,
        beta_vq: float = 0.1,
    ):
        super().__init__()
        self.vocab_size_a = vocab_size_a
        self.vocab_size_b = vocab_size_b
        self.use_oracle = use_oracle
        self.pooling = pooling
        self.latent_regularizer = latent_regularizer
        self.n_codes = n_codes
        self.n_groups = n_groups
        self.d_model = d_model
        self.lr = lr
        self.beta_cycle = beta_cycle
        self.beta_vq = beta_vq

        self.autoencoder_a = build_autoencoder(
            vocab_size_a,
            hidden_size=self.d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.autoencoder_b = build_autoencoder(
            vocab_size_b,
            hidden_size=self.d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        if self.latent_regularizer in ["lnorm", "lnorm+vq"]:
            self.lnorm = nn.LayerNorm(d_model)
        if self.latent_regularizer in ["vq", "lnorm+vq"]:
            self.vq_embed = VectorQuantizeEMA(d_model, self.n_codes, self.n_groups)
        if self.latent_regularizer not in ["lnorm", "lnorm+vq", "none"]:
            raise ValueError(f"Unknown regularizer: {self.latent_regularizer}")

        self.save_hyperparameters()

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

        if self.latent_regularizer in ["lnorm", "lnorm+vq"]:
            # normalize sentence embeddings
            z = self.lnorm(z)

        if self.latent_regularizer in ["vq", "lnorm+vq"]:
            # vector-quantize embeddings
            vq_z = self.vq_embed(z)
            return vq_z
        else:
            return {"z": z}

    def _decode(self, decoder, input_ids, z, **kwargs):
        decoder_out = decoder(input_ids, encoder_hidden_states=z, **kwargs)
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
        enc_a = self.encode_a(batch["input_ids"], attention_mask=batch["attention_mask_src"])
        enc_b = self.encode_b(batch["labels"], attention_mask=batch["attention_mask_tgt"])
        z_a, z_b = enc_a["z"], enc_b["z"]

        logits_a = self.decode_a(batch["input_ids"][:, :-1], z_a)
        logits_b = self.decode_b(batch["labels"][:, :-1], z_b)

        if not self.use_oracle:
            # set models to evalution mode
            self.autoencoder_a.eval()
            self.autoencoder_b.eval()
            with torch.no_grad():
                x_hat_b = self.autoencoder_b.decoder.generate(
                    input_ids=batch["input_ids"][:,:1],
                    encoder_hidden_states=z_a,
                    max_new_tokens=128-1,
                )
                x_hat_a = self.autoencoder_a.decoder.generate(
                    input_ids=batch["labels"][:,:1],
                    encoder_hidden_states=z_b,
                    max_new_tokens=128-1,
                )
            # set models back to training mode
            if self.train:
                self.autoencoder_a.train()
                self.autoencoder_b.train()

            # TODO generate attention mask for generated tokens
            enc_hat_a = self.encode_a(x_hat_a)
            enc_hat_b = self.encode_b(x_hat_b)
            z_hat_a, z_hat_b = enc_hat_a["z"], enc_hat_b["z"]

            # compute cycle consistency loss
            l_cycle = F.mse_loss(z_a, z_hat_b)+F.mse_loss(z_b, z_hat_a)

        else:
            # compute cycle consistency loss
            l_cycle = 2*F.mse_loss(z_a, z_b) # is equal to F.mse_loss(z_a, z_b) + F.mse_loss(z_b, z_a)

        # compute reconstruction loss
        l_rec_a = F.cross_entropy(logits_a.transpose(1, 2), batch["input_ids"][:, 1:], ignore_index=0)
        l_rec_b = F.cross_entropy(logits_b.transpose(1, 2), batch["labels"][:, 1:], ignore_index=0)
        l_rec = l_rec_a + l_rec_b
        # compute total loss
        loss = l_rec + self.beta_cycle*l_cycle
        # add VQ loss if applicable
        if "loss" in enc_a:
            loss += self.beta_vq * (enc_a["loss"] + enc_b["loss"])

        metrics = {}
        if "loss" in enc_a:
            metrics["l_vq"] = (enc_a["loss"] + enc_b["loss"]) / 2
        if "entropy" in enc_a:
            metrics["entropy"] = (enc_a["entropy"] + enc_b["entropy"]) / 2
        if "avg_usage" in enc_a:
            metrics["avg_usage"] = (enc_a["avg_usage"] + enc_b["avg_usage"]) / 2

        return {
            "loss": loss,
            "l_rec": l_rec / 2,
            "l_cycle": l_cycle / 2,
            **metrics,
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
