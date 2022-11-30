from typing import Optional, Literal

import pytorch_lightning as pl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertLMHeadModel, BertTokenizer

from src.unsupervised.vq_vae import VectorQuantizeEMA
from src.unsupervised.pooling import AttentionPooling, MaxPooling, MeanPooling


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


class CosineLoss(nn.Module):
    def __init__(self, dim=1, reduction="mean", eps=1e-8):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.eps = eps

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction: {reduction}")

    def forward(self, x, y):
        loss = 1 - F.cosine_similarity(x, y, dim=self.dim, eps=self.eps)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class UnsupervisedTranslation(pl.LightningModule):
    def __init__(
        self,
        tokenizer_path_a: str = "bert-base-cased",
        tokenizer_path_b: str = "deepset/gbert-base",
        use_oracle: bool = True,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        pooling: Literal["mean", "max", "attention"] = "max",
        n_pools: int = 1,
        latent_regularizer: Literal["vq", "norm", "norm+vq", "none"] = "none",
        distance_metric: Literal["cosine", "l2"] = "l2",
        use_latent_projection: bool = False,
        d_model: int = 512,
        n_codes: int = 1024,
        n_groups: int = 2,
        beta_cycle: float = 0.1,
        beta_vq: float = 0.1,
        beta_cycle_warmup_steps: int = 1000,
        beta_vq_warmup_steps: int = 100,
        lr: float = 1e-4,
        lr_schedule: Literal["constant", "cosine"] = "constant",
        lr_warmup_steps: int = 2000,
        lr_max_steps: int = 100000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.use_oracle = use_oracle
        self.pooling = pooling
        self.n_pools = n_pools
        self.latent_regularizer = latent_regularizer
        self.distance_metric = distance_metric
        self.use_latent_projection = use_latent_projection
        self.n_codes = n_codes
        self.n_groups = n_groups
        self.d_model = d_model
        self.beta_cycle = beta_cycle
        self.beta_vq = beta_vq
        self.beta_cycle_warmup_steps = beta_cycle_warmup_steps
        self.beta_vq_warmup_steps = beta_vq_warmup_steps
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_max_steps = lr_max_steps

        self.tokenizer_a, self.tokenizer_b = get_tokenizers(tokenizer_path_a, tokenizer_path_b)
        self.vocab_size_a = self.tokenizer_a.vocab_size
        self.vocab_size_b = self.tokenizer_b.vocab_size

        self.autoencoder_a = build_autoencoder(
            self.vocab_size_a,
            hidden_size=self.d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.autoencoder_b = build_autoencoder(
            self.vocab_size_b,
            hidden_size=self.d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        if self.pooling == "attention":
            self.pooling_a = AttentionPooling(self.d_model, self.n_pools)
            self.pooling_b = AttentionPooling(self.d_model, self.n_pools)
        elif self.pooling == "max":
            self.pooling_a = MaxPooling(self.d_model, self.n_pools)
            self.pooling_b = MaxPooling(self.d_model, self.n_pools)
        elif self.pooling == "mean":
            self.pooling_a = MeanPooling(self.d_model, self.n_pools)
            self.pooling_b = MeanPooling(self.d_model, self.n_pools)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling}")


        self.lnorm = nn.LayerNorm(d_model)

        if self.use_latent_projection:
            self.proj_a_to_b = nn.Linear(d_model, d_model)
            self.proj_b_to_a = nn.Linear(d_model, d_model)
        else:
            self.proj_a_to_b = nn.Identity()
            self.proj_b_to_a = nn.Identity()

        if "vq" in self.latent_regularizer:
            self.vq_embed = VectorQuantizeEMA(d_model, self.n_codes, self.n_groups)
        if self.latent_regularizer not in ["norm", "lnorm", "vq", "norm+vq", "lnorm+vq", "none"]:
            raise ValueError(f"Unknown regularizer: {self.latent_regularizer}")

        if self.distance_metric == "cosine":
            self.cycle_loss = CosineLoss(dim=-1)
        elif self.distance_metric == "l2":
            self.cycle_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")


    def _beta_cycle(self, step):
        if self.beta_cycle_warmup_steps == 0:
            return self.beta_cycle
        return min(1.0, (step + 1) / self.beta_cycle_warmup_steps) * self.beta_cycle

    def _beta_vq(self, step):
        if self.beta_vq_warmup_steps == 0:
            return self.beta_vq
        return min(1.0, (step + 1) / self.beta_vq_warmup_steps) * self.beta_vq

    def _encode(self, encoder: BertModel, pooling: nn.Module, *args, **kwargs):
        # get compute hidden states
        encoder_out = encoder(*args, **kwargs)

        # pool hidden states to get sentence embeddings
        h = encoder_out.last_hidden_state
        mask = kwargs.get("attention_mask", None)
        z = pooling(h, mask=mask)

        # layer norm
        z = self.lnorm(z)

        if "norm" in self.latent_regularizer.split("+"):
            z = F.normalize(z, dim=-1)

        if "vq" in self.latent_regularizer.split("+"):
            # vector-quantize embeddings
            vq_z = self.vq_embed(z)
            vq_z["z_pre"] = z
            return vq_z
        else:
            return {"z_pre": z, "z": z}

    def _decode(self, decoder: CustomBertLMHeadModel, input_ids: torch.Tensor, z: torch.Tensor, **kwargs):
        decoder_out = decoder(input_ids, encoder_hidden_states=z, **kwargs)
        return decoder_out.logits

    def encode_a(self, *args, **kwargs):
        return self._encode(self.autoencoder_a.encoder, self.pooling_a, *args, **kwargs)

    def encode_b(self, *args, **kwargs):
        return self._encode(self.autoencoder_b.encoder, self.pooling_b, *args, **kwargs)

    def decode_a(self, *args, **kwargs):
        return self._decode(self.autoencoder_a.decoder, *args, **kwargs)

    def decode_b(self, *args, **kwargs):
        return self._decode(self.autoencoder_b.decoder, *args, **kwargs)

    def get_loss(self, batch):
        enc_a = self.encode_a(batch["input_ids"], attention_mask=batch["attention_mask_src"])
        enc_b = self.encode_b(batch["labels"], attention_mask=batch["attention_mask_tgt"])
        z_a, z_b = enc_a["z"], enc_b["z"]
        
        if self.use_latent_projection:
            enc_ab = self.vq_embed(self.proj_a_to_b(enc_a["z_pre"]))
            enc_ba = self.vq_embed(self.proj_b_to_a(enc_b["z_pre"]))
            z_ab = enc_ab["z"]
            z_ba = enc_ba["z"]
        else:
            z_ab = z_a
            z_ba = z_b

        # logits_a = self.decode_a(batch["input_ids"][:, :-1], z_a)
        # logits_b = self.decode_b(batch["labels"][:, :-1], z_b)

        # for testing: learn to translate with labels
        logits_a = self.decode_a(batch["input_ids"][:, :-1], z_b)
        logits_b = self.decode_b(batch["labels"][:, :-1], z_a)

        if not self.use_oracle:
            # set models to evalution mode
            self.autoencoder_a.eval()
            self.autoencoder_b.eval()
            with torch.no_grad():
                x_hat_b = self.autoencoder_b.decoder.generate(
                    input_ids=batch["input_ids"][:,:1],
                    encoder_hidden_states=z_ab,
                    max_new_tokens=128-1,
                )
                x_hat_a = self.autoencoder_a.decoder.generate(
                    input_ids=batch["labels"][:,:1],
                    encoder_hidden_states=z_ba,
                    max_new_tokens=128-1,
                )
            # set models back to training mode
            if self.training:
                self.autoencoder_a.train()
                self.autoencoder_b.train()

            # TODO: generate attention mask for generated tokens
            enc_hat_a = self.encode_a(x_hat_a)
            enc_hat_b = self.encode_b(x_hat_b)
            z_hat_a, z_hat_b = enc_hat_a["z"], enc_hat_b["z"]

            # compute cycle consistency loss
            l_cycle_a = self.cycle_loss(z_ab, z_hat_b)
            l_cycle_b = self.cycle_loss(z_ba, z_hat_a)

        else:
            # compute cycle consistency loss
            l_cycle_a = self.cycle_loss(z_ab, z_b)
            l_cycle_b = self.cycle_loss(z_ba, z_a)
        l_cycle = l_cycle_a + l_cycle_b

        # compute reconstruction loss
        l_rec_a = F.cross_entropy(logits_a.transpose(1, 2), batch["input_ids"][:, 1:], ignore_index=self.tokenizer_a.pad_token_id)
        l_rec_b = F.cross_entropy(logits_b.transpose(1, 2), batch["labels"][:, 1:], ignore_index=self.tokenizer_b.pad_token_id)
        l_rec = l_rec_a + l_rec_b
        # compute total loss
        loss = l_rec + self._beta_cycle(self.global_step) * l_cycle
        # add VQ loss if applicable
        if "loss" in enc_a:
            loss += self._beta_vq(self.global_step) * (enc_a["loss"] + enc_b["loss"])

        metrics = {}
        if "loss" in enc_a:
            metrics["l_vq"] = (enc_a["loss"] + enc_b["loss"]) / 2
            metrics["l_vq_a"] = enc_a["loss"]
            metrics["l_vq_b"] = enc_b["loss"]
        if "entropy" in enc_a:
            metrics["entropy"] = (enc_a["entropy"] + enc_b["entropy"]) / 2
            metrics["entropy_a"] = enc_a["entropy"]
            metrics["entropy_b"] = enc_b["entropy"]
        if "avg_usage" in enc_a:
            metrics["avg_usage"] = (enc_a["avg_usage"] + enc_b["avg_usage"]) / 2
            metrics["avg_usage_a"] = enc_a["avg_usage"]
            metrics["avg_usage_b"] = enc_b["avg_usage"]

        return {
            "loss": loss,
            "l_rec": l_rec / 2,
            "l_rec_a": l_rec_a,
            "l_rec_b": l_rec_b,
            "l_cycle": l_cycle / 2,
            "l_cycle_a": l_cycle_a,
            "l_cycle_b": l_cycle_b,
            **metrics,
        }

    def training_step(self, batch, batch_idx):
        metrics = self.get_loss(batch)
        self.log("train", metrics, prog_bar=False, sync_dist=True)
        return metrics

    def validation_step(self, batch, batch_idx):
        # compute batch metrics
        metrics = self.get_loss(batch)
        # also compute loss for translations
        enc_a = self.encode_a(batch["input_ids"], attention_mask=batch["attention_mask_src"])
        enc_b = self.encode_b(batch["labels"], attention_mask=batch["attention_mask_tgt"])
        z_a, z_b = enc_a["z"], enc_b["z"]
        logits_ab = self.decode_b(batch["labels"][:, :-1], z_a)
        logits_ba = self.decode_a(batch["input_ids"][:, :-1], z_b)
        l_rec_ab = F.cross_entropy(logits_ab.transpose(1, 2), batch["labels"][:, 1:], ignore_index=self.tokenizer_b.pad_token_id)
        l_rec_ba = F.cross_entropy(logits_ba.transpose(1, 2), batch["input_ids"][:, 1:], ignore_index=self.tokenizer_a.pad_token_id)
        metrics["l_rec_ab"] = l_rec_ab
        metrics["l_rec_ba"] = l_rec_ba
        # log metrics
        self.log("val", metrics, prog_bar=False, sync_dist=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01,
        )
        if self.lr_schedule == "constant":
            # linear warmup with constant learning rate
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda t: min(1.0, (t+1) / max(1, self.lr_warmup_steps)),
            )
        elif self.lr_schedule == "cosine":
            # linear warmup with cosine decay to 0.1 * lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda t: min(1, ((t+1) / max(1, self.lr_warmup_steps))) * (0.9 * (1 + math.cos(math.pi * t / self.lr_max_steps))/2 + 0.1),
            )
        else:
            raise ValueError(f"Unknown lr_schedule {self.lr_schedule}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def _translate(
        self,
        autoencoder_src: AutoEncoder,
        autoencoder_tgt: AutoEncoder,
        pooling_src: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        enc = self._encode(autoencoder_src.encoder, pooling_src, input_ids, attention_mask)
        return autoencoder_tgt.decoder.generate(
            input_ids=decoder_input_ids,
            encoder_hidden_states=enc["z"],
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
            self.pooling_a,
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
            self.pooling_b,
            input_ids,
            attention_mask,
            decoder_input_ids,
            **kwargs,
        )
