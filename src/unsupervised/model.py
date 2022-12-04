from typing import Optional, Literal

import evaluate
import math
import pytorch_lightning as pl
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


class Critic(nn.Module):
    def __init__(self, d_model: int, n_pools: int, d_proj: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_pools = n_pools
        self.d_proj = d_proj

        self.proj = nn.Linear(self.d_model, self.d_proj)
        self.fc = nn.Sequential(
            nn.Linear(self.n_pools * self.d_proj, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, x):
        # input shape: (batch_size, n_pools, d_model)
        # output shape: (batch_size,)
        x = self.proj(x)
        x = x.view(-1, self.n_pools * self.d_proj)
        x = self.fc(x)
        return x.squeeze(-1)


def _bleu_score(tokens, target, tokenizer):
    bleu = evaluate.load("bleu")
    xs = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    ys = tokenizer.batch_decode(target, skip_special_tokens=True)

    filtered = [(x.strip(), y.strip()) for x, y in zip(xs, ys) if x.strip() and y.strip()]
    if len(filtered) == 0:
        return {"bleu": 0}

    xs, ys = tuple(zip(*filtered))
    return bleu.compute(predictions=xs, references=ys)

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
        tokenizer_path_a: str = "bert-base-cased",
        tokenizer_path_b: str = "deepset/gbert-base",
        use_oracle: bool = True,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        pooling: Literal["mean", "max", "attention"] = "max",
        n_pools: int = 1,
        d_model: int = 512,
        n_codes: int = 1024,
        n_groups: int = 2,
        beta_critic: float = 1.0,
        lr_rec: float = 1e-4,
        lr_critic: float = 2e-5,
        lr_enc: float = 2e-5,
        critic_loss: Literal["wasserstein", "classifier"] = "wasserstein",
        n_critic_steps: int = 5,
        lr_schedule: Literal["constant", "cosine"] = "constant",
        lr_warmup_steps: int = 2000,
        lr_max_steps: int = 100000,
        bleu_eval_freq: int = 2048,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.use_oracle = use_oracle
        self.pooling = pooling
        self.n_pools = n_pools
        self.n_codes = n_codes
        self.n_groups = n_groups
        self.d_model = d_model
        self.beta_critic = beta_critic
        self.lr_rec = lr_rec
        self.lr_critic = lr_critic
        self.lr_enc = lr_enc
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_max_steps = lr_max_steps
        self.critic_loss = critic_loss
        self.n_critic_steps = n_critic_steps
        self.bleu_eval_freq = bleu_eval_freq

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

        self.critic = Critic(d_model, n_pools)

        self.cycle_loss = nn.MSELoss()


    def _encode(self, encoder: BertModel, pooling: nn.Module, *args, **kwargs):
        # get compute hidden states
        encoder_out = encoder(*args, **kwargs)

        # pool hidden states to get sentence embeddings
        h = encoder_out.last_hidden_state
        mask = kwargs.get("attention_mask", None)
        z = pooling(h, mask=mask)

        # layer norm
        z = self.lnorm(z)

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

        logits_a = self.decode_a(batch["input_ids"][:, :-1], z_a)
        logits_b = self.decode_b(batch["labels"][:, :-1], z_b)

        if self.critic_loss == "wasserstein":
            l_adv_a = self.critic(z_a).mean()
            l_adv_b = -self.critic(z_b).mean()
        elif self.critic_loss == "classifier":
            l_adv_a = F.binary_cross_entropy_with_logits(self.critic(z_a), torch.zeros_like(self.critic(z_a)))
            l_adv_b = F.binary_cross_entropy_with_logits(self.critic(z_b), torch.ones_like(self.critic(z_b)))
        l_adv = l_adv_a + l_adv_b

        # compute reconstruction loss
        l_rec_a = F.cross_entropy(logits_a.transpose(1, 2), batch["input_ids"][:, 1:], ignore_index=self.tokenizer_a.pad_token_id)
        l_rec_b = F.cross_entropy(logits_b.transpose(1, 2), batch["labels"][:, 1:], ignore_index=self.tokenizer_b.pad_token_id)
        l_rec = l_rec_a + l_rec_b
        # compute total loss
        loss = l_rec - self.beta_critic*l_adv

        return {
            "loss": loss,
            "l_rec": l_rec / 2,
            "l_rec_a": l_rec_a,
            "l_rec_b": l_rec_b,
            "l_adv": l_adv / 2,
            "l_adv_a": l_adv_a,
            "l_adv_b": l_adv_b,
        }

    def training_step(self, batch, batch_idx, optimizer_idx):
        with torch.no_grad():
            for param in self.critic.parameters():
                param.clamp_(-0.01, 0.01)

        if optimizer_idx == 0:
            # train autoencoders for reconstruction
            metrics = self.get_loss(batch)
            loss = metrics["loss"]
            for key, val in metrics.items():
                self.log(f"train.{key}", val, prog_bar=False)
        elif optimizer_idx == 1:
            # train critic to discriminate between z_a and z_b
            enc_a = self.encode_a(batch["input_ids"], attention_mask=batch["attention_mask_src"])
            enc_b = self.encode_b(batch["labels"], attention_mask=batch["attention_mask_tgt"])
            z_a, z_b = enc_a["z"], enc_b["z"]

            if self.critic_loss == "wasserstein":
                loss = self.critic(z_a).mean() - self.critic(z_b).mean()
            elif self.critic_loss == "classifier":
                loss = F.binary_cross_entropy_with_logits(self.critic(z_a), torch.zeros_like(self.critic(z_a))) + \
                       F.binary_cross_entropy_with_logits(self.critic(z_b), torch.ones_like(self.critic(z_b)))

            self.log(f"train.l_critic", loss, prog_bar=False)
        elif optimizer_idx == 2:
            # train only encoder to fool critic
            enc_a = self.encode_a(batch["input_ids"], attention_mask=batch["attention_mask_src"])
            enc_b = self.encode_b(batch["labels"], attention_mask=batch["attention_mask_tgt"])
            z_a, z_b = enc_a["z"], enc_b["z"]

            if self.critic_loss == "wasserstein":
                l_adv_a = self.critic(z_a).mean()
                l_adv_b = -self.critic(z_b).mean()
            elif self.critic_loss == "classifier":
                l_adv_a = F.binary_cross_entropy_with_logits(self.critic(z_a), torch.zeros_like(self.critic(z_a)))
                l_adv_b = F.binary_cross_entropy_with_logits(self.critic(z_b), torch.ones_like(self.critic(z_b)))
            loss = l_adv_a + l_adv_b

            metrics = {
                "l_adv": loss / 2,
                "l_adv_a": l_adv_a,
                "l_adv_b": l_adv_b,
            }
            for key, val in metrics.items():
                self.log(f"train.{key}", val, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # compute batch metrics
        metrics = self.get_loss(batch)

        # compute loss for translations
        enc_a = self.encode_a(batch["input_ids"], attention_mask=batch["attention_mask_src"])
        enc_b = self.encode_b(batch["labels"], attention_mask=batch["attention_mask_tgt"])
        z_a, z_b = enc_a["z"], enc_b["z"]
        logits_ab = self.decode_b(batch["labels"][:, :-1], z_a)
        logits_ba = self.decode_a(batch["input_ids"][:, :-1], z_b)
        l_rec_ab = F.cross_entropy(logits_ab.transpose(1, 2), batch["labels"][:, 1:], ignore_index=self.tokenizer_b.pad_token_id)
        l_rec_ba = F.cross_entropy(logits_ba.transpose(1, 2), batch["input_ids"][:, 1:], ignore_index=self.tokenizer_a.pad_token_id)
        metrics["l_rec_ab"] = l_rec_ab
        metrics["l_rec_ba"] = l_rec_ba

        # compute BLEU score
        if (
            self.bleu_eval_freq > 0 and
            self.global_step % self.bleu_eval_freq == 0 and
            self.global_step > 0 and
            self.global_rank == 0 and
            batch_idx < 16
        ):
            with torch.no_grad():
                beam_z_a = z_a.repeat_interleave(4, dim=0)
                beam_z_b = z_b.repeat_interleave(4, dim=0)
                x_hat_b = self.autoencoder_b.decoder.generate(
                    input_ids=batch["input_ids"][:,:1],
                    encoder_hidden_states=beam_z_a,
                    eos_token_id=self.tokenizer_b.sep_token_id,
                    max_new_tokens=64,
                    num_beams=4,
                )
                x_hat_a = self.autoencoder_a.decoder.generate(
                    input_ids=batch["labels"][:,:1],
                    encoder_hidden_states=beam_z_b,
                    eos_token_id=self.tokenizer_a.sep_token_id,
                    max_new_tokens=64,
                    num_beams=4,
                )
                bleu_score_a = _bleu_score(x_hat_a, batch["input_ids"], self.tokenizer_a)
                bleu_score_b = _bleu_score(x_hat_b, batch["labels"], self.tokenizer_b)

                scores = {
                    "bleu_ba": bleu_score_a["bleu"],
                    "bleu_ab": bleu_score_b["bleu"],
                    "bleu": (bleu_score_a["bleu"] + bleu_score_b["bleu"]) / 2,
                }
                self.log("eval", scores, prog_bar=True)

        # log metrics
        self.log("val", metrics, prog_bar=False, sync_dist=True)
        return metrics

    def configure_optimizers(self):
        optimizer_rec = torch.optim.AdamW(
            [
                {"params": self.autoencoder_a.parameters()},
                {"params": self.autoencoder_b.parameters()},
                {"params": self.lnorm.parameters()},
            ],
            lr=self.lr_rec,
            weight_decay=0.01,
        )
        optimizer_critic = torch.optim.RMSprop(self.critic.parameters(), lr=self.lr_critic)
        optimizer_enc = torch.optim.AdamW(
            [
                {"params": self.autoencoder_a.encoder.parameters()},
                {"params": self.autoencoder_b.encoder.parameters()},
            ],
            lr=self.lr_enc,
            weight_decay=0.01,
        )
        if self.lr_schedule == "constant":
            # linear warmup with constant learning rate
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer_rec,
                lambda t: min(1.0, (t+1) / max(1, self.lr_warmup_steps)),
            )
        elif self.lr_schedule == "cosine":
            # linear warmup with cosine decay to 0.1 * lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer_rec,
                lambda t: min(1, ((t+1) / max(1, self.lr_warmup_steps))) * (0.9 * (1 + math.cos(math.pi * t / self.lr_max_steps))/2 + 0.1),
            )
        else:
            raise ValueError(f"Unknown lr_schedule {self.lr_schedule}")
        return (
            {
                "optimizer": optimizer_rec,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
                "frequency": 1,
            },
            {
                "optimizer": optimizer_critic,
                "frequency": self.n_critic_steps,
            },
            {
                "optimizer": optimizer_enc,
                "frequency": 1,
            },
        )

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
