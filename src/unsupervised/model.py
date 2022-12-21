from typing import Optional, Literal

import evaluate
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertLMHeadModel, BertTokenizer

from src.unsupervised.vq_embed import VectorQuantizeEMA
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
        do_backtranslation: bool = True,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        pooling: Literal["mean", "max", "attention"] = "max",
        n_pools: int = 1,
        d_model: int = 512,
        n_codes: int = 1024,
        n_groups: int = 16,
        beta_critic: float = 0.25,
        beta_vq: float = 0.1,
        lr_rec: float = 1e-4,
        lr_critic: float = 2e-5,
        lr_bt: float = 1e-4,
        critic_loss: Literal["wasserstein", "classifier"] = "wasserstein",
        n_critic_steps: int = 5,
        bleu_eval_freq: int = 2048,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.do_backtranslation = do_backtranslation
        self.pooling = pooling
        self.n_pools = n_pools
        self.n_codes = n_codes
        self.n_groups = n_groups
        self.d_model = d_model
        self.beta_critic = beta_critic
        self.beta_vq = beta_vq
        self.lr_rec = lr_rec
        self.lr_critic = lr_critic
        self.lr_bt = lr_bt
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
        self.vq = VectorQuantizeEMA(d_model, n_codes, n_groups)
        self.critic = Critic(d_model, n_pools)


    def _encode(self, encoder: BertModel, pooling: nn.Module, *args, **kwargs):
        # get compute hidden states
        encoder_out = encoder(*args, **kwargs)
        # pool hidden states to get sentence embeddings
        h = encoder_out.last_hidden_state
        mask = kwargs.get("attention_mask", None)
        z = pooling(h, mask=mask)
        # layer norm
        z = self.lnorm(z)
        # vector quantize
        return self.vq(z)

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

    def cycle_loss(self, z, labels, generate_tgt, encode_tgt, decode_src, tokenizer_src, tokenizer_tgt):
        with torch.no_grad():
            y_hat = generate_tgt(
                input_ids=torch.zeros_like(labels[:, :1]).fill_(tokenizer_tgt.cls_token_id),
                encoder_hidden_states=z,
                eos_token_id=tokenizer_tgt.sep_token_id,
                max_new_tokens=64,
                num_beams=1,
                do_sample=False,
            )
        # generate attention mask for y_hat
        mask = (y_hat == tokenizer_tgt.sep_token_id).long()
        attention_mask = 1 - mask.cumsum(dim=-1) + mask
        z_hat = encode_tgt(y_hat, attention_mask=attention_mask).z_q
        logits = decode_src(labels[:, :-1], z_hat)
        l_cycle = F.cross_entropy(logits.transpose(1, 2), labels[:, 1:], ignore_index=tokenizer_src.pad_token_id)
        return l_cycle

    def get_loss(self, batch):
        enc_a = self.encode_a(batch["src_input_ids"], attention_mask=batch["src_attention_mask"])
        enc_b = self.encode_b(batch["tgt_input_ids"], attention_mask=batch["tgt_attention_mask"])
        z_a, z_b = enc_a.z_q, enc_b.z_q

        logits_a = self.decode_a(batch["src_labels"][:, :-1], z_a)
        logits_b = self.decode_b(batch["tgt_labels"][:, :-1], z_b)

        # l_adv = self.compute_critic_loss(batch, enc_a, enc_b)

        # compute reconstruction loss
        l_rec_a = F.cross_entropy(logits_a.transpose(1, 2), batch["src_labels"][:, 1:], ignore_index=self.tokenizer_a.pad_token_id)
        l_rec_b = F.cross_entropy(logits_b.transpose(1, 2), batch["tgt_labels"][:, 1:], ignore_index=self.tokenizer_b.pad_token_id)
        l_rec = l_rec_a + l_rec_b
        # compute total loss
        # loss = l_rec - self.beta_critic*l_adv + self.beta_vq*(enc_a.loss + enc_b.loss)
        loss = l_rec + self.beta_vq*(enc_a.loss + enc_b.loss)

        return {
            "loss": loss,
            "l_rec": l_rec / 2,
            "l_rec_a": l_rec_a,
            "l_rec_b": l_rec_b,
            "l_adv": l_adv / 2,
            "l_vq": (enc_a.loss + enc_b.loss) / 2,
            "l_vq_a": enc_a.loss,
            "l_vq_b": enc_b.loss,
            "entropy": (enc_a.entropy + enc_b.entropy) / 2 if self.training else None,
        }

    def compute_cycle_loss(self, batch):
        enc_a = self.encode_a(batch["src_labels"], attention_mask=batch["src_label_mask"])
        enc_b = self.encode_b(batch["tgt_labels"], attention_mask=batch["tgt_label_mask"])
        z_a, z_b = enc_a.z_q, enc_b.z_q

        l_adv = self.compute_critic_loss(batch, enc_a, enc_b)

        l_cycle_a = self.cycle_loss(
            z=z_a,
            labels=batch["src_labels"],
            generate_tgt=self.autoencoder_b.decoder.generate,
            encode_tgt=self.encode_b,
            decode_src=self.decode_a,
            tokenizer_src=self.tokenizer_a,
            tokenizer_tgt=self.tokenizer_b,
        )
        l_cycle_b = self.cycle_loss(
            z=z_b,
            labels=batch["tgt_labels"],
            generate_tgt=self.autoencoder_a.decoder.generate,
            encode_tgt=self.encode_a,
            decode_src=self.decode_b,
            tokenizer_src=self.tokenizer_b,
            tokenizer_tgt=self.tokenizer_a,
        )
        l_cycle = l_cycle_a + l_cycle_b - self.beta_critic*l_adv + self.beta_vq*(enc_a.loss + enc_b.loss)
        
        return {
            "loss": l_cycle,
            "l_cycle": l_cycle / 2,
            "l_cycle_a": l_cycle_a,
            "l_cycle_b": l_cycle_b,
            "l_adv": l_adv / 2,
            "l_vq": (enc_a.loss + enc_b.loss) / 2,
            "l_vq_a": enc_a.loss,
            "l_vq_b": enc_b.loss,
            "entropy": (enc_a.entropy + enc_b.entropy) / 2,
        }

    def compute_critic_loss(self, batch, enc_a=None, enc_b=None):
        self.vq.eval()
        if enc_a is None:
            enc_a = self.encode_a(batch["src_labels"], attention_mask=batch["src_label_mask"])
        if enc_b is None:
            enc_b = self.encode_b(batch["tgt_labels"], attention_mask=batch["tgt_label_mask"])
        z_a, z_b = enc_a.z_q, enc_b.z_q

        if self.critic_loss == "wasserstein":
            loss = self.critic(z_a).mean() - self.critic(z_b).mean()
        elif self.critic_loss == "classifier":
            loss = F.binary_cross_entropy_with_logits(self.critic(z_a), torch.zeros_like(self.critic(z_a))) + \
                    F.binary_cross_entropy_with_logits(self.critic(z_b), torch.ones_like(self.critic(z_b)))
        self.vq.train()
        return loss

    def compute_bleu(self, batch, z_a, z_b):
        with torch.no_grad():
            beam_z_a = z_a.repeat_interleave(4, dim=0)
            beam_z_b = z_b.repeat_interleave(4, dim=0)
            x_hat_b = self.autoencoder_b.decoder.generate(
                input_ids=batch["src_labels"][:,:1],
                encoder_hidden_states=beam_z_a,
                eos_token_id=self.tokenizer_b.sep_token_id,
                max_new_tokens=64,
                num_beams=4,
            )
            x_hat_a = self.autoencoder_a.decoder.generate(
                input_ids=batch["tgt_labels"][:,:1],
                encoder_hidden_states=beam_z_b,
                eos_token_id=self.tokenizer_a.sep_token_id,
                max_new_tokens=64,
                num_beams=4,
            )
            try:
                bleu_score_a = _bleu_score(x_hat_a, batch["src_labels"], self.tokenizer_a)
                bleu_score_b = _bleu_score(x_hat_b, batch["tgt_labels"], self.tokenizer_b)

                return {
                    "bleu_ba": bleu_score_a["bleu"],
                    "bleu_ab": bleu_score_b["bleu"],
                    "bleu": (bleu_score_a["bleu"] + bleu_score_b["bleu"]) / 2,
                }
            except FileNotFoundError as e:
                print("WARNING: Bleu score could not be computed because script was not found:", e)
                return {}

    def log_metrics(self, metrics, prefix=""):
        if prefix:
            prefix += "."
        for key, val in metrics.items():
            if val is not None:
                self.log(prefix + key, val, prog_bar=False)

    def training_step(self, batch, batch_idx, optimizer_idx):
        with torch.no_grad():
            for param in self.critic.parameters():
                param.clamp_(-0.01, 0.01)

        if optimizer_idx == 0:
            # train autoencoders for reconstruction
            metrics = self.get_loss(batch)
        elif optimizer_idx == 1:
            # train autoencoders on backtranslation
            metrics = self.compute_cycle_loss(batch)
        elif optimizer_idx == 2:
            # train critic to discriminate between z_a and z_b
            loss = self.compute_critic_loss(batch)
            metrics = {
                "loss": loss,
                "l_critic": loss,
            }

        loss = metrics["loss"]
        self.log_metrics(metrics, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        # compute batch metrics
        metrics = self.get_loss(batch)

        # compute loss for translations
        enc_a = self.encode_a(batch["src_labels"], attention_mask=batch["src_label_mask"])
        enc_b = self.encode_b(batch["tgt_labels"], attention_mask=batch["tgt_label_mask"])
        z_a, z_b = enc_a.z_q, enc_b.z_q
        logits_ab = self.decode_b(batch["tgt_labels"][:, :-1], z_a)
        logits_ba = self.decode_a(batch["src_labels"][:, :-1], z_b)
        l_rec_ab = F.cross_entropy(logits_ab.transpose(1, 2), batch["tgt_labels"][:, 1:], ignore_index=self.tokenizer_b.pad_token_id)
        l_rec_ba = F.cross_entropy(logits_ba.transpose(1, 2), batch["src_labels"][:, 1:], ignore_index=self.tokenizer_a.pad_token_id)
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
            scores = self.compute_bleu(batch, z_a, z_b)
            self.log_metrics(scores, prefix="eval")

        self.log_metrics(metrics, prefix="val")
        return metrics

    def configure_optimizers(self):
        model_params = [
            {"params": self.autoencoder_a.parameters()},
            {"params": self.autoencoder_b.parameters()},
            {"params": self.lnorm.parameters()},
            {"params": self.vq.parameters()},
        ]

        optimizer_rec = torch.optim.AdamW(model_params, lr=self.lr_rec, weight_decay=0.01)
        optimizer_bt = torch.optim.AdamW(model_params, lr=self.lr_bt, weight_decay=0.01)
        optimizer_critic = torch.optim.RMSprop(self.critic.parameters(), lr=self.lr_critic)
        
        return (
            {"optimizer": optimizer_rec, "frequency": 1},
            {"optimizer": optimizer_bt, "frequency": 1},
            {"optimizer": optimizer_critic, "frequency": self.n_critic_steps},
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
            encoder_hidden_states=enc.z_q,
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
