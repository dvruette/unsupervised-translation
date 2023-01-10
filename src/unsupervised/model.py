from typing import Optional, Literal

import evaluate
import ot
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertLMHeadModel, BertTokenizer

from src.unsupervised.vq_embed import VectorQuantizeEMA, VQOutput
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
        assert x.shape[1] == self.n_pools
        assert x.shape[2] == self.d_model
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
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        pooling: Literal["mean", "max", "attention"] = "max",
        n_pools: int = 1,
        alignment: Literal["ot", "random", "identity"] = "ot",
        d_model: int = 384,
        n_heads: int = 6,
        beta_ot: float = 1e-3,
        beta_ce: float = 1e-4,
        lr: float = 1e-4,
        bleu_eval_freq: int = 2048,
        max_steps: int = 100_000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pooling_type = pooling
        self.alignment_type = alignment
        self.n_pools = n_pools
        self.d_model = d_model
        self.n_heads = n_heads
        self.beta_ot = beta_ot
        self.beta_ce = beta_ce
        self.lr = lr
        self.bleu_eval_freq = bleu_eval_freq
        self.max_steps = max_steps

        self.tokenizer_a, self.tokenizer_b = get_tokenizers(tokenizer_path_a, tokenizer_path_b)
        self.vocab_size_a = self.tokenizer_a.vocab_size
        self.vocab_size_b = self.tokenizer_b.vocab_size

        self.autoencoder_a = build_autoencoder(
            self.vocab_size_a,
            hidden_size=self.d_model,
            num_attention_heads=self.n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.autoencoder_b = build_autoencoder(
            self.vocab_size_b,
            hidden_size=self.d_model,
            num_attention_heads=self.n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        if self.pooling_type == "attention":
            self.pooling = AttentionPooling(self.d_model, self.n_pools)
        elif self.pooling_type == "max":
            self.pooling = MaxPooling(self.d_model, self.n_pools)
        elif self.pooling_type == "mean":
            self.pooling = MeanPooling(self.d_model, self.n_pools)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        self.lnorm = nn.LayerNorm(d_model)


    def _encode(self, encoder: BertModel, *args, **kwargs):
        # get compute hidden states
        encoder_out = encoder(*args, **kwargs)
        # pool hidden states to get sentence embeddings
        h = encoder_out.last_hidden_state
        mask = kwargs.get("attention_mask", None)
        z = self.pooling(h, mask=mask)
        # layer norm
        z = self.lnorm(z)
        return z

    def _decode(self, decoder: CustomBertLMHeadModel, input_ids: torch.Tensor, z: torch.Tensor, **kwargs):
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
                self.log(prefix + key, val, prog_bar=False, sync_dist=True)

    def _backtranslate(self, z, decoder_tgt, tokenizer_tgt):
        with torch.no_grad():
            input_ids = torch.zeros((z.size(0), 1), dtype=torch.int32, device=self.device).fill_(tokenizer_tgt.cls_token_id)
            y_hat = decoder_tgt.generate(
                input_ids=input_ids,
                encoder_hidden_states=z,
                eos_token_id=tokenizer_tgt.sep_token_id,
                max_new_tokens=64,
                num_beams=1,
                do_sample=False,
            )
        # generate attention mask for y_hat
        mask = (y_hat == tokenizer_tgt.sep_token_id).long()
        attention_mask = 1 - mask.cumsum(dim=-1) + mask
        return y_hat, attention_mask


    def compute_rec_loss(self, batch):
        enc_a = self.encode_a(batch["src_input_ids"], attention_mask=batch["src_attention_mask"])
        enc_b = self.encode_b(batch["tgt_input_ids"], attention_mask=batch["tgt_attention_mask"])
        z_a, z_b = enc_a, enc_b

        logits_a = self.decode_a(batch["src_labels"][:, :-1], z_a)
        logits_b = self.decode_b(batch["tgt_labels"][:, :-1], z_b)

        # compute reconstruction loss
        l_rec_a = F.cross_entropy(logits_a.transpose(1, 2), batch["src_labels"][:, 1:], ignore_index=self.tokenizer_a.pad_token_id)
        l_rec_b = F.cross_entropy(logits_b.transpose(1, 2), batch["tgt_labels"][:, 1:], ignore_index=self.tokenizer_b.pad_token_id)
        l_rec = l_rec_a + l_rec_b
        # compute total loss
        # loss = l_rec - self.beta_adv*l_adv + self.beta_vq*(enc_a.loss + enc_b.loss)
        loss = l_rec

        return {
            "loss": loss,
            "l_rec": l_rec / 2,
            "l_rec_a": l_rec_a,
            "l_rec_b": l_rec_b,
        }

    def compute_backtranslations(self, batch):
        self.vq.eval()
        enc_a = self.encode_a(batch["src_labels"], attention_mask=batch["src_label_mask"])
        enc_b = self.encode_b(batch["tgt_labels"], attention_mask=batch["tgt_label_mask"])
        self.vq.train()
        z_a, z_b = enc_a, enc_b

        y_hat_a, attention_mask_a = self._backtranslate(z_b, self.autoencoder_a.decoder, self.tokenizer_a)
        y_hat_b, attention_mask_b = self._backtranslate(z_a, self.autoencoder_b.decoder, self.tokenizer_b)
        return (z_a, y_hat_a, attention_mask_a), (z_b, y_hat_b, attention_mask_b)

    def get_loss(self, batch):
        batch_size = batch["src_input_ids"].size(0)
        z_a = self.encode_a(batch["src_input_ids"], attention_mask=batch["src_attention_mask"])
        z_b = self.encode_b(batch["tgt_input_ids"], attention_mask=batch["tgt_attention_mask"])
        shape_a = z_a.shape
        shape_b = z_b.shape

        z_a_flat = z_a.reshape(batch_size, -1)
        z_b_flat = z_b.reshape(batch_size, -1)
        dists = torch.cdist(z_a_flat, z_b_flat, p=2)**2 / (z_a_flat.size(-1) ** 0.5)
        
        if self.alignment_type == "ot":
            gamma = ot.emd(ot.unif(batch_size), ot.unif(batch_size), dists.cpu().detach().numpy())
            gamma = torch.from_numpy(gamma).to(self.device)
        elif self.alignment_type == "identity":
            gamma = torch.eye(batch_size, batch_size, device=self.device) / batch_size
        elif self.alignment_type == "random":
            idx = torch.randperm(batch_size, device=self.device)
            gamma = torch.eye(batch_size, batch_size, device=self.device)[idx] / batch_size
        else:
            raise ValueError(f"Unknown alignment type: {self.alignment_type}")

        src_ids, tgt_ids = torch.where(gamma > 0)
        # gather src and tgt embeddings
        src_embs = z_a_flat.gather(0, src_ids.unsqueeze(-1).expand(-1, z_a_flat.size(-1))).view(*shape_a)
        tgt_embs = z_b_flat.gather(0, tgt_ids.unsqueeze(-1).expand(-1, z_b_flat.size(-1))).view(*shape_b)
        # gather src and tgt labels
        src_labels = batch["src_labels"].gather(0, src_ids.unsqueeze(-1).expand(-1, batch["src_labels"].size(-1)))
        tgt_labels = batch["tgt_labels"].gather(0, tgt_ids.unsqueeze(-1).expand(-1, batch["tgt_labels"].size(-1)))
        # compute cross-entropy loss for src and tgt
        src_logits = self.decode_a(src_labels[:, :-1], tgt_embs)
        tgt_logits = self.decode_b(tgt_labels[:, :-1], src_embs)
        src_ce_loss = F.cross_entropy(src_logits.transpose(1, 2), src_labels[:, 1:], ignore_index=self.tokenizer_a.pad_token_id).sum(-1).mean()
        tgt_ce_loss = F.cross_entropy(tgt_logits.transpose(1, 2), tgt_labels[:, 1:], ignore_index=self.tokenizer_b.pad_token_id).sum(-1).mean()
        
        ce_loss = src_ce_loss + tgt_ce_loss
        ot_loss = (gamma * dists).sum()

        jdot_loss = self.beta_ot*ot_loss + self.beta_ce*ce_loss

        logits_a = self.decode_a(batch["src_labels"][:, :-1], z_a)
        logits_b = self.decode_b(batch["tgt_labels"][:, :-1], z_b)
        l_rec_a = F.cross_entropy(logits_a.transpose(1, 2), batch["src_labels"][:, 1:], ignore_index=self.tokenizer_a.pad_token_id).sum(-1).mean()
        l_rec_b = F.cross_entropy(logits_b.transpose(1, 2), batch["tgt_labels"][:, 1:], ignore_index=self.tokenizer_b.pad_token_id).sum(-1).mean()
        l_rec = l_rec_a + l_rec_b

        loss = l_rec + jdot_loss

        return {
            "loss": loss,
            "ot_loss": ot_loss,
            "ce_loss": ce_loss / 2,
            "src_ce_loss": src_ce_loss,
            "tgt_ce_loss": tgt_ce_loss,
            "l_rec": l_rec / 2,
            "l_rec_a": l_rec_a,
            "l_rec_b": l_rec_b,
        }

    def training_step(self, batch, batch_idx):
        metrics = self.get_loss(batch)
        self.log_metrics(metrics, prefix="train")
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        # compute batch metrics
        metrics = self.get_loss(batch)

        # compute loss for translations
        enc_a = self.encode_a(batch["src_labels"], attention_mask=batch["src_label_mask"])
        enc_b = self.encode_b(batch["tgt_labels"], attention_mask=batch["tgt_label_mask"])
        z_a, z_b = enc_a, enc_b
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        # cosine scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_steps, eta_min=self.lr/10)
        # schedule every step
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def _translate(
        self,
        autoencoder_src: AutoEncoder,
        autoencoder_tgt: AutoEncoder,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        enc = self._encode(autoencoder_src.encoder, input_ids, attention_mask)
        return autoencoder_tgt.decoder.generate(
            input_ids=decoder_input_ids,
            encoder_hidden_states=enc,
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
