import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertLMHeadModel, BertTokenizer

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


class SupervisedTranslation(pl.LightningModule):
    def __init__(self, autoencoder: AutoEncoder, lr: float = 1e-4):
        super().__init__()
        self.autoencoder = autoencoder
        self.lr = lr

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
        # self.log("train", {"loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr)
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
    decoder = BertLMHeadModel(decoder_config)
    autoencoder = AutoEncoder(encoder, decoder)
    return autoencoder, encoder_tokenizer, decoder_tokenizer