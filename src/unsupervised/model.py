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


class UnsupervisedTranslation(pl.LightningModule):
    def __init__(self, autoencoder_a: AutoEncoder, autoencoder_b: AutoEncoder, lr: float = 1e-4):
        super().__init__()
        self.autoencoder_a = autoencoder_a
        self.autoencoder_b = autoencoder_b
        self.lr = lr

    def get_loss(self, batch):
        # tgt_ids = batch["labels"]  # shape: (batch_size, tgt_seq_len + 1)
        # decoder_input_ids = tgt_ids[:, :-1]  # don't include the last one
        # # labels = tgt_ids[:, 1:]  # shift decoder input by 1 to prevent cheating
        # # labels = -100*(labels == 0) + labels  # mask labels where there is padding

        # output = self.autoencoder(
        #     input_ids=batch["input_ids"],
        #     attention_mask=batch["attention_mask"],
        #     decoder_input_ids=decoder_input_ids,
        # )

        # get latent space of encoder and decoder for language a
        encoder_a_out = self.autoencoder_a.encoder(
            batch["input_ids"],
            attention_mask=batch["attention_mask_src"],
        )
        decoder_a_out = self.autoencoder_a.decoder(
            batch["input_ids"],
            encoder_hidden_states=encoder_a_out.last_hidden_state,
            encoder_attention_mask=batch["attention_mask_src"],
        )
        logits_a = decoder_a_out.logits 

        # get latent space of encoder and decoder for language b
        encoder_b_out = self.autoencoder_b.encoder(
            batch["labels"],
            attention_mask=batch["attention_mask_tgt"],
        )
        decoder_b_out = self.autoencoder_b.decoder(
            batch["labels"],
            encoder_hidden_states=encoder_b_out.last_hidden_state,
            encoder_attention_mask=batch["attention_mask_tgt"], 
        )
        logits_b = decoder_b_out.logits

        l_rec_a = F.cross_entropy(logits_a.transpose(1, 2), batch["input_ids"], ignore_index=0)
        l_rec_b = F.cross_entropy(logits_b.transpose(1, 2), batch["labels"], ignore_index=0)
        l_rec = l_rec_a + l_rec_b

        # logits = output.logits  # shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        # # cross_entropy requires shapes (batch_size, vocab_size, seq_len) and (batch_size, seq_len)
        # loss = F.cross_entropy(logits.transpose(1, 2), labels)

        # maxpool over the second dimension
        z_a = encoder_a_out.last_hidden_state
        z_a = torch.max(z_a, dim=1)[0]
        z_b = encoder_b_out.last_hidden_state
        z_b = torch.max(z_b, dim=1)[0]

        l_cycle = 2*F.mse_loss(z_a, z_b) # is equal to F.mse_loss(z_a, z_b) + F.mse_loss(z_b, z_a)

        loss = l_rec + l_cycle
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


def get_tokenizer():
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
    decoder = BertLMHeadModel(decoder_config)
    autoencoder = AutoEncoder(encoder, decoder)
    return autoencoder
