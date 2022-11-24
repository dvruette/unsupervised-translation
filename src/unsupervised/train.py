import sys
import os
from multiprocessing import freeze_support

import hydra
import torch
import pytorch_lightning as pl
import dotenv
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset

# add the working directory to $PYTHONPATH
# needed to make local imports work
sys.path.append(os.getenv("PWD", "."))
# load the `.env` file
dotenv.load_dotenv()

from src.unsupervised.model import get_tokenizers
from src.unsupervised.model import UnsupervisedTranslation
from src.data import get_dataset, DataCollatorForUnsupervisedMT



@hydra.main(config_path="../config", config_name="unsupervised", version_base="1.1")
def main(config):
    logger = pl.loggers.wandb.WandbLogger(
        entity="getsellerie",
        project="unsupervised-translation",
        group="unsup-oracle"
    )
    # convert config object to python dict with `OmegaConf.to_container(...)`
    logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))

    # load model
    tokenizer_a, tokenizer_b = get_tokenizers()

    model = UnsupervisedTranslation(
        tokenizer_a.vocab_size,
        tokenizer_b.vocab_size,
        latent_regularizer=config.model.latent_regularizer,
        n_codes=config.model.vq.n_codes,
        n_groups=config.model.vq.n_groups,
        lr=config.training.learning_rate
    )

    ds = get_dataset(
        dataset_name=config.data.dataset_name,
        language_pair=config.data.language_pair,
        stream=config.data.stream,
    )
    train_ds = ds[config.data.train_split]
    val_ds = ds[config.data.val_split]

    collator = DataCollatorForUnsupervisedMT(tokenizer_a, tokenizer_b, max_seq_len=config.training.max_seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        collate_fn=collator,
        shuffle=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        collate_fn=collator,
        shuffle=False,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        strategy="dp" if torch.cuda.device_count() > 1 else None,
        # precision=16 if torch.cuda.is_available() else 32,
        max_steps=config.training.max_steps,
        max_epochs=config.training.max_epochs,
        logger=logger,
        accumulate_grad_batches=config.training.accumulate_batches,
        limit_val_batches=config.training.val.limit_batches,
        val_check_interval=config.training.val.check_interval,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    freeze_support()
    main()
