import sys
import os
from multiprocessing import freeze_support

import hydra
import pytorch_lightning as pl
import dotenv
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# add the working directory to $PYTHONPATH
# needed to make local imports work
sys.path.append(os.getenv("PWD", "."))
# load the `.env` file 
dotenv.load_dotenv()

from src.supervised.model import get_model_and_tokenizer
from src.supervised.model import SupervisedTranslation
from src.data import get_dataset, DataCollatorForSupervisedMT


@hydra.main(config_path="../config", config_name="supervised", version_base="1.1")
def main(config):
    # load model
    autoencoder, src_tokenizer, tgt_tokenizer = get_model_and_tokenizer()

    model = SupervisedTranslation(autoencoder, lr=config.training.learning_rate)

    ds = get_dataset()
    train_ds = ds["train"].shuffle(buffer_size=4*config.training.batch_size)
    val_ds = ds["validation"]

    collator = DataCollatorForSupervisedMT(src_tokenizer, tgt_tokenizer)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        num_workers=train_ds.n_shards,
        collate_fn=collator,
        shuffle=False,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        num_workers=val_ds.n_shards,
        collate_fn=collator,
        shuffle=False,
    )

    logger = pl.loggers.wandb.WandbLogger(entity="getsellerie", project="unsupervised-translation", group="baseline")
    # convert config object to python dict with `OmegaConf.to_container(...)`
    logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))

    trainer = pl.Trainer(
        accelerator="auto",
        max_steps=config.training.max_steps,
        max_epochs=config.training.max_epochs,
        logger=logger,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    freeze_support()
    main()