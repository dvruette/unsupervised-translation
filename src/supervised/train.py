import sys
import os
from multiprocessing import freeze_support

import torch
import hydra
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
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
    train_ds = ds["train"].shuffle()
    val_ds = ds["validation"]

    collator = DataCollatorForSupervisedMT(src_tokenizer, tgt_tokenizer, max_seq_len=config.training.max_seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        collate_fn=collator,
        shuffle=False,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        collate_fn=collator,
        shuffle=False,
    )

    logger = WandbLogger(entity="getsellerie", project="unsupervised-translation", group="baseline")
    # convert config object to python dict with `OmegaConf.to_container(...)`
    logger.experiment.config.update(OmegaConf.to_container(config, resolve=True))

    trainer = Trainer(
        accelerator="auto",
        auto_select_gpus=True,
        accumulate_grad_batches=config.training.gradient_accumulation,
        strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else None,
        precision=16 if torch.cuda.is_available() else 32,
        max_steps=config.training.max_steps,
        max_epochs=config.training.max_epochs,
        logger=logger,
        limit_val_batches=config.training.val.limit_batches,
        val_check_interval=config.training.val.check_interval,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    freeze_support()
    main()
