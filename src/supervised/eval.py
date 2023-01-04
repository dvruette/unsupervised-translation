import sys
import os
from multiprocessing import freeze_support

import torch
import hydra

import pytorch_lightning as pl
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


@hydra.main(config_path="../config/supervised/eval", config_name="de-en", version_base="1.1")
def main(config):
    # logger = WandbLogger(
    #     entity="getsellerie",
    #     project="unsupervised-translation",
    #     group="baseline-eval",
    #     config=OmegaConf.to_container(config, resolve=True)
    # )

    autoencoder, src_tokenizer, tgt_tokenizer = get_model_and_tokenizer()

    model = SupervisedTranslation(autoencoder, bleu_eval_freq=config.training.val.bleu_eval_freq, num_beams=config.training.num_beams, lr=config.training.learning_rate)

    ds = get_dataset(
        dataset_name=config.data.dataset_name,
        language_pair=config.data.language_pair,
        stream=config.data.stream,
    )

    ds = ds["test"]

    collator = DataCollatorForSupervisedMT(src_tokenizer, tgt_tokenizer, max_seq_len=config.training.max_seq_len)

    dl = DataLoader(
        ds,
        batch_size=config.training.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=config.data.num_workers
    )

    trainer = Trainer(
        accelerator="cpu",
        gpus=1,
        num_nodes=1,
        precision=16 if torch.cuda.is_available() else 32,
    )

    trainer.test(model, ckpt_path=config.checkpoint_path, dataloaders=dl)

    # trainer = Trainer(
    #     accelerator="gpu",
    #     auto_select_gpus=True,
    #     num_nodes=1,
    #     callbacks = callbacks,
    #     accumulate_grad_batches=config.training.gradient_accumulation,
    #     strategy=strategy if torch.cuda.device_count() > 1 else None,
    #     precision=16 if torch.cuda.is_available() else 32,
    #     max_steps=config.training.max_steps,
    #     max_epochs=config.training.max_epochs,
    #     logger=logger,
    #     limit_val_batches=config.training.val.limit_batches,
    #     val_check_interval=config.training.val.check_interval,
    # )

    # trainer.fit(model, train_dl, val_dl)

if __name__ == "__main__":
    freeze_support()
    main()
