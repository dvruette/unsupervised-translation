import sys
import os
from multiprocessing import freeze_support

import hydra
import torch
import pytorch_lightning as pl
import dotenv
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# add the working directory to $PYTHONPATH
# needed to make local imports work
sys.path.append(os.getenv("PWD", "."))
# load the `.env` file
dotenv.load_dotenv()

from src.unsupervised.model import UnsupervisedTranslation
from src.data import get_dataset, DataCollatorForUnsupervisedMT


@hydra.main(config_path="../config", config_name="unsupervised", version_base="1.1")
def main(config):
    logger = pl.loggers.wandb.WandbLogger(
        entity="getsellerie",
        project="unsupervised-translation",
        group=config.group,
        # convert config object to python dict with `OmegaConf.to_container(...)`
        config={"config": OmegaConf.to_container(config, resolve=True)},
    )

    # load model

    model = UnsupervisedTranslation(
        tokenizer_path_a=config.data.tokenizer_path_a,
        tokenizer_path_b=config.data.tokenizer_path_b,
        do_backtranslation=config.training.do_backtranslation,
        pooling=config.model.pooling,
        n_pools=config.model.n_pools,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        lr_rec=config.training.optimizer.lr_rec,
        lr_critic=config.training.optimizer.lr_critic,
        lr_enc=config.training.optimizer.lr_enc,
        critic_loss=config.training.critic_loss,
        n_critic_steps=config.training.optimizer.n_critic_steps,
        lr_schedule=config.training.optimizer.schedule,
        lr_warmup_steps=config.training.optimizer.warmup_steps,
        lr_max_steps=config.training.optimizer.max_steps,
        beta_critic=config.training.beta_critic,
        beta_cycle=config.training.beta_cycle,
    )
    tokenizer_a, tokenizer_b = model.tokenizer_a, model.tokenizer_b

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

    # initialize callbacks: learning rate monitor, model checkpoint
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]

    if config.training.strategy == "ddp":
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
    else:
        strategy = config.training.strategy

    trainer = pl.Trainer(
        accelerator="auto",
        devices=config.training.devices if torch.cuda.is_available() else None,
        strategy=strategy if torch.cuda.device_count() > 1 else None,
        callbacks=callbacks,
        max_steps=config.training.max_steps,
        max_epochs=config.training.max_epochs,
        logger=logger,
        accumulate_grad_batches=config.training.accumulate_batches,
        limit_val_batches=config.training.val.limit_batches,
        val_check_interval=config.training.val.check_interval,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    freeze_support()
    main()
