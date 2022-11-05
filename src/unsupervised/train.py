import sys
import os

import hydra
import pytorch_lightning as pl
import dotenv
from torch.utils.data import DataLoader

# add the working directory to $PYTHONPATH
# needed to make local imports work
sys.path.append(os.getenv("PWD", "."))
# load the `.env` file 
dotenv.load_dotenv()

from src.supervised.model import get_model_and_tokenizer
from src.supervised.model import SupervisedTranslation
from src.data import get_dataset, DataCollatorForSupervisedMT



@hydra.main(config_path="../config", config_name="supervised")
def main(config):
    # load model
    autoencoder, src_tokenizer, tgt_tokenizer = get_model_and_tokenizer()

    model = SupervisedTranslation(autoencoder, lr=config.training.learning_rate)

    ds = get_dataset()

    collator = DataCollatorForSupervisedMT(src_tokenizer, tgt_tokenizer)

    train_dl = DataLoader(
        ds["train"],
        batch_size=config.training.batch_size,
        # num_workers=config.data.num_workers,
        collate_fn=collator,
        shuffle=False,
    )

    trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model, train_dl)


if __name__ == "__main__":
    main()