import os, sys
import functools

import hydra
import wandb
from omegaconf import OmegaConf

# add the working directory to $PYTHONPATH
# needed to make local imports work
sys.path.append(os.getenv("PWD", "."))

from src.unsupervised.train import train

def run_sweep(config):
    wandb.init()
    sweep_config = OmegaConf.from_dotlist([f"{key[len('config.'):]}={val}" for key, val in wandb.config.items()])
    config = OmegaConf.merge(config, sweep_config)
    train(config)

@hydra.main(config_path="../config", config_name="unsupervised", version_base="1.1")
def main(config):
    run_fn = functools.partial(run_sweep, config=config)
    wandb.agent("getsellerie/unsupervised-translation/q0utfmat", function=run_fn, count=1)

if __name__ == "__main__":
    main()