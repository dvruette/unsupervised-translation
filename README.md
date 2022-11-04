# Unsupervised Translation

## Getting started

0. (recommended) Create a virtual environment using `conda` or `virtualenv`
1. Install Python requirements: `python -m pip install -r requirements.txt`
2. Run the training script: `python src/supervised/train.py`

### Logging
We log our runs to Weights & Biases ([our workspace](https://wandb.ai/dvruette/unsupervised-translation/overview)).
To be able to submit runs on your own, you'll need to create an environment file (name it `.env`) and set the WandB API eky environment variable:
```bash
WANDB_API_KEY=<your_api_key>
```

## Euler

### Setup
(Note: You need to be connected to the ETH network in order to log into Euler)

- Log into euler through SSH: `ssh <your_nethz>@euler.ethz.ch`
- (recommended) Add an SSH key for not having to enter your password every time: https://scicomp.ethz.ch/wiki/Accessing_the_cluster#SSH_Keys
  - You can store your SSH key for the current session to not have to enter the SSH-key-password every time: `ssh-add ~/.ssh/<your_ssh_key>`. This needs to be done everytime you restart the computer or change users.

### Development
- Developing on Euler is easiest using `code-server`, which is a remotely hosted VSCode instance. To start the server, run `scripts/start_code_server.sh <your_nethz>@euler.ethz.ch`.
  - If you're starting the server for the first time, you need to manually log into Euler in order to find the code server password.
  - If your connection is interrupted, so is the connection to the VSCode server. To avoid starting a new job every time, use the dedicated script to reconnect to any running server: `scripts/reconnect_code_server.sh <your_nethz>@euler.ethz.ch`
- Running Python scripts on Euler GPUs can be done using `scripts/bsub_python.sh path/to/python_script.py`.