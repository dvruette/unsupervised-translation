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
  - It is also recommended to change the Huggingface cache directory to the scratch storage (available in bulk but gets deleted after ~2 weeks) by adding `HF_HOME=${SCRATCH}/.cache/huggingface` to the `.env` file.
- Install the required packages. One way to do this (also possible e.g. via Conda):
  - Activate the python module: `module load gcc/6.3.0 python_gpu/3.8.5 cuda/11.1.1`
  - Install requirements `pip install -r requirements.txt`

### Development
- Developing on Euler is easiest using `code-server`, which is a remotely hosted VSCode instance. To start the server, run `scripts/start_code_server.sh <your_nethz>@euler.ethz.ch`.
  - If you're starting the server for the first time, you need to manually log into Euler in order to find the code server password: `cat ~/.config/code-server/config.yaml`
  - You might need to switch to the "new" software stack by running `set_software_stack.sh new` and relogging.
  - If your connection is interrupted, so is the connection to the VSCode server. To avoid starting a new job every time, use the dedicated script to reconnect to any running server: `scripts/reconnect_code_server.sh <your_nethz>@euler.ethz.ch`

### Submitting Runs
Running Python scripts on Euler GPUs can be done using the helper script:
```bash
scripts/bsub_python.sh path/to/python_script.py
```

Adding arguments to the script can be done as follows:
```bash
scripts/bsub_python.sh "src/supervised/train.py training.batch_size=8"
```
