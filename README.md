# Unsupervised Translation

## Getting started

0. (recommended) Create a virtual environment using `conda` or `virtualenv`
1. Install Python requirements: `python -m pip install -r requirements.txt`
2. Run the training script: `python src/supervised/train.py`

## Euler
(Euler is down for maintenance until October 24)

- Developing on Euler is easiest using `code-server`, which is a remotely hosted VSCode instance. To start the server, run `scripts/start_code_server.sh <your_nethz>@euler.ethz.ch`.
  - If you're starting the server for the first time, you need to manually log into Euler in order to find the code server password.
  - If your connection is interrupted, so is the connection to the VSCode server. To avoid starting a new job every time, use the dedicated script to reconnect to any running server: `scripts/reconnect_code_server.sh <your_nethz>@euler.ethz.ch`
- Running Python scripts on Euler GPUs can be done using `scripts/bsub_python.sh path/to/python_script.py`.