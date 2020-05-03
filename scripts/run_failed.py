import wandb
import os
from tqdm import tqdm
import argparse

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

parser = argparse.ArgumentParser(description="re-run failed runs in a Weights and Biases project.")
parser.add_argument('--project', help='project in which to search for failed runs. for example: felixkreuk/unsupervised_segmentor')
parser.add_argument('--pybin', help='path to a python bin. this python bin will be use to re-run the failed runs (good for when using virtual env)')
parser.add_argument('--n_gpus', default=4, type=int, help='amount of gpus for re-run')
args = parser.parse_args()

api = wandb.Api()
runs = api.runs(args.project)

python_bin = args.pybin
num_gpus = args.n_gpus

cmds = []
failed = 0
for i, run in enumerate(runs):
    if run.state in ["failed", "crashed"]:
        failed += 1
        cmd = f"CUDA_VISIBLE_DEVICES={i%4} ts{i%4} {python_bin}"
        cfg = run.config
        for k,v in cfg.items():
            cmd += f" {k}={v}"
        
        print(cmd)
        print("-" * 90)

print(bcolors.OKBLUE + f"ran {failed} failed runs.")