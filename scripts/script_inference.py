import torch
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int)
parser.add_argument('--dataset', type=str, default='ml-1m')
parser.add_argument('--temp_type', type=str, default='simple')
parser.add_argument('--K', nargs='+', type=int, default=[5, 10, 15, 20, 25, 30])
args = parser.parse_args()

MODEL_PATH = ""
LOG_DIR = f"log_inference_{args.dataset}"
PORT_ID = 15637
os.makedirs(LOG_DIR, exist_ok=True)

if torch.cuda.device_count() < args.num_gpu:
    print(f"Only {torch.cuda.device_count()} GPU available.")
    args.num_gpu = torch.cuda.device_count()


for K in args.K:
    fp = f"K{K}_{args.temp_type}"
    command = f"python -m torch.distributed.launch --nproc_per_node {args.num_gpu} --master_port {PORT_ID} inference.py "\
                f"--model_path {MODEL_PATH} "\
                f"--dataset {args.dataset} "\
                f"--K {K} "\
                f">> {LOG_DIR}/{fp}"
    subprocess.run(command, shell=True)
