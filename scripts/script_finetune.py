import torch
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int)
parser.add_argument('--dataset', type=str, default='ml-1m')
parser.add_argument('--train_type', type=str, default='mixed')
parser.add_argument('--test_type', type=str, default='sequential')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--total_batch_size', type=int, default=256)
parser.add_argument('--lr', nargs='+', type=float, default=[1e-3])
parser.add_argument('--K', nargs='+', type=int, default=[5, 10, 15, 20, 25, 30])
parser.add_argument('--train_size', nargs='+', type=int, default=[256, 512, 1024, 2048, 4096, 8192])
args = parser.parse_args()

MODEL_PATH = ""
LOG_DIR = f"log_finetune_{args.dataset}"
OUTPUT_DIR = f"lora-Vicuna_finetune_{args.dataset}"
PORT_ID = 15637
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.cuda.device_count() < args.num_gpu:
    print(f"Only {torch.cuda.device_count()} GPU available.")
    args.num_gpu = torch.cuda.device_count()


for lr in args.lr:
    for K in args.K:
        for train_size in args.train_size:
            fp = f"lr{lr}_K{K}_ts{train_size}_bs{args.total_batch_size}_{args.train_type}_{args.test_type}_epoch{args.epochs}"
            command = f"python -m torch.distributed.launch --nproc_per_node {args.num_gpu} --master_port {PORT_ID} finetune.py "\
                        f"--model_path {MODEL_PATH} "\
                        f"--output_path {OUTPUT_DIR}/{fp} "\
                        f"--dataset {args.dataset} "\
                        f"--lr {lr} "\
                        f"--K {K} "\
                        f"--train_size {train_size} "\
                        f"--total_batch_size {args.total_batch_size} "\
                        f"--train_type {args.train_type} "\
                        f"--test_type {args.test_type} "\
                        f"--epochs {args.epochs} "\
                        f">> {LOG_DIR}/{fp}"
            subprocess.run(command, shell=True)