import json, os
import argparse
import pandas as pd
import random


parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=15)
parser.add_argument("--temp_type", type=str, default="simple")
parser.add_argument("--dataset", type=str, default="ml-1m")
parser.add_argument("--set", type=str, default="train", help="train/valid/test")
args = parser.parse_args()

DATA_DIR = f"./data/{args.dataset}/proc_data/data"


train_set = json.load(open('/'.join([DATA_DIR, f"train/train_5_simple.json"])))
random.seed(42)
train_index = random.sample(range(len(train_set), 70000)


for set in ["train"]:
    for K in [args.K]:
        for temp_type in [ "simple", "sequential"]:
            print(f"==> {set}, {K}, {temp_type}")
            
            args.K = K
            args.set = set
            args.temp_type = temp_type
            print(args)

            # assert args.temp_type in ["simple", "sequential"]

            fp = '/'.join([DATA_DIR, f"{args.set}/{args.set}_{args.K}_{args.temp_type}.json"])
            data = json.load(open(fp, 'r'))
            indice = train_index
            sampled = [data[i] for i in indice]
            json.dump(sampled, open('/'.join([DATA_DIR, f"{args.set}/{args.set}_{args.K}_{args.temp_type}_sampled.json"]), "w"), indent=4)
            print("  Dumped.")
            
for set in ["train"]:
    for K in [args.K]:
        a = json.load(open(f"{DATA_DIR}/{set}/{set}_{K}_simple_sampled.json"))
        print(len(a))
        b = json.load(open(f"{DATA_DIR}/{set}/{set}_{K}_sequential_sampled.json"))
        print(len(b))
        t = []
        for m, n in zip(a, b):
            t.append(m)
            t.append(n)
        print(len(t))
        json.dump(t, open(f"{DATA_DIR}/{set}/{set}_{K}_mixed_sampled.json", "w"), indent=4)

