import json, os
from load_prompt_ml1m import zero_shot_get_prompt, zero_shot_ret_get_prompt
from load_prompt_BookCrossing import book_zero_shot_get_prompt, book_zero_shot_ret_get_prompt
from load_prompt_ml25m import ml_25m_zero_shot_get_prompt, ml_25m_zero_shot_ret_get_prompt
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=15)
parser.add_argument("--temp_type", type=str, default="simple")
parser.add_argument("--dataset", type=str, default="BookCrossing")
parser.add_argument("--set", type=str, default="test", help="train/valid/test")
args = parser.parse_args()
print(args)

DATA_DIR = f"./data/{args.dataset}/proc_data"

assert args.temp_type in ["simple", "sequential", "high", "low"]
assert args.set in ["train", "test"]

fp = os.path.join(DATA_DIR, "data")
os.makedirs(fp, exist_ok=True)
fp = os.path.join(fp, args.set)
os.makedirs(fp, exist_ok=True)
file_name = '_'.join([args.set, str(args.K), args.temp_type])
fp = os.path.join(fp, file_name+'.json')

if args.temp_type in ["simple"]: # w/o ret
    if args.dataset == "ml-25m":
        msg_iter = ml_25m_zero_shot_get_prompt(
            K=args.K,
            istrain=args.set,
            temp_type=args.temp_type, 
            data_dir=DATA_DIR
        )
    elif args.dataset == "BookCrossing":
        msg_iter = book_zero_shot_get_prompt(
            K=args.K,
            istrain=args.set,
            temp_type=args.temp_type, 
            data_dir=DATA_DIR
        )
    elif args.dataset == "ml-1m":
        msg_iter = zero_shot_get_prompt(
            K=args.K,
            istrain=args.set,
            temp_type=args.temp_type, 
            data_dir=DATA_DIR
        )

elif args.temp_type in ["sequential"]: # w/ ret
    if args.dataset == "ml-25m":
        msg_iter = ml_25m_zero_shot_ret_get_prompt(
            K=args.K,
            istrain=args.set,
            temp_type=args.temp_type, 
            data_dir=DATA_DIR
        )
    elif args.dataset == "BookCrossing":
        msg_iter = book_zero_shot_ret_get_prompt(
            K=args.K,
            istrain=args.set,
            temp_type=args.temp_type, 
            data_dir=DATA_DIR
        )
    elif args.dataset == "ml-1m":
        msg_iter = zero_shot_ret_get_prompt(
            K=args.K,
            istrain=args.set,
            temp_type=args.temp_type, 
            data_dir=DATA_DIR
        )



ori_data_fp = os.path.join(DATA_DIR, f"{args.set}.parquet.gz")
df = pd.read_parquet(ori_data_fp)

data_list = []
for msg, idx in zip(msg_iter, df.index):
    labels = df.loc[idx, "labels"]
    data_dict = {}
    data_dict['input'] = msg
    data_dict['output'] = "Yes." if int(labels) == 1 else "No."
    data_list.append(data_dict)

assert len(data_list) == len(df.index)

json.dump(data_list, open(fp, "w"), indent=4)
