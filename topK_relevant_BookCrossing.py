import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="test")
parser.add_argument("--pooling", type=str, default="average")
parser.add_argument("--data_dir", type=str, default="./data/BookCrossing/proc_data")
args = parser.parse_args()

isbn2id = json.load(open(f"{args.data_dir}/isbn2id.json"))
id2book = json.load(open(f"{args.data_dir}/id2book.json"))
embeddings = np.load(f"./embeddings/BookCrossing_{args.pooling}.npy")
print("Embeddings loaded.")
print(embeddings.shape)

all_indice = []

df = pd.read_parquet(f"{args.data_dir}/{args.set}.parquet.gz")
df = df.reset_index(drop=True)

for idx, row in tqdm(df.iterrows()):
    tgt_id = isbn2id[row['ISBN']]
    hist_id = [isbn2id[isbn] for isbn in row['user_hist']]
    
    tgt_embed, hist_embed = embeddings[tgt_id], embeddings[hist_id]
    
    seq_id_to_book_id = {i: book_id for i, book_id in enumerate(hist_id)}
    sim_matrix = np.sum(hist_embed * tgt_embed, axis=-1)
    indice = np.argsort(-sim_matrix)[:100].tolist()
    sorted_indice = list(map(lambda x: id2book[str(seq_id_to_book_id[x])][0], indice))
    all_indice.append(sorted_indice)

json.dump(all_indice, open(f'./embeddings/BookCroosing_{args.pooling}_indice_{args.set}.json', 'w'), indent=4)
