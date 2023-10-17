import os, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="test")
parser.add_argument("--pooling", type=str, default="average")
parser.add_argument("--data_dir", type=str, default="./data/ml-25m/proc_data")
args = parser.parse_args()

embeddings = np.load(f'./embeddings/ml-25m_{args.pooling}.npy')
print(embeddings.shape)

pca = PCA(n_components=512)
embeddings = pca.fit_transform(embeddings)
print("PCA finished.")

all_indice = []

df = pd.read_parquet(f"{args.data_dir}/{args.set}.parquet.gz")


for idx, row in tqdm(df.iterrows()):
    tgt_id = row['Movie ID']
    hist_id = row['history ID']
    
    tgt_embed, hist_embed = embeddings[tgt_id], embeddings[hist_id]
    seq_id_to_movie_id = {i: movie_id for i, movie_id in enumerate(hist_id)}
    sim_matrix = np.sum(hist_embed * tgt_embed, axis=-1)
    indice = np.argsort(-sim_matrix)[:80].tolist()
    sorted_indice = list(map(lambda x: int(seq_id_to_movie_id[x]), indice))
    all_indice.append(sorted_indice)  
    
json.dump(all_indice, open(f'./embeddings/ml-25m_{args.pooling}_indice_{args.set}.json', 'w'), indent=4)

