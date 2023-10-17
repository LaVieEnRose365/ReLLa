import os, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from tqdm import trange
# import faiss


def item_sim(args):
    fp = f"./embeddings/ml-1m_{args.pooling}.npy"
    embed = np.load(fp)
    
    print(embed.shape)
    pca = PCA(n_components=args.dim)
    embed = pca.fit_transform(embed)
    print("PCA finished.")

    sim_matrix = cosine_similarity(embed)
    print("Similarity matrix computed.")

    sorted_indice = np.argsort(-sim_matrix, axis=1)
    print("Sorted.")

    fp_indice =os.path.join(args.embed_dir, '_'.join(["ml-1m", args.pooling, "indice"])+".npy")
    np.save(fp_indice, sorted_indice)
    print("Saved.")
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, default="./embeddings")
    parser.add_argument("--pooling", type=str, default="average", help="average/last")
    parser.add_argument("--dim", type=int, default=512)
    args = parser.parse_args()
    item_sim(args)

