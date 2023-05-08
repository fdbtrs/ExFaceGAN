import os
import numpy as np
from os.path import join as ojoin
from tqdm import tqdm
from scipy.spatial.distance import cosine


def save_emb_2_id(embs, img_paths, save_path):
    """assigns embeddings to corresponding identity and saves embeddings per identity
    args:
        embs: numpy array of embeddings
        img_paths: list of image paths in same order as embs
        save_path: path to save the embeddings
    """
    id_embs = {}
    for emb, img in zip(embs, img_paths):
        identity = img.split(os.path.sep)[-2]
        if identity in id_embs:
            id_embs[identity].append(emb)
        else:
            id_embs[identity] = [emb]

    for i, emb in id_embs.items():
        emb = np.array(emb)
        save_file = ojoin(save_path, i + ".npy")
        np.save(save_file, emb)
    print(f"{len(embs)} embeddings saved in {save_path}")


def save_embeddings(save_dir, img_paths, embeddings):
    """saves embedding under corresponding image filename to save_dir
    args:
        save_dir: path to directory to save the embeddings
        img_paths: list of image paths in same order as embs
        embeddings: numpy array of embeddings
    """
    for img_path, emb in zip(img_paths, embeddings):
        img_name = img_path.split(os.path.sep)[-1].split(".")[0]
        save_file = ojoin(save_dir, img_name + ".npy")
        np.save(save_file, emb)
    print(f"{len(embeddings)} inferred embeddings saved in:", save_dir)


def load_embeddings(dir, num_embs=0):
    """loads embeddings and slightly incorrect image paths from directory
    image paths have the correct image name, which is important for further processing
    args:
        dir: path to embedding directory
        num_embs: number of maximal embeddings that should be loaded
    return:
        numpy array of embeddings, list of corresponding image paths
    """
    emb_files = sorted(os.listdir(dir))
    if num_embs > 0:
        emb_files = emb_files[:num_embs]
    embs, img_paths = [], []
    print("Loading embeddings from:", dir)
    for emb_file in tqdm(emb_files):
        emb = np.load(ojoin(dir, emb_file))
        embs.append(emb)
        img_file = emb_file.replace(".npy", ".jpg")
        img_path = ojoin(dir, img_file)
        img_paths.append(img_path)
    print(f"{len(emb_files)} embeddings loaded from", dir)
    return np.vstack(embs), img_paths


def pairwise_cos_sim(embs1, embs2, show_pbar=False):
    """calculates the cosine similarity between each pair
    args:
        embs1: array num_samples x feature_dim
        embs2: array num_samples x feature_dim
        show_pbar: bool show progress bar
    return:
        cosine similarity [emb1_1 * emb2_1, emb1_2 * emb2_2]
    """
    if show_pbar:
        print("Calculate cosine similarity...")
    cos_sims = []
    pbar = zip(embs1, embs2)
    if show_pbar:
        pbar = tqdm(zip(embs1, embs2), total=len(embs1))
    for e1, e2 in pbar:
        cos_dist = cosine(e1, e2)
        cos_sims.append(1 - cos_dist)
    return np.array(cos_sims)
