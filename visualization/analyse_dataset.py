import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import inspect
from tqdm import tqdm
from multiprocessing import Process, Queue

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from create_boundary_data import extract_features
from utils.utils import save_emb_2_id, pairwise_cos_sim

rcParams.update({"figure.autolayout": True})


def load_embeddings(path, num_ids=0, num_imgs=0):
    """load embeddings from files in directory path
    args:
        path: path to directory containing embedding files
        num_ids: number of identities loaded; 0=all
        num_imgs: number of images per identity loaded; 0=all
    return:
        list of embeddings per identity [identities, embs_per_id, emb_dim]
    """
    files = sorted(os.listdir(path))
    if num_ids != 0:
        files = files[:num_ids]
    embeddings = []
    for f in files:
        e = np.load(os.path.join(path, f))
        if num_imgs != 0:
            e = e[:num_imgs]
        embeddings.append(e)
    return embeddings


def split_gen_imp(embeddings):
    """split embeddings in genuine and impostor pairs
    args:
        embeddings: list of embeddings per identity
    return:
        numpy array of genuine pairs, numpy array of impostor pairs
    """
    gens1, gens2, impos1, impos2 = [], [], [], []
    num_ids = len(embeddings)
    print("Create genuine and impostor pairs...")
    for p in tqdm(range(num_ids)):
        id_embs = embeddings[p]
        for i in range(len(id_embs)):
            emb1 = id_embs[i]
            for j in range(i + 1, len(id_embs)):
                emb2 = id_embs[j]
                gens1.append(emb1)
                gens2.append(emb2)

        num_id_embs = len(id_embs)
        num_id_samples = min(num_id_embs, 4)
        for ref_idx in range(p + 1, num_ids, 8):
            ref_id_embs = embeddings[ref_idx]
            num_ref_embs = len(ref_id_embs)
            num_ref_samples = min(num_ref_embs, 4)
            img1_idx = np.random.choice(num_id_embs, num_id_samples, replace=False)
            for i in img1_idx:
                emb1 = id_embs[i]
                img2_idx = np.random.choice(
                    num_ref_embs, num_ref_samples, replace=False
                )
                for j in img2_idx:
                    emb2 = ref_id_embs[j]
                    impos1.append(emb1)
                    impos2.append(emb2)

    print("Genuine shape:", len(gens1))
    print("Impostor shape:", len(impos2))
    return gens1, gens2, impos1, impos2


def genereate_embeddings(args):
    """infer and save embeddings"""
    img_path = os.path.join(args.datadir, "images")
    emb_path = os.path.join(args.datadir, "embeddings")
    os.makedirs(emb_path, exist_ok=True)
    num_ids = len(os.listdir(img_path))
    if args.num_ids != 0:
        num_ids = args.num_ids
    if len(os.listdir(emb_path)) >= num_ids-10:
        return
    embs, img_paths = extract_features(
        img_path, args.batchsize, 0, args.fr_path, num_ids=num_ids
    )
    save_emb_2_id(embs, img_paths, emb_path)


def split_list(l, num):
    """splits given list or numpy array into num amount of chunks
    args:
        l: list or numpy array
        num: amount of chunks
    return:
        list of lists or numpy arrays
    """
    chunk = len(l) // num
    out = []
    for i in range(num):
        start_idx = i * chunk
        if i == num - 1:
            out.append(l[start_idx:])
        else:
            out.append(l[start_idx : start_idx + chunk])
    return out


def cos_sim(emb1, emb2, queue):
    sims = pairwise_cos_sim(emb1, emb2)
    queue.put(sims)


def calculate_cos_sim(emb1, emb2, nthreads=10):
    """calculates the pairwise cosine similarity between both embeddings
    in multiple parallel processes.
    args:
        emb1: list of embeddings
        emb2: list of embeddings
        nthreads: number of parallel threads
    return:
        list of pairwise cosine similarities
    """
    q = Queue()
    processes = []
    emb_list1 = split_list(emb1, nthreads)
    emb_list2 = split_list(emb2, nthreads)
    cos_sims = []
    for embs1, embs2 in zip(emb_list1, emb_list2):
        p = Process(
            target=cos_sim,
            args=(embs1, embs2, q),
        )
        processes.append(p)
        p.start()

    for p in processes:
        ret = q.get()  # will block
        cos_sims.append(ret)
    for p in processes:
        p.join()
    return np.hstack(cos_sims)


def plot_gen_imp_distribution(datadir, num_ids=0, num_imgs=0):
    """plots genuine impostor distribution and saves genuine and impostor scores
    args:
        datadir: directory containing embeddings folder
        num_ids: number of identities
        num_imgs: number of images per identity
    """
    emb_path = os.path.join(datadir, "embeddings")
    print("Load embeddings from:", emb_path)
    embeddings = load_embeddings(emb_path, num_ids, num_imgs)
    
    gens1, gens2, impos1, impos2 = split_gen_imp(embeddings)
    print("Calculating genuine cosine similarity...")
    gen_sims = calculate_cos_sim(gens1, gens2, nthreads=5)
    print("Calculating impostor cosine similarity...")
    impo_sims = calculate_cos_sim(impos1, impos2, nthreads=10)

    dataname = datadir.split(os.path.sep)[-1]
    save_dir = os.path.join("data_plots", dataname + "_both_classes")
    os.makedirs(save_dir, exist_ok=True)
    genuine_file = os.path.join(save_dir, "genuines.txt")
    impostor_file = os.path.join(save_dir, "impostors.txt")
    np.savetxt(genuine_file, gen_sims)
    np.savetxt(impostor_file, impo_sims)

    print("Plot histogram...")
    savename = os.path.join(save_dir, "distribution_" + dataname + ".png")
    plt.hist(gen_sims, bins=200, alpha=0.5, label="Genuine Similarities", density=True)
    plt.hist(
        impo_sims, bins=200, alpha=0.5, label="Impostor Similarities", density=True
    )
    plt.xlabel("Cosine Similarity", size=14)
    plt.ylabel("Probability Density", size=14)
    plt.legend(loc="upper left")
    plt.savefig(savename)
    plt.close()
    print("Histogram saved")


def main(args):
    genereate_embeddings(args)
    plot_gen_imp_distribution(
        args.datadir,
        args.num_ids,
        args.num_imgs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Study of datasets")
    parser.add_argument(
        "--datadir",
        type=str,
        default="/data/synthetic_imgs/ExFaceGAN_SG3",
        help="path to directory containing the image folder",
    )
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--num_ids", type=int, default=5000)
    parser.add_argument("--num_imgs", type=int, default=50)
    parser.add_argument(
        "--fr_path",
        default="path/to/pre-trained/FR/model.pth",
    )
    args = parser.parse_args()
    main(args)
