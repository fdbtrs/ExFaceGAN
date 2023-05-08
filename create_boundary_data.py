import os
import argparse
from os.path import join as ojoin
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from backbones.iresnet import iresnet100, iresnet50
from utils.dataloader import InferenceDataset
from utils.augmentation import get_conventional_aug_policy
from utils.utils import load_embeddings, save_embeddings, pairwise_cos_sim


@torch.no_grad()
def extract_features(datadir, batchsize, num_imgs, fr_path, num_ids=0):
    """feature extraction of images in datadir with FR model specified in fr_path
    args:
        datadir: path to dir containing images
        batchsize: batch size
        num_imgs: number of images to extract; 0: all images
        fr_path: path to FR model
        num_ids: maximal number of identities
    return:
        numpy array of features, list of image paths
    """
    device = torch.device(0)
    print(f"Loading {fr_path}...")
    encoder = iresnet50(num_features=512, dropout=0.0).to(device)
    ckpt = torch.load(fr_path, map_location=device)
    encoder.load_state_dict(ckpt)
    encoder.eval()

    transform = get_conventional_aug_policy("totensor")
    dataset = InferenceDataset(datadir, transform, num_imgs=num_imgs, num_ids=num_ids)
    print(f"{len(dataset)} images loaded")
    loader = DataLoader(
        dataset, batch_size=batchsize, shuffle=False, num_workers=1, pin_memory=True
    )
    print("Feature extraction...")
    embs = []
    img_paths = []
    for images, img_path in tqdm(loader):
        images = images.to(device)
        emb = encoder(images)
        emb = F.normalize(emb)
        emb = emb.detach().cpu().numpy()
        embs.append(emb)
        img_paths += img_path
    embs = np.vstack(embs)
    print("Embedding shape:", embs.shape)
    return embs, img_paths


def load_latents(img_paths, dir_path, latent_type):
    """load latents corresponding to images
    args:
        img_paths: list of image paths
        dir_path: path to dir containing images and latents
        latent_type: load z space, w space, or wp space
    return:
        numpy array of latents
    """
    lats = []
    for img_path in img_paths:
        img_name = os.path.split(img_path)[1]
        lat_name = img_name.replace(".jpg", ".npy")
        lat_path = os.path.join(dir_path, f"{latent_type}_latents", lat_name)
        lats.append(np.load(lat_path))
    return np.array(lats)


def cos_sims_to_ref(ref, probes):
    """calculates pair wise cosine similarity between ref and every probe
    args:
        ref: numpy array of reference feature
        probes: numpy array of probe features
    return:
        numpy array of cosine similarities
    """
    ref = np.repeat(ref[np.newaxis, :], len(probes), axis=0)
    return pairwise_cos_sim(ref, probes)


def create_id_svm_data(
    pr_embs, img_paths, lat_path, ref_emb, ref_img, id, save_path, latent_type
):
    """creates svm training data for given class/identity
    args:
        embs: infered embeddings of FR model
        img_path: list of image paths corresponding to embeddings
        lat_path: path to directory containing generator latents
        id: class number
        save_path: path to save the svm training data
    """
    cos_sims = cos_sims_to_ref(ref_emb, pr_embs)

    sorted_sims = np.sort(cos_sims)
    threshold = sorted_sims[len(sorted_sims) // 2]
    # print("Threshold:", threshold)

    labels = (cos_sims < threshold).astype(int)
    labels = np.expand_dims(labels, axis=1)

    probe_lats = load_latents(img_paths, lat_path, latent_type)
    ref_lat = load_latents([ref_img], lat_path, latent_type)
    ref_lat = np.expand_dims(ref_lat, axis=0)

    save_path = ojoin(save_path, "data_%05d" % (id))
    os.makedirs(save_path, exist_ok=True)
    np.save(ojoin(save_path, "id_bound_latents.npy"), probe_lats)
    np.save(ojoin(save_path, "id_bound_labels.npy"), labels)
    if id % 100 == 0:
        print(f"Image {id} finished")


def create_svm_data(embs, img_paths, lat_path, save_path, latent_type, pool_size=30):
    """creates svm training data for each class in embeddings
    args:
        embs: infered embeddings of FR model
        img_path: list of image paths corresponding to embeddings
        lat_path: path to directory containing generator latents
        save_path: path to save the svm training data
        pool_size: size of process pool to run in parallel
    """
    executor = ProcessPoolExecutor(pool_size)
    processes = []
    for i in range(len(embs)):
        pr_embs = np.delete(embs, i, axis=0)
        pr_imgs = img_paths[:i] + img_paths[i + 1 :]
        ref_emb = embs[i]
        ref_img = img_paths[i]
        p = executor.submit(
            create_id_svm_data,
            pr_embs,
            pr_imgs,
            lat_path,
            ref_emb,
            ref_img,
            i,
            save_path,
            latent_type,
        )
        processes.append(p)
    wait(processes)


def create_5k_svm_data(embs, img_paths, lat_path, save_path, latent_type, pool_size=30):
    """creates 5K svm training data for arbitrary number of identities
    args:
        embs: infered embeddings of FR model
        img_path: list of image paths corresponding to embeddings
        lat_path: path to directory containing generator latents
        save_path: path to save the svm training data
        pool_size: size of process pool to run in parallel
    """
    executor = ProcessPoolExecutor(pool_size)
    processes = []
    num_embs = len(embs)
    print("Start generating SVM training data...")
    print("Number of embeddings:", num_embs)
    for i in range(num_embs):
        comp_idxs = np.random.choice(num_embs, 4999, replace=False)
        emb = embs[comp_idxs]
        img_path = [img_paths[i] for i in comp_idxs]
        ref_emb = embs[i]
        ref_img = img_paths[i]
        p = executor.submit(
            create_id_svm_data,
            emb,
            img_path,
            lat_path,
            ref_emb,
            ref_img,
            i,
            save_path,
            latent_type,
        )
        processes.append(p)
    wait(processes)


def main(args):
    img_dir = os.path.join(args.datadir, "images_aligned")
    emb_dir = os.path.join(args.datadir, "embeddings")
    if os.path.isdir(emb_dir):
        embs, img_paths = load_embeddings(emb_dir, num_embs=args.num_classes)
    else:
        os.makedirs(emb_dir, exist_ok=True)
        embs, img_paths = extract_features(
            img_dir, args.batchsize, args.num_classes, args.fr_path
        )
        save_embeddings(emb_dir, img_paths, embs)
    create_svm_data(embs, img_paths, args.datadir, args.save_path, args.latent_type)
    # create_5k_svm_data(embs, img_paths, args.datadir, args.save_path, args.latent_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Training Data for SVM Boundary Training"
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="/data/synthetic_imgs/StyleGAN3_with_w_latents",
        help="path to directory with generated images and latents",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/data/synthetic_imgs/SG3_SVM_data",
        help="where to save the boundary training data",
    )
    parser.add_argument("--num_classes", type=int, default=5000)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--latent_type", type=str, default="w")
    parser.add_argument(
        "--fr_path",
        type=str,
        default="path/to/pre-trained/FR/model.pth",
        help="path to pretrained FR model used for similarity score calculation",
    )
    args = parser.parse_args()
    main(args)
