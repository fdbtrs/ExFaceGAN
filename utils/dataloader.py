import os
import logging
from os.path import join as ojoin
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def check_for_folder_structure(datadir):
    """checks if datadir contains folders (like CASIA) or images (synthetic datasets)"""
    img_path = sorted(os.listdir(datadir))[0]
    img_path = ojoin(datadir, img_path)
    return os.path.isdir(img_path)


def load_real_paths(datadir, num_imgs=0, num_classes=0):
    """loads complete real image paths
    args:
        datadir: path to image folders
        num_imgs: number of total images
        num_classes: number of classes that should be loaded
    return:
        list of image paths
    """
    img_paths = []
    id_folders = sorted(os.listdir(datadir))
    if num_classes != 0:
        id_folders = id_folders[:num_classes]
    for id in id_folders:
        img_files = sorted(os.listdir(ojoin(datadir, id)))
        img_paths += [os.path.join(datadir, id, f_name) for f_name in img_files]
    if num_imgs != 0:
        img_paths = img_paths[:num_imgs]
    return img_paths


def load_syn_paths(datadir, num_imgs=0, start_img=0):
    """loads first level paths, i.e. image folders for DFG that contain augmentation images
    args:
        datadir: path to image folder
        num_imgs: number of images / folders
        start_img: start image index
    return:
        list of image paths
    """
    img_files = sorted(os.listdir(datadir))
    if num_imgs != 0:
        img_files = img_files[start_img : start_img + num_imgs]
    return [os.path.join(datadir, f_name) for f_name in img_files]


def load_supervised_paths(datadir, num_ids, num_imgs):
    """load e.g. DFG images with folder structure as supervised dataset
    args:
        datadir: path to directory containing the images
        num_ids: number of identities (folders) that should be loaded
        num_imgs: number of images per identity that should be loaded
    return:
        list of image paths, corresponding list of labels
    """
    img_paths, labels = [], []
    id_folders = sorted(os.listdir(datadir))[:num_ids]
    for i, id in enumerate(id_folders):
        id_path = ojoin(datadir, id)
        img_files = sorted(os.listdir(id_path))[:num_imgs]
        img_paths += [ojoin(id_path, f_name) for f_name in img_files]

        labels += [int(i)] * len(img_files)

    return img_paths, labels


def load_latents(datadir, num_lats=0):
    """load numpy latents from directory
    args:
        datadir: path to latent folder
        num_lats: number of latents
    return:
        numpy array of latents
    """
    lat_files = sorted(os.listdir(datadir))
    if num_lats != 0:
        lat_files = lat_files[:num_lats]
    lats = []
    for lat_file in lat_files:
        lats.append(np.load(ojoin(datadir, lat_file)))
    return np.array(lats)


class LimitedDataset(Dataset):
    def __init__(self, datadir, transform, num_persons, num_imgs):
        """Similar to ImageDataset, but limit the number of persons and images per person"""
        self.img_paths, self.labels = load_supervised_paths(
            datadir, num_persons, num_imgs
        )
        self.transform = transform
        dirname = os.path.basename(os.path.normpath(datadir))
        logging.info(f"{dirname}: {len(self.img_paths)} images")

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns with corresponding label."""
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")
        img = self.transform(image)
        return img, self.labels[index]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


class LatsDataset(Dataset):
    def __init__(self, num_imgs, latent_dim=512, lat_path=None, seed=42):
        self.lat_dim = latent_dim
        if lat_path == "None":
            np.random.seed(seed)
            self.latents = np.random.randn(num_imgs, latent_dim)
            self.norm = False
            print("random latent generation")
        else:
            self.latents = load_latents(lat_path, num_imgs)
            self.norm = False
        logging.info(f"Create {len(self.latents)} latent representations")

    def __getitem__(self, index):
        latent_codes = self.latents[index]  # .reshape(-1, self.lat_dim)
        if self.norm:
            norm = np.linalg.norm(latent_codes, axis=0, keepdims=True)
            latent_codes = latent_codes / norm * np.sqrt(self.lat_dim)
        return latent_codes

    def __len__(self):
        return len(self.latents)


class InferenceDataset(Dataset):
    def __init__(self, datadir, transform, num_imgs=0, num_ids=0):
        """Initializes image paths and preprocessing module."""
        self.is_folder_struct = check_for_folder_structure(datadir)
        if self.is_folder_struct:
            self.img_paths = load_real_paths(
                datadir, num_imgs, num_classes=num_ids
            )  # load_first_dfg_path()
        else:
            self.img_paths = load_syn_paths(datadir, num_imgs)

        self.transform = transform
        dirname = os.path.basename(os.path.normpath(datadir))
        logging.info(f"{dirname}: {len(self.img_paths)} images")

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")
        img = self.transform(image)
        return img, self.img_paths[index]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)
