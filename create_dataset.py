import os
import argparse
from os.path import join as ojoin
import cv2
import numpy as np
from tqdm import tqdm
from torch.nn.functional import interpolate
from facenet_pytorch import MTCNN

from generators.stylegan_generator import StyleGANGenerator
from generators.stylegan3_generator import StyleGAN3Generator
from generators.GANControl_generator import GANControlGenerator
from utils.align_trans import norm_crop


def random_sampling(
    latent_code,
    boundary,
    num,
    min_dist=0,
    max_dist=10,
    latent_type="w",
    sample_both_sides=False,
):
    """randomly sample codes on identity preservation side of boundary
    args:
        latent_code: latent code for manipulation
        boundary: semantic class preservation boundary
        num: number of manipulations
        min_dist: minimal distance from input latent code
        max_dist: maximal distance from input latent code
        latent_type: type of latent code (z,w)
        sample_both_sides: consider both sides of boundary as 2 different identites
    return:
        numpy array of manipulated latent codes that preserve the identity
    """
    lat_dim = latent_code.shape[-1]
    offset = np.random.uniform(min_dist, max_dist, (num, lat_dim))
    gen_points = latent_code - offset * boundary
    if sample_both_sides:
        gen_points2 = latent_code + offset * boundary
        gen_points = np.vstack((gen_points, gen_points2))
    if latent_type == "z":
        norm = np.linalg.norm(gen_points, axis=1, keepdims=True)
        gen_points = gen_points / norm * np.sqrt(latent_code.shape[0])
    return gen_points


def load_boundaries(boundary_dir, num_boundaries=0, start_num=0):
    """load boundaries
    args:
        boundary_dir: path to directory containing boundaries
        num_boundaries: maximal number of boundaries; 0: load all boundaries
    return:
        numpy array of boundaries [num_boundaries, 1, latent_dim],
        list of boundary filenames
    """
    b_files = sorted(os.listdir(boundary_dir))
    if num_boundaries != 0:
        b_files = b_files[start_num : start_num + num_boundaries]
    boundaries = []
    for bf in b_files:
        b = np.load(ojoin(boundary_dir, bf))
        boundaries.append(b)
    return np.array(boundaries), b_files


def load_lats_to_boundary(lat_dir, b_names, lat_dim=512):
    """load latent codes corresponding to boundaries
    args:
        lat_dir: path to directory containing latent codes
        b_names: list of boundary filenames
        lat_dim: dimension of latent codes
    return:
        numpy array of latent codes [len(b_names), latent_dim]
    """
    latents = np.zeros((len(b_names), lat_dim))
    for i, b_name in enumerate(b_names):
        # get only image number of boundary
        img_num = int(b_name.split("_")[-1].split(".")[0])
        lat_filename = "%07d.npy" % (img_num)
        lat = np.load(ojoin(lat_dir, lat_filename))
        latents[i] = lat
    return latents


def generate_latents(
    latent_dir,
    boundary_dir,
    num_classes,
    imgs_p_class,
    latent_type="w",
    distance=30,
    start_num=0,
    sample_both_sides=False,
):
    """generate latents and labels for all classes
    args:
        latent_dir: path to directory containing latent codes
        boundary_dir: path to directory containing boundaries
        num_classes: number of classes that should be generated
        imgs_p_class: images per class that should be generated
        latent_type: z / w
        distance: distance to boundary (max-off)
        start_num: start offset to create images in parallel
        sample_both_sides: consider both sides of boundary as 2 different identites
    return:
        numpy array of latent codes [num_classes*imgs_p_class, latent_dim],
        numpy array of class labels of dim num_classes*imgs_p_class [0,0,..,1,1,..,2,2,..]
    """
    boundaries, b_names = load_boundaries(boundary_dir, num_classes, start_num)
    latents = load_lats_to_boundary(latent_dir, b_names)
    assert len(boundaries) == len(latents), (
        f"boundaries and latents should have same length "
        f"but received boundary length: {len(boundaries)} and latents length: {len(latents)}"
    )
    lats_per_class = []
    print("Sample latents randomly...")
    for lat, boundary in zip(latents, boundaries):
        class_lats = random_sampling(
            lat,
            boundary,
            imgs_p_class,
            max_dist=distance,
            latent_type=latent_type,
            sample_both_sides=sample_both_sides,
        )
        lats_per_class.append(class_lats)
    # new dimension of [num_classes*imgs_p_class, latent_dim]
    data_lats = np.vstack(lats_per_class)
    print("shape of latent codes:", data_lats.shape)
    # classes of dim: num_classes*imgs_p_class [0,0,..,1,1,..,2,2,..]
    if sample_both_sides:
        start_num = start_num * 2
        num_classes = num_classes * 2
    clss = np.arange(start_num, start_num + num_classes)
    clss = np.repeat(clss, repeats=imgs_p_class)
    return data_lats, clss


def generate_images(
    latents, labels, modelname, batchsize, out_dir, imgs_per_class, latent_type="z"
):
    """generates and saves all images
    args:
        latents: numpy array of latents [len of dataset, latent_dim]
        labels: numpy array of class labels of dimension length of dataset
        batchsize: batch size
        out_dir: path where to save the images
        imgs_p_class: images per class that should be generated
    """
    num_dataset = len(latents)
    out_dir = ojoin(out_dir, "images")
    if modelname == "stylegan_ffhq":
        model = StyleGANGenerator("stylegan_ffhq", batchsize)
    elif modelname == "stylegan2_ffhq":
        model = StyleGAN3Generator("stylegan2_ffhq", batchsize)
    elif modelname == "stylegan3_ffhq":
        model = StyleGAN3Generator("stylegan3_ffhq", batchsize)
    elif modelname == "gan_control":
        model = GANControlGenerator(batchsize=batchsize)
    mtcnn = MTCNN(
        select_largest=True, min_face_size=60, post_process=False, device="cuda:0"
    )

    no_faces = 0
    img_counter = 0
    print(f"Generate {num_dataset} images...")
    lat_batches = model.get_batch_inputs(latents)
    labels_batches = model.get_batch_inputs(labels)
    pbar = tqdm(zip(lat_batches, labels_batches), total=(num_dataset // batchsize + 1))
    for lats, labs in pbar:
        imgs = model.synthesize(lats, latent_type)
        imgs = interpolate(imgs, 150)
        imgs = model.postprocess(imgs)
        imgs = imgs.permute(0, 2, 3, 1)
        _, _, landmarks = mtcnn.detect(imgs, landmarks=True)
        imgs = imgs.detach().cpu().numpy()

        for img, label, landmark in zip(imgs, labs, landmarks):
            if landmark is None:
                no_faces += 1
                continue
            cls_dir = ojoin(out_dir, "%05d" % (label))
            os.makedirs(cls_dir, exist_ok=True)

            facial5points = landmark[0]
            warped_face = norm_crop(img, landmark=facial5points, image_size=112)
            img_name = "%05d_%03d.jpg" % (label, img_counter % imgs_per_class)
            cv2.imwrite(os.path.join(cls_dir, img_name), warped_face[:, :, ::-1])
            img_counter += 1
    print(f"No faces detected in {no_faces} images.")


def main(args):
    latents, labels = generate_latents(
        args.latent_dir,
        args.boundary_dir,
        args.num_classes,
        args.img_p_class,
        args.latent_type,
        args.dist,
        args.start_offset,
        args.sample_both_sides,
    )
    generate_images(
        latents,
        labels,
        args.modelname,
        args.batchsize,
        args.output_path,
        args.img_p_class,
        args.latent_type,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DIRGAN dataset")
    parser.add_argument(
        "--latent_dir",
        type=str,
        default="/data/synthetic_imgs/StyleGAN3_with_w_latents/w_latents",
        help="path to directory with latents",
    )
    parser.add_argument(
        "--boundary_dir",
        type=str,
        default="/home/boundaries/boundaries_SG3_w_space",
        help="path to directory contaning all boundaries",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5000,
        help="how many classes should be genereated",
    )
    parser.add_argument(
        "--start_offset",
        type=int,
        default=0,
        help="start offset to create images in parallel",
    )
    parser.add_argument(
        "--img_p_class",
        type=int,
        default=60,
        help="how many images per class should be genereated",
    )
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--dist", type=float, default=30, help="max off parameter")
    parser.add_argument("--latent_type", type=str, default="w")
    parser.add_argument(
        "--sample_both_sides",
        type=bool,
        default=True,
        help="sample both sides of boundary as 2 different identities",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="stylegan3_ffhq",
        help="stylegan_ffhq, stylegan2_ffhq, stylegan3_ffhq, gan_control",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/synthetic_imgs/ExFaceGAN_SG3",
        help="where to save the generated images",
    )
    args = parser.parse_args()
    main(args)
