import os
import argparse
import numpy as np
from torch.nn.functional import interpolate
from facenet_pytorch import MTCNN
import sys
import inspect
import cv2
from os.path import join as ojoin
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from generators.stylegan_generator import StyleGANGenerator
from generators.stylegan3_generator import StyleGAN3Generator
from generators.GANControl_generator import GANControlGenerator
from utils.align_trans import norm_crop
from create_dataset import random_sampling
from create_boundary_data import extract_features, cos_sims_to_ref
from utils.utils import pairwise_cos_sim


def plot_sims(sims, save_path, class_side="class1"):
    imgs_p_id = 20
    num_dists = 4
    if class_side == "class1":
        sims = sims[: len(sims) // 2]
    else:
        sims = sims[len(sims) // 2 :]

    y_pos = np.arange(imgs_p_id)
    colors = ["b", "orange", "g", "r"]
    for i in range(num_dists):
        sim_p_dist = sims[i * imgs_p_id : (i + 1) * imgs_p_id]
        x = (y_pos + (1 / 4) * i) - 0.375
        x = x[:len(sim_p_dist)]
        mean = np.mean(sim_p_dist)
        plt.bar(x, sim_p_dist, align="center", width=1 / 5)
        plt.axhline(mean, color=colors[i], linestyle="dashed", label='_nolegend_')
        
    label = ["max-off: 10", "max-off: 20", "max-off: 30", "max-off: 40"]
    plt.xticks(y_pos, y_pos, rotation=90)
    plt.ylabel("Cosine Similarity", size=14)
    # plt.title(title)
    # Striche auf x-Achse ausschalten
    plt.tick_params(
        axis="x",
        which="both",  # major und minor ticks
        bottom=False,  # ticks auf der x-Achse (unten)
    )
    plt.legend(label)
    save_file = os.path.join(save_path, f"{class_side}_cos_sims.png")
    plt.savefig(save_file, format="png", dpi=600)
    plt.show()
    plt.close()


def intra_class_sims(embs, save_path, class_side="class1"):
    imgs_p_id = 20
    num_dists = 4
    if class_side == "class1":
        embs = embs[: len(embs) // 2]
    else:
        embs = embs[len(embs) // 2 :]
    sims = []
    colors = ["b", "orange", "g", "r"]
    for i in range(num_dists):
        emb_p_dist = embs[i * imgs_p_id : (i + 1) * imgs_p_id]
        gens1, gens2 = [], []
        for k in range(len(emb_p_dist)):
            e1 = emb_p_dist[k]
            for l in range(k + 1, len(emb_p_dist)):
                e2 = emb_p_dist[l]
                gens1.append(e1)
                gens2.append(e2)
        sim = pairwise_cos_sim(gens1, gens2)
        mean = np.mean(sim)
        plt.axvline(mean, color=colors[i], linestyle="dashed", label='_nolegend_')
        sims.append(sim)

    label = ["max-off: 10", "max-off: 20", "max-off: 30", "max-off: 40"]
    plt.hist(sims, bins=20, density=True)  # , rwidth=0.8)
    plt.xlabel("Cosine Similarity", size=14)
    plt.ylabel("Probability Density", size=14)
    plt.legend(label)
    save_file = os.path.join(save_path, f"intra_{class_side}_cos_sims.png")
    plt.savefig(save_file, format="png", dpi=600)
    plt.show()
    plt.close()


def get_latents(lat_path, boundary_path, latent_type):
    lat = np.load(lat_path)
    boundary = np.load(boundary_path)
    min_dist = 0
    imgs_per_id = 20
    distances = [10, 20, 30, 40]
    latents1, latents2 = [], []
    for dist in distances:
        lats = random_sampling(
            lat,
            boundary,
            imgs_per_id,
            min_dist=min_dist,
            max_dist=dist,
            latent_type=latent_type,
            sample_both_sides=True,
        )
        latents1.append(lats[:imgs_per_id])
        latents2.append(lats[imgs_per_id:])
    latents1 = np.vstack(latents1)
    latents2 = np.vstack(latents2)
    return np.vstack((lat, latents1, latents2))


def edit_image(lat_path, boundary_path, latent_type, modelname):
    if modelname == "stylegan_ffhq":
        model = StyleGANGenerator("stylegan_ffhq", batchsize=4)
    elif modelname == "stylegan2_ffhq":
        model = StyleGAN3Generator("stylegan2_ffhq", batchsize=4)
    elif modelname == "stylegan3_ffhq":
        model = StyleGAN3Generator("stylegan3_ffhq", batchsize=4)
    elif modelname == "gan_control":
        model = GANControlGenerator(batchsize=1)
    mtcnn = MTCNN(
        select_largest=True, min_face_size=60, post_process=False, device="cuda:0"
    )

    latents = get_latents(lat_path, boundary_path, latent_type)

    print("Latents shape:", latents.shape)
    lat_name = os.path.split(lat_path)[-1].split(".")[0]
    save_dir = f"vis_imgs/{modelname}_z_all_dists_{lat_name}"
    img_path = ojoin(save_dir, "imgs")
    ref_path = ojoin(save_dir, "ref")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(ref_path, exist_ok=True)

    no_faces = 0
    img_counter = 0
    for lats in model.get_batch_inputs(latents):
        imgs_large = model.synthesize(lats, latent_type)
        imgs = interpolate(imgs_large, 150)
        imgs = model.postprocess(imgs)
        imgs_large = model.postprocess(imgs_large)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs_large = imgs_large.permute(0, 2, 3, 1)
        _, _, landmarks = mtcnn.detect(imgs, landmarks=True)
        imgs = imgs.detach().cpu().numpy()
        imgs_large = imgs_large.detach().cpu().numpy()

        for i, (img, landmark, img_large) in enumerate(
            zip(imgs, landmarks, imgs_large)
        ):
            if landmark is None:
                no_faces += 1
                continue
            facial5points = landmark[0]
            warped_face = norm_crop(img, landmark=facial5points, image_size=112)
            img_name = "%03d.jpg" % (img_counter)
            if img_counter == 0:
                cv2.imwrite(ojoin(ref_path, f"{lat_name}.png"), warped_face[:, :, ::-1])
                # cv2.imwrite(ojoin(save_dir, f"{lat_name}.png"), img_large[:, :, ::-1])
            else:
                cv2.imwrite(ojoin(img_path, img_name), warped_face[:, :, ::-1])
                # cv2.imwrite(ojoin(save_dir, img_name), img_large[:, :, ::-1])
            img_counter += 1
    print(f"No faces detected in {no_faces} images.")
    return save_dir


def calculate_cos_sims(id_dir, fr_path):
    emb_file = ojoin(id_dir, "cossims.npy")
    if os.path.isfile(emb_file):
        print("loading embeddings and cosine similaritites...")
        cos_sims = np.load(emb_file)
        embs = np.load(ojoin(id_dir, "embs.npy"))
    else:
        ref_dir = ojoin(id_dir, "ref")
        imgs_dir = ojoin(id_dir, "imgs")
        embs, _ = extract_features(imgs_dir, 2, 0, fr_path)
        emb_ref, _ = extract_features(ref_dir, 1, 0, fr_path)
        emb_ref = emb_ref[0]
        cos_sims = cos_sims_to_ref(emb_ref, embs)
        np.save(emb_file, cos_sims)
        np.save(ojoin(id_dir, "embs.npy"), embs)
    print("plotting identity information perservation results...")
    plot_sims(cos_sims, id_dir, "class1")
    plot_sims(cos_sims, id_dir, "class2")
    print("plotting intra-class identity information perservation results...")
    intra_class_sims(embs, id_dir, "class1")
    intra_class_sims(embs, id_dir, "class2")
    print("successfully plotted")


def main(args):
    lat_path = args.lat_path
    b_path = args.boundary_path
    latent_type = args.latent_type
    modelname = args.modelname
    fr_path = args.fr_path
    save_dir = edit_image(lat_path, b_path, latent_type, modelname)
    # save_dir = "vis_imgs/stylegan2_ffhq_all_dists_0000003"
    calculate_cos_sims(save_dir, fr_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Master MoCo Training")
    parser.add_argument(
        "--lat_path",
        type=str,
        default="/data/synthetic_imgs/StyleGAN3_with_w_latents/w_latents/w_latents/0000000.npy",
        help="path image",
    )
    parser.add_argument(
        "--boundary_path",
        type=str,
        default="/home/boundaries/boundaries_SG3_w_space/boundary_00000.npy",
        help="path to boundary",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="stylegan3_ffhq",
        help="stylegan_ffhq, stylegan2_ffhq, stylegan3_ffhq, gan_control",
    )
    parser.add_argument("--latent_type", type=str, default="w")
    parser.add_argument(
        "--fr_path",
        type=str,
        default="path/to/pre-trained/FR/model.pth",
        help="path to pretrained FR model used for similarity score calculation",
    )
    args = parser.parse_args()
    main(args)
