import os
from os.path import join as ojoin
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.dataloader import LatsDataset
from generators.stylegan_generator import StyleGANGenerator
from generators.stylegan3_generator import StyleGAN3Generator
from generators.GANControl_generator import GANControlGenerator


@torch.no_grad()
def main(args):
    save_path = ojoin(args.save_path, "images")
    save_path_z = ojoin(args.save_path, "z_latents")
    save_path_w = ojoin(args.save_path, "w_latents")
    save_path_wp = ojoin(args.save_path, "wp_latents")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_z, exist_ok=True)
    os.makedirs(save_path_w, exist_ok=True)
    os.makedirs(save_path_wp, exist_ok=True)

    if args.modelname == "stylegan_ffhq":
        generator = StyleGANGenerator("stylegan_ffhq", args.batchsize)
    elif args.modelname == "stylegan2_ffhq":
        generator = StyleGAN3Generator("stylegan2_ffhq", args.batchsize)
    elif args.modelname == "stylegan3_ffhq":
        generator = StyleGAN3Generator("stylegan3_ffhq", args.batchsize)
    elif args.modelname == "gan_control":
        generator = GANControlGenerator(batchsize=args.batchsize)

    dataset = LatsDataset(
        num_imgs=args.num_imgs, lat_path=args.lat_path, seed=args.offset
    )
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)

    i = args.offset
    for latents in tqdm(loader):
        latents = latents.numpy()
        output = generator.synthesize(latents, latent_space_type="Z", return_lats=True)
        imgs = generator.postprocess(output["image"])
        imgs = interpolate(imgs, 150)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.detach().cpu().numpy()
        zs = output["z"]
        ws = output["w"]
        wps = output["wp"]
        for img, z, w, wp in zip(imgs, zs, ws, wps):
            Image.fromarray(img.astype(np.uint8), "RGB").save(
                ojoin(save_path, "%07d.jpg" % (i))
            )
            np.save(ojoin(save_path_z, "%07d.npy" % (i)), z)
            np.save(ojoin(save_path_w, "%07d.npy" % (i)), w)
            np.save(ojoin(save_path_wp, "%07d.npy" % (i)), wp)
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Master MoCo Training")
    parser.add_argument("--num_imgs", type=int, default=11000, help="number of images")
    parser.add_argument(
        "--offset", type=int, default=0, help="naming offset to create more images"
    )
    parser.add_argument(
        "--lat_path",
        type=str,
        default="None",
        help="/data/synthetic_imgs/StyleGAN3_with_w_latents/w_latents or None for random latents",
    )
    parser.add_argument("--batchsize", type=int, default=8, help="batch size")
    parser.add_argument(
        "--modelname",
        type=str,
        default="stylegan3_ffhq",
        help="stylegan_ffhq, stylegan2_ffhq, stylegan3_ffhq, gan_control",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/data/maklemt/synthetic_imgs/StyleGAN3_with_w_latents",
    )
    args = parser.parse_args()
    main(args)
