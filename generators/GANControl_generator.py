import logging
import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from os.path import join as ojoin
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from generators.GANControl_src.gan_control.utils.spherical_harmonics_utils import (
    sh_eval_basis_1,
)
from generators.GANControl_src.gan_control.inference.controller import Controller

BASE_DIR = os.path.dirname(os.path.relpath(__file__))
if BASE_DIR:
    MODEL_DIR = ojoin(BASE_DIR, "GANControl_resources")
else:
    MODEL_DIR = "GANControl_resources"

truncation = 0.7


def load_latent_w(dir):
    w = []
    fs = sorted(os.listdir(dir))
    for f in fs:
        w.append(np.load(os.path.join(dir, f)))
    return torch.tensor(np.array(w))


def random_pose():
    yaw = torch.randint(-50, 51, (1,))
    pitch = torch.randint(-10, 11, (1,))
    # [yaw, pitch, roll]
    return torch.tensor([[yaw, pitch, 0]])


def random_illumination():
    strangth = 0.6
    axis = torch.randint(0, 2, (1,))
    direction = torch.randint(-1, 2, (1,))
    if axis == 0:
        coding = sh_eval_basis_1(direction, 0, 0)
    else:
        coding = sh_eval_basis_1(0, 0, direction)
    return torch.tensor(np.expand_dims(coding, axis=0)) * strangth


def get_expressions():
    attributes_df = pd.read_pickle(
        ojoin(MODEL_DIR, "ffhq_1K_attributes_samples_df.pkl")
    )
    return np.array(attributes_df.expression3d.to_list())


def get_random_attributes(expressions):
    pose = random_pose()
    age = torch.randint(10, 81, (1,))
    illu = random_illumination()
    express_idx = torch.randint(0, len(expressions), (1,))
    express = torch.tensor(np.expand_dims(expressions[express_idx], axis=0))
    return pose, age, illu, express


class GANControlGenerator:
    def __init__(self, batchsize=16, local_rank=0) -> None:
        controller_path = ojoin(
            MODEL_DIR, "gan_models/controller_age015id025exp02hai04ori02gam15"
        )
        logging.info("Init and load Controller...")
        self.controller = Controller(controller_path, local_rank)

        self.expressions = get_expressions()
        self.bs = batchsize

    def synthesize(
        self,
        latent_codes,
        latent_space_type="Z",
        generate_style=False,
        return_lats=False,
    ):
        latent_space_type = latent_space_type.upper()
        results = {}
        pose, age, illu, express = get_random_attributes(self.expressions)

        if latent_space_type == "Z":
            latent_codes = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            images, latent_z, latent_w = self.controller.gen_batch_by_controls(
                latent=latent_codes,
                input_is_latent=False,
                orientation=pose,
                age=age,
                gamma=illu,
                expression=express,
                to_cpu=False,
            )
        elif latent_space_type == "W":
            latent_codes = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            images, latent_z, latent_w = self.controller.gen_batch_by_controls(
                latent=latent_codes,
                input_is_latent=True,
                orientation=pose,
                age=age,
                gamma=illu,
                expression=express,
                to_cpu=False,
            )
        results["z"] = latent_z.cpu().numpy()
        results["w"] = latent_w.cpu().numpy()
        results["wp"] = latent_w.cpu().numpy()
        if return_lats:
            results["image"] = images
            return results

        return images

    def postprocess(self, images):
        images *= 255  # (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return images

    def get_batch_inputs(self, latent_codes):
        """Gets batch inputs from a collection of latent codes.

        This function will yield at most `self.batch_size` latent_codes at a time.

        Args:
        latent_codes: The input latent codes for generation. First dimension
            should be the total number.
        """
        total_num = latent_codes.shape[0]
        for i in range(0, total_num, self.bs):
            yield latent_codes[i : i + self.bs]


def gen_neutral_latent_w(out_dir, batchsize, num_imgs=10000):
    controller_path = ojoin(
        MODEL_DIR, "gan_models/controller_age015id025exp02hai04ori02gam15"
    )
    controller = Controller(controller_path)
    out_z = ojoin(out_dir, "z_latents")
    out_w = ojoin(out_dir, "w_latents")
    out_img = ojoin(out_dir, "images")
    os.makedirs(out_z, exist_ok=True)
    os.makedirs(out_w, exist_ok=True)
    os.makedirs(out_img, exist_ok=True)

    for i in tqdm(range(0, num_imgs, batchsize)):
        (
            initial_image_tensors,
            initial_latent_z,
            initial_latent_w,
        ) = controller.gen_batch(batch_size=batchsize, truncation=truncation)
        initial_latent_z = initial_latent_z.detach().cpu().numpy()
        initial_latent_w = initial_latent_w.detach().cpu().numpy()
        for j, (z, w, img) in enumerate(
            zip(initial_latent_z, initial_latent_w, initial_image_tensors)
        ):
            np.save(os.path.join(out_z, "%07d.npy" % (i + j)), z)
            np.save(os.path.join(out_w, "%07d.npy" % (i + j)), w)
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(out_img, "%07d.jpg" % (i + j)))


def main(args):
    gen_neutral_latent_w(args.out_dir, args.batch_size, args.num_imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate neutral images")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/maklemt/synthetic_imgs/GAN_control",
        help="where to save the images and latents",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_imgs", type=int, default=11000)
    args = parser.parse_args()
    main(args)
