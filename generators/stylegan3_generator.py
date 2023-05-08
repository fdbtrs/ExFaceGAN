# python3.7
"""Contains the generator class of StyleGAN.

Basically, this class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import os
import numpy as np
import torch

from typing import List, Optional, Tuple, Union
from generators import model_settings
from generators.base_generator import BaseGenerator
from generators import dnnlib
from generators import stylegan_legacy

__all__ = ["StyleGAN3Generator"]


class TruncationModule(torch.nn.Module):
    """Implements the truncation module used in StyleGAN."""

    def __init__(
        self, resolution=1024, w_space_dim=512, truncation_psi=0.7, truncation_layers=8, stylegan3=False
    ):
        super().__init__()

        self.num_layers = int(np.log2(resolution)) * 2 - 2
        if stylegan3:
            self.num_layers -= 2
        self.w_space_dim = w_space_dim
        if truncation_psi is not None and truncation_layers is not None:
            self.use_truncation = True
        else:
            self.use_truncation = False
            truncation_psi = 1.0
            truncation_layers = 0
        self.register_buffer("w_avg", torch.zeros(w_space_dim))
        layer_idx = np.arange(self.num_layers).reshape(1, self.num_layers, 1)
        coefs = np.ones_like(layer_idx, dtype=np.float32)
        coefs[layer_idx < truncation_layers] *= truncation_psi
        self.register_buffer("truncation", torch.from_numpy(coefs))

    def forward(self, w):
        if len(w.shape) == 2:
            w = w.view(-1, 1, self.w_space_dim).repeat(1, self.num_layers, 1)
        if self.use_truncation:
            w_avg = self.w_avg.view(1, 1, self.w_space_dim)
            w = w_avg + (w - w_avg) * self.truncation
        return w


class StyleGAN3Generator(BaseGenerator):
    """Defines the generator class of StyleGAN3 and StyleGAN2.

    Different from conventional GAN, StyleGAN introduces a disentangled latent
    space (i.e., W space) besides the normal latent space (i.e., Z space). Then,
    the disentangled latent code, w, is fed into each convolutional layer to
    modulate the `style` of the synthesis through AdaIN (Adaptive Instance
    Normalization) layer. Normally, the w's fed into all layers are the same. But,
    they can actually be different to make different layers get different styles.
    Accordingly, an extended space (i.e. W+ space) is used to gather all w's
    together. Taking the official StyleGAN model trained on FF-HQ dataset as an
    instance, there are
    (1) Z space, with dimension (512,)
    (2) W space, with dimension (512,)
    (3) W+ space, with dimension (18, 512)
    """

    def __init__(self, model_name, batchsize, logger=None):
        self.truncation_psi = model_settings.STYLEGAN_TRUNCATION_PSI
        self.truncation_layers = model_settings.STYLEGAN_TRUNCATION_LAYERS
        self.randomize_noise = model_settings.STYLEGAN_RANDOMIZE_NOISE
        self.model_specific_vars = []  # ["truncation.truncation"]
        super().__init__(model_name, batchsize, logger)
        self.num_layers = (int(np.log2(self.resolution)) - 1) * 2
        self.truncation = TruncationModule(
            self.resolution,
            self.w_space_dim,
            self.truncation_psi,
            self.truncation_layers,
            stylegan3=self.gan_type == "stylegan3"
        ).to(self.run_device)
        assert self.gan_type == "stylegan2" or self.gan_type == "stylegan3"

    def build(self):
        import sys

        sys.path.append(model_settings.BASE_DIR)
        self.check_attr("w_space_dim")
        self.check_attr("fused_scale")
        with dnnlib.util.open_url(self.model_path) as f:
            self.model = stylegan_legacy.load_network_pkl(f)["G_ema"]

    def load(self):
        self.logger.info(f"Loading pytorch from `{self.model_path}`.")
        self.logger.info(f"Successfully loaded!")

    def sample(self, num, latent_space_type="Z"):
        """Samples latent codes randomly.

        Args:
          num: Number of latent codes to sample. Should be positive.
          latent_space_type: Type of latent space from which to sample latent code.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

        Returns:
          A `numpy.ndarray` as sampled latend codes.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        latent_space_type = latent_space_type.upper()
        if latent_space_type == "Z":
            latent_codes = np.random.randn(num, self.latent_space_dim)
        elif latent_space_type == "W":
            latent_codes = np.random.randn(num, self.w_space_dim)
        elif latent_space_type == "WP":
            latent_codes = np.random.randn(num, self.num_layers, self.w_space_dim)
        else:
            raise ValueError(f"Latent space type `{latent_space_type}` is invalid!")

        return latent_codes.astype(np.float32)

    def preprocess(self, latent_codes, latent_space_type="Z"):
        """Preprocesses the input latent code if needed.

        Args:
          latent_codes: The input latent codes for preprocessing.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

        Returns:
          The preprocessed latent codes which can be used as final input for the
            generator.

        Raises:
          ValueError: If the given `latent_space_type` is not supported.
        """
        if not isinstance(latent_codes, np.ndarray):
            raise ValueError(f"Latent codes should be with type `numpy.ndarray`!")

        latent_space_type = latent_space_type.upper()
        if latent_space_type == "Z":
            latent_codes = latent_codes.reshape(-1, self.latent_space_dim)
        elif latent_space_type == "W":
            latent_codes = latent_codes.reshape(-1, self.w_space_dim)
        elif latent_space_type == "WP":
            latent_codes = latent_codes.reshape(-1, self.num_layers, self.w_space_dim)
        else:
            raise ValueError(f"Latent space type `{latent_space_type}` is invalid!")

        return latent_codes.astype(np.float32)

    def easy_sample(self, num, latent_space_type="Z"):
        return self.preprocess(self.sample(num, latent_space_type), latent_space_type)

    def synthesize(
        self,
        latent_codes,
        latent_space_type="Z",
        generate_style=False,
        return_lats=False,
    ):
        """Synthesizes images with given latent codes.

        One can choose whether to generate the layer-wise style codes.

        Args:
          latent_codes: Input latent codes for image synthesis.
          latent_space_type: Type of latent space to which the latent codes belong.
            Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
          generate_style: Whether to generate the layer-wise style codes. (default:
            False)
          return_lats: Whether to return dictionary with latents z, w, wp, and image
            or only the image (default: False)

        Returns:
          A dictionary whose values are raw outputs from the generator.
        """
        # if not isinstance(latent_codes, np.ndarray):
        #     raise ValueError(f"Latent codes should be with type `numpy.ndarray`!")

        results = {}

        latent_space_type = latent_space_type.upper()
        latent_codes_shape = latent_codes.shape
        # Generate from Z space.
        if latent_space_type == "Z":
            if not (
                len(latent_codes_shape) == 2
                and latent_codes_shape[0] <= self.batch_size
                and latent_codes_shape[1] == self.latent_space_dim
            ):
                raise ValueError(
                    f"Latent_codes should be with shape [batch_size, "
                    f"latent_space_dim], where `batch_size` no larger "
                    f"than {self.batch_size}, and `latent_space_dim` "
                    f"equal to {self.latent_space_dim}!\n"
                    f"But {latent_codes_shape} received!"
                )
            zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            zs = zs.to(self.run_device)
            ws = self.model.mapping(zs, None)
            wps = self.truncation(ws)
            results["z"] = latent_codes
            results["w"] = self.get_value(ws)[:, 0]
            results["wp"] = self.get_value(wps)
        # Generate from W space.
        elif latent_space_type == "W":
            if not (
                len(latent_codes_shape) == 2
                and latent_codes_shape[0] <= self.batch_size
                and latent_codes_shape[1] == self.w_space_dim
            ):
                raise ValueError(
                    f"Latent_codes should be with shape [batch_size, "
                    f"w_space_dim], where `batch_size` no larger than "
                    f"{self.batch_size}, and `w_space_dim` equal to "
                    f"{self.w_space_dim}!\n"
                    f"But {latent_codes_shape} received!"
                )
            ws = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            ws = ws.to(self.run_device)
            wps = self.truncation(ws)
            results["w"] = latent_codes
            results["wp"] = self.get_value(wps)
        # Generate from W+ space.
        elif latent_space_type == "WP":
            if not (
                len(latent_codes_shape) == 3
                and latent_codes_shape[0] <= self.batch_size
                and latent_codes_shape[1] == self.num_layers
                and latent_codes_shape[2] == self.w_space_dim
            ):
                raise ValueError(
                    f"Latent_codes should be with shape [batch_size, "
                    f"num_layers, w_space_dim], where `batch_size` no "
                    f"larger than {self.batch_size}, `num_layers` equal "
                    f"to {self.num_layers}, and `w_space_dim` equal to "
                    f"{self.w_space_dim}!\n"
                    f"But {latent_codes_shape} received!"
                )
            wps = torch.from_numpy(latent_codes).type(torch.FloatTensor)
            wps = wps.to(self.run_device)
            results["wp"] = latent_codes
        else:
            raise ValueError(f"Latent space type `{latent_space_type}` is invalid!")

        if generate_style:
            for i in range(self.num_layers):
                style = self.model.synthesis.__getattr__(
                    f"layer{i}"
                ).epilogue.style_mod.dense(wps[:, i, :])
                results[f"style{i:02d}"] = self.get_value(style)

        images = self.model.synthesis(wps, noise_mode="const")
        if return_lats:
            results["image"] = images
            return results
        return images

    def postprocess(self, images):
        images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return images
