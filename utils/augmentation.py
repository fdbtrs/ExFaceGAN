import torchvision.transforms as transforms
import torch
import logging
from utils.rand_augment import RandAugment


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

to_tensor = [transforms.ToTensor(), normalize]

aug_h_flip = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]

aug_rand_4_16 = [
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=4, magnitude=16),
    transforms.ToTensor(),
    normalize,
]


def get_conventional_aug_policy(aug_type):
    """get geometric and color augmentations
    args:
        aug_type: string defining augmentation type
        operation: RA augmentation operation under testing
        num_ops: number of sequential operations under testing
        mag: magnitude under testing
    return:
        augmentation policy
    """
    aug = aug_type.lower()
    if aug == "gan_hf" or aug == "nogan_hf" or aug == "hf":
        augmentation = aug_h_flip
    elif aug == "gan_ra_4_16" or aug == "nogan_ra_4_16" or aug == "ra_4_16":
        augmentation = aug_rand_4_16
    elif aug == "totensor":
        augmentation = to_tensor
    elif aug == "mag_totensor":
        augmentation = [transforms.ToTensor()]
    else:
        logging.error("Unknown augmentation method: {}".format(aug_type))
        exit()
    return transforms.Compose(augmentation)
