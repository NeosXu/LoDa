"""
based on IQA-PyTorch: https://github.com/chaofengc/IQA-PyTorch
"""
import collections
import os
import random
from collections.abc import Sequence
from itertools import repeat

import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from PIL import Image


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    return Image.open(path).convert("RGB")


def transform_mapping(key, args):
    if key == "hflip" and args:
        return [PairedRandomHorizontalFlip()]
    if key == "vflip" and args:
        return [PairedRandomHorizontalFlip()]
    elif key == "random_crop":
        return [PairedRandomCrop(args)]
    elif key == "center_crop":
        return [PairedCenterCrop(args)]
    elif key == "resize":
        return [PairedResize(args)]
    elif key == "adaptive_resize":
        return [PairedAdaptiveResize(args)]
    elif key == "random_square_resize":
        return [PairedRandomSquareResize(args)]
    elif key == "random_arp_resize":
        return [PairedRandomARPResize(args)]
    elif key == "ada_pad":
        return [PairedAdaptivePadding(args)]
    elif key == "rot90" and args:
        return [PairedRandomRot90(args)]
    elif key == "randomerase":
        return [PairedRandomErasing(**args)]
    elif key == "totensor" and args:
        return [PairedToTensor()]
    elif key == "normalize" and args:
        return [PairedNormalize(args)]
    else:
        return []


def _check_pair(x):
    if isinstance(x, (tuple, list)) and len(x) >= 2:
        return True


class PairedToTensor(tf.ToTensor):
    """Pair version of center crop"""

    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        else:
            return F.to_tensor(x)

    def __call__(self, imgs):
        if _check_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = self.to_tensor(imgs[i])
            return imgs
        else:
            return self.to_tensor(imgs)


class PairedNormalize(tf.Normalize):
    """Pair version of normalize"""

    def __call__(self, imgs):
        if _check_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = super().forward(imgs[i])
            return imgs
        else:
            return super().forward(imgs)


class PairedCenterCrop(tf.CenterCrop):
    """Pair version of center crop"""

    def forward(self, imgs):
        if _check_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = super().forward(imgs[i])
            return imgs
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)


class PairedRandomCrop(tf.RandomCrop):
    """Pair version of random crop"""

    def _pad(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = img.size
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        return img

    def forward(self, imgs):
        if _check_pair(imgs):
            i, j, h, w = self.get_params(imgs[0], self.size)
            for i in range(len(imgs)):
                img = self._pad(imgs[i])
                img = F.crop(img, i, j, h, w)
                imgs[i] = img
            return imgs
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)


class PairedRandomErasing(tf.RandomErasing):
    """Pair version of random erasing"""

    def forward(self, imgs):
        if _check_pair(imgs):
            if torch.rand(1) < self.p:
                # cast self.value to script acceptable type
                if isinstance(self.value, (int, float)):
                    value = [self.value]
                elif isinstance(self.value, str):
                    value = None
                elif isinstance(self.value, tuple):
                    value = list(self.value)
                else:
                    value = self.value

                if value is not None and not (len(value) in (1, imgs[0].shape[-3])):
                    raise ValueError(
                        "If value is a sequence, it should have either a single value or "
                        f"{imgs[0].shape[-3]} (number of input channels)"
                    )

                x, y, h, w, v = self.get_params(
                    imgs[0], scale=self.scale, ratio=self.ratio, value=value
                )
                for i in range(len(imgs)):
                    imgs[i] = F.erase(imgs[i], x, y, h, w, v, self.inplace)
            return imgs
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)


class PairedRandomHorizontalFlip(tf.RandomHorizontalFlip):
    """Pair version of random hflip"""

    def forward(self, imgs):
        if _check_pair(imgs):
            if torch.rand(1) < self.p:
                for i in range(len(imgs)):
                    imgs[i] = F.hflip(imgs[i])
            return imgs
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)


class PairedRandomVerticalFlip(tf.RandomVerticalFlip):
    """Pair version of random hflip"""

    def forward(self, imgs):
        if _check_pair(imgs):
            if torch.rand(1) < self.p:
                for i in range(len(imgs)):
                    imgs[i] = F.vflip(imgs[i])
            return imgs
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)


class PairedRandomRot90(torch.nn.Module):
    """Pair version of random hflip"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, imgs):
        if _check_pair(imgs):
            if torch.rand(1) < self.p:
                for i in range(len(imgs)):
                    imgs[i] = F.rotate(imgs[i], 90)
            return imgs
        elif isinstance(imgs, Image.Image):
            if torch.rand(1) < self.p:
                imgs = F.rotate(imgs, 90)
            return imgs


class PairedResize(tf.Resize):
    """Pair version of resize"""

    def forward(self, imgs):
        if _check_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = super().forward(imgs[i])
            return imgs
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)


class PairedAdaptiveResize(tf.Resize):
    """ARP preserved resize when necessary"""

    def forward(self, imgs):
        if _check_pair(imgs):
            for i in range(len(imgs)):
                tmpimg = imgs[i]
                min_size = min(tmpimg.size)
                if min_size < self.size:
                    tmpimg = super().forward(tmpimg)
                imgs[i] = tmpimg
            return imgs
        elif isinstance(imgs, Image.Image):
            tmpimg = imgs
            min_size = min(tmpimg.size)
            if min_size < self.size:
                tmpimg = super().forward(tmpimg)
            return tmpimg


class PairedRandomARPResize(torch.nn.Module):
    """Pair version of resize"""

    def __init__(
        self, size_range, interpolation=tf.InterpolationMode.BILINEAR, antialias=None
    ):
        super().__init__()
        self.interpolation = interpolation
        self.antialias = antialias
        self.size_range = size_range
        if not (isinstance(size_range, Sequence) and len(size_range) == 2):
            raise TypeError(
                f"size_range should be sequence with 2 int. Got {size_range} with {type(size_range)}"
            )

    def forward(self, imgs):
        min_size, max_size = sorted(self.size_range)
        target_size = random.randint(min_size, max_size)
        if _check_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = F.resize(imgs[i], target_size, self.interpolation)
            return imgs
        elif isinstance(imgs, Image.Image):
            return F.resize(imgs, target_size, self.interpolation)


class PairedRandomSquareResize(torch.nn.Module):
    """Pair version of resize"""

    def __init__(
        self, size_range, interpolation=tf.InterpolationMode.BILINEAR, antialias=None
    ):
        super().__init__()
        self.interpolation = interpolation
        self.antialias = antialias
        self.size_range = size_range
        if not (isinstance(size_range, Sequence) and len(size_range) == 2):
            raise TypeError(
                f"size_range should be sequence with 2 int. Got {size_range} with {type(size_range)}"
            )

    def forward(self, imgs):
        min_size, max_size = sorted(self.size_range)
        target_size = random.randint(min_size, max_size)
        target_size = (target_size, target_size)
        if _check_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = F.resize(imgs[i], target_size, self.interpolation)
            return imgs
        elif isinstance(imgs, Image.Image):
            return F.resize(imgs, target_size, self.interpolation)


class PairedAdaptivePadding(torch.nn.Module):
    """Pair version of resize"""

    def __init__(self, target_size, fill=0, padding_mode="constant"):
        super().__init__()
        self.target_size = self.to_2tuple(target_size)
        self.fill = fill
        self.padding_mode = padding_mode

    def get_padding(self, x):
        w, h = x.size
        th, tw = self.target_size
        assert (
            th >= h and tw >= w
        ), f"Target size {self.target_size} should be larger than image size ({h}, {w})"
        pad_row = th - h
        pad_col = tw - w
        pad_l, pad_r, pad_t, pad_b = (
            pad_col // 2,
            pad_col - pad_col // 2,
            pad_row // 2,
            pad_row - pad_row // 2,
        )
        return (pad_l, pad_t, pad_r, pad_b)

    def forward(self, imgs):
        if _check_pair(imgs):
            for i in range(len(imgs)):
                padding = self.get_padding(imgs[i])
                imgs[i] = F.pad(imgs[i], padding, self.fill, self.padding_mode)
            return imgs
        elif isinstance(imgs, Image.Image):
            padding = self.get_padding(imgs)
            imgs = F.pad(imgs, padding, self.fill, self.padding_mode)
            return imgs

    def to_2tuple(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, 2))
