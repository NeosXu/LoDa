"""
based on IQA-PyTorch: https://github.com/chaofengc/IQA-PyTorch
"""
import csv
import os
import pickle
import random

import pandas as pd
import pyrootutils
import torchvision
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.dataset_utils import pil_loader

"""
    The FLIVE dataset introduced by:

        Zhenqiang Ying, Haoran Niu, Praful Gupta, Dhruv Mahajan, Deepti Ghadiyaram, Alan Bovik.
        "From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality.""
        CVPR2020.

    Reference github:
        [1] https://github.com/niu-haoran/FLIVE_Database
        [2] https://github.com/baidut/PaQ-2-PiQ

    Image/patch labels are in [1], please copy the following prepare script from [2] to [1]
    because there are bugs in the script of [1]

        https://github.com/baidut/PaQ-2-PiQ/blob/master/database_prep.ipynb

    Besides, the patch labels in [1] are not complete. 9 patches from EE371R are in

        https://github.com/baidut/PaQ-2-PiQ/tree/master/database/patches
"""


def get_meta_info():
    """
    Only use whole image.
    """
    img_label_file = "data/flive/labels_image.csv"
    save_meta_path = "data/meta_info/meta_info_FLIVEDataset.csv"

    all_img_label = pd.read_csv(img_label_file)

    with open(save_meta_path, "w") as sf:
        csvwriter = csv.writer(sf)
        head = ["img_name", "mos"]
        csvwriter.writerow(head)

        # get image info
        for i in tqdm(range(all_img_label.shape[0])):
            name = all_img_label.loc[i]["name"]
            mos = all_img_label.loc[i]["mos"]

            row = [name, mos]
            csvwriter.writerow(row)


def get_random_splits(seed=3407):
    random.seed(seed)
    total_num = 39810
    all_img_index = list(range(total_num))
    num_splits = 10
    save_path = f"data/train_split_info/flive_82_seed{seed}.pkl"

    # ratio = [0.8, 0.2]  # train/val/test
    sep_index = int(round(0.8 * total_num))

    split_info = {}
    for i in range(num_splits):
        random.shuffle(all_img_index)
        split_info[i] = {
            "train": all_img_index[:sep_index],
            "val": [],
            "test": all_img_index[sep_index:],
        }
        print(
            "train num: {} | val num: {} | test num: {}".format(
                len(split_info[i]["train"]),
                len(split_info[i]["val"]),
                len(split_info[i]["test"]),
            )
        )
    with open(save_path, "wb") as sf:
        pickle.dump(split_info, sf)


def downsample_images(down_size=384):
    image_path = "data/flive"
    meta_info_file = "data/meta_info/meta_info_FLIVEDataset.csv"
    save_image_path = "data/flive/flive_384"

    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path, exist_ok=True)

    meta_info = pd.read_csv(meta_info_file)

    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=down_size,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            )
        ]
    )

    for i in tqdm(range(meta_info.shape[0])):
        img_name = meta_info.loc[i]["img_name"]
        img = pil_loader(os.path.join(image_path, img_name))
        resized_img = preprocess(img)
        img_name = os.path.splitext(img_name)[0] + ".png"
        img_folder = os.path.split(img_name)[0]
        img_folder = os.path.join(save_image_path, img_folder)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder, exist_ok=True)
        resized_img.save(os.path.join(save_image_path, img_name))


if __name__ == "__main__":
    get_meta_info()
    get_random_splits()
    downsample_images()
