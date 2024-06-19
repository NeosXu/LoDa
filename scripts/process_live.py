"""
based on IQA-PyTorch: https://github.com/chaofengc/IQA-PyTorch
"""
import csv
import os
import pickle
import random

import pandas as pd
import pyrootutils
import scipy.io as sio
import torchvision
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.dataset_utils import pil_loader


def get_meta_info():
    root_dir = "data/live_iqa"

    dmos = sio.loadmat(
        os.path.join(root_dir, "dmos_realigned.mat")
    )  # difference of mos: test - ref. lower is better
    mos = dmos["dmos_new"][0]
    org_flag = dmos["orgs"][0]

    refnames = sio.loadmat(os.path.join(root_dir, "refnames_all.mat"))
    refnames = refnames["refnames_all"][0]

    sub_folders = (
        ["jp2k"] * 227
        + ["jpeg"] * 233
        + ["wn"] * 174
        + ["gblur"] * 174
        + ["fastfading"] * 174
    )
    sub_indexes = list(range(1, 228)) + list(range(1, 234)) + list(range(1, 175)) * 3

    save_meta_path = "data/meta_info/meta_info_LIVEIQADataset.csv"
    with open(save_meta_path, "w") as f:
        csvwriter = csv.writer(f)
        header = ["ref_name", "dist_name", "mos"]
        csvwriter.writerow(header)
        for i in range(len(sub_folders)):
            ref_name = f"refimgs/{refnames[i][0]}"
            dis_name = f"{sub_folders[i]}/img{sub_indexes[i]}.bmp"
            tmpmos = mos[i]
            if org_flag[i] != 1:
                csvwriter.writerow([ref_name, dis_name, tmpmos])


def get_random_splits(seed=3407):
    random.seed(seed)
    meta_info_file = "data/meta_info/meta_info_LIVEIQADataset.csv"
    save_path = f"data/train_split_info/live_82_seed{seed}.pkl"
    ratio = 0.8

    meta_info = pd.read_csv(meta_info_file)

    ref_img_list = list(set(meta_info["ref_name"].tolist()))
    ref_img_num = len(ref_img_list)
    num_splits = 10
    train_num = int(ratio * ref_img_num)

    split_info = {}
    for i in range(num_splits):
        split_info[i] = {"train": [], "val": [], "test": []}

    for i in range(num_splits):
        random.shuffle(ref_img_list)
        train_ref_img_names = ref_img_list[:train_num]
        for j in range(meta_info.shape[0]):
            tmp_ref_name = meta_info.loc[j]["ref_name"]
            if tmp_ref_name in train_ref_img_names:
                split_info[i]["train"].append(j)
            else:
                split_info[i]["test"].append(j)
        print(
            meta_info.shape[0], len(split_info[i]["train"]), len(split_info[i]["test"])
        )
    with open(save_path, "wb") as sf:
        pickle.dump(split_info, sf)


def downsample_images(down_size=384):
    image_path = "data/live_iqa"
    meta_info_file = "data/meta_info/meta_info_LIVEIQADataset.csv"
    save_image_path = "data/live_iqa/images_384"

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
        img_name = meta_info.loc[i]["dist_name"]
        img = pil_loader(os.path.join(image_path, img_name))
        resized_img = preprocess(img)
        img_name = os.path.splitext(img_name)[0] + ".bmp"
        img_folder = os.path.split(img_name)[0]
        img_folder = os.path.join(save_image_path, img_folder)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder, exist_ok=True)
        resized_img.save(os.path.join(save_image_path, img_name))


if __name__ == "__main__":
    get_meta_info()
    get_random_splits()
    downsample_images()
