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
    root_dir = "data/LIVEC"
    names = sio.loadmat(os.path.join(root_dir, "Data", "AllImages_release.mat"))
    mos_labels = sio.loadmat(os.path.join(root_dir, "Data", "AllMOS_release.mat"))
    mos_std = sio.loadmat(os.path.join(root_dir, "Data", "AllStdDev_release.mat"))

    img_names = names["AllImages_release"]
    mos_labels = mos_labels["AllMOS_release"][0]
    mos_std = mos_std["AllStdDev_release"][0]

    save_meta_path = "data/meta_info/meta_info_LIVEChallengeDataset.csv"
    with open(save_meta_path, "w") as f:
        csvwriter = csv.writer(f)
        header = ["img_name", "mos", "std"]
        csvwriter.writerow(header)
        for idx, name_item in enumerate(img_names):
            img_name = name_item[0][0]
            mos = mos_labels[idx]
            std = mos_std[idx]
            csvwriter.writerow([img_name, mos, std])


def get_random_splits(seed=3407):
    random.seed(seed)
    total_num = 1162
    all_img_index = list(range(total_num))
    num_splits = 10

    # ratio = [0.8, 0.2]  # train/test
    train_index = int(round(0.8 * total_num))

    save_path = f"data/train_split_info/livechallenge_82_seed{seed}.pkl"
    split_info = {}
    for i in range(num_splits):
        random.shuffle(all_img_index)
        split_info[i] = {
            "train": all_img_index[:train_index],
            "val": [],
            "test": all_img_index[train_index:],
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
    image_path = "data/LIVEC/Images"
    meta_info_file = "data/meta_info/meta_info_LIVEChallengeDataset.csv"
    save_image_path = "data/LIVEC/Images_384"

    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path, exist_ok=True)

    meta_info = pd.read_csv(meta_info_file)
    # remove first 7 training images as previous works
    # https://github.com/chaofengc/IQA-PyTorch/blob/fe95923f9c48188c65666930048597b45c9046de/pyiqa/data/livechallenge_dataset.py#L38
    meta_info = meta_info[7:].reset_index()

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
        img_name = os.path.splitext(img_name)[0] + ".bmp"
        resized_img.save(os.path.join(save_image_path, img_name))


if __name__ == "__main__":
    get_meta_info()
    get_random_splits()
    downsample_images()
