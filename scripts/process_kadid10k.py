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


def get_meta_info():
    info_file = "data/kadid10k/dmos.csv"

    save_meta_path = "data/meta_info/meta_info_KADID10kDataset.csv"
    with open(info_file, "r") as f, open(save_meta_path, "w+") as sf:
        csvreader = csv.reader(f)
        head = next(csvreader)
        print(head)

        new_head = ["ref_name", "dist_name", "dmos", "std"]
        csvwriter = csv.writer(sf)
        csvwriter.writerow(new_head)
        for _, row in enumerate(csvreader):
            dis_name = row[0]
            ref_name = row[1]
            dmos = row[2]
            std = row[3]
            csvwriter.writerow([ref_name, dis_name, dmos, std])


def get_random_splits(seed=3407):
    random.seed(seed)
    meta_info_file = "data/meta_info/meta_info_KADID10kDataset.csv"
    save_path = f"data/train_split_info/kadid10k_82_seed{seed}.pkl"
    ratio = 0.8

    meta_info = pd.read_csv(meta_info_file)

    ref_img_list = sorted(
        list(set(meta_info["ref_name"].tolist()))
    )  # sequence initialized without sorted() is random
    ref_img_num = len(ref_img_list)
    num_splits = 10
    train_num = int(round(ratio * ref_img_num))

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
            meta_info.shape[0],
            len(split_info[i]["train"]),
            len(split_info[i]["test"]),
        )
    with open(save_path, "wb") as sf:
        pickle.dump(split_info, sf)


def downsample_images(down_size=384):
    image_path = "data/kadid10k/images"
    meta_info_file = "data/meta_info/meta_info_KADID10kDataset.csv"
    save_image_path = "data/kadid10k/images_384"

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
        img_name = os.path.splitext(img_name)[0] + ".png"
        resized_img.save(os.path.join(save_image_path, img_name))


if __name__ == "__main__":
    get_meta_info()
    get_random_splits()
    downsample_images()
