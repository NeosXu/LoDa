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
    """
    Train/Val/Test split file from official github:
        https://github.com/subpic/koniq/blob/master/metadata/koniq10k_distributions_sets.csv
    """
    info_file = "data/koniq10k/koniq10k_scores_and_distributions.csv"

    save_meta_path = "data/meta_info/meta_info_KonIQ10kDataset.csv"
    with open(info_file, "r") as f, open(save_meta_path, "w+") as sf:
        csvreader = csv.reader(f)
        _ = next(csvreader)

        csvwriter = csv.writer(sf)
        new_head = [
            "img_name",
            "mos",
            "std",
            "mos_zscore",
            "c1",
            "c2",
            "c3",
            "c4",
            "c5",
            "c_total",
        ]
        csvwriter.writerow(new_head)
        for _, row in enumerate(csvreader):
            new_row = [row[0]] + row[7:10] + row[1:7]
            csvwriter.writerow(new_row)


def get_random_splits(seed=3407):
    """
    Use 10 splits as most paper
    """
    random.seed(seed)
    total_num = 10073
    all_img_index = list(range(total_num))
    num_splits = 10

    # ratio = [0.8, 0.2]  # train/test
    train_index = int(round(0.8 * total_num))

    save_path = f"./data/train_split_info/koniq10k_82_seed{seed}.pkl"
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
    image_path = "data/koniq10k/1024x768"
    meta_info_file = "data/meta_info/meta_info_KonIQ10kDataset.csv"
    save_image_path = "data/koniq10k/512x384"

    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path, exist_ok=True)

    meta_info = pd.read_csv(meta_info_file)

    preprocess = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size=down_size)]
    )

    for i in tqdm(range(meta_info.shape[0])):
        img_name = meta_info.loc[i]["img_name"]
        img = pil_loader(os.path.join(image_path, img_name))
        resized_img = preprocess(img)
        img_name = os.path.splitext(img_name)[0] + ".png"
        resized_img.save(os.path.join(save_image_path, img_name))


if __name__ == "__main__":
    get_meta_info()
    get_random_splits()
    downsample_images()
