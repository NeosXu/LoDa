"""
based on IQA-PyTorch: https://github.com/chaofengc/IQA-PyTorch
"""
import csv
import os
import pickle
import random

import pandas as pd
import torchvision
from PIL import Image, ImageOps
from tqdm import tqdm


def get_meta_info():
    mos_label_file = "data/spaq/Annotations/MOS and Image attribute scores.xlsx"
    scene_label_file = "data/spaq/Annotations/Scene category labels.xlsx"
    exif_label_file = "data/spaq/Annotations/EXIF_tags.xlsx"

    mos_label = pd.read_excel(mos_label_file)
    scene_label = pd.read_excel(scene_label_file)
    exif_label = pd.read_excel(exif_label_file)

    new_head = (
        mos_label.keys().tolist()
        + scene_label.keys().tolist()[1:]
        + exif_label.keys().tolist()[1:]
    )
    new_head[0] = "img_name"
    new_head[1] = "mos"
    new_head[-2] = "Time0"
    new_head[-1] = "Time1"

    save_meta_path = "data/meta_info/meta_info_SPAQDataset.csv"
    with open(save_meta_path, "w+") as sf:
        csvwriter = csv.writer(sf)
        csvwriter.writerow(new_head)
        for ridx in range(mos_label.shape[0]):
            mos_row = mos_label.loc[ridx].tolist()
            scene_row = scene_label.loc[ridx].tolist()
            exif_row = exif_label.loc[ridx].tolist()
            # print(mos_row, scene_row, exif_row)
            assert mos_row[0] == scene_row[0] == exif_row[0]
            row_label = mos_row + scene_row[1:] + exif_row[1:]
            csvwriter.writerow(row_label)


def get_random_splits(seed=3407):
    random.seed(seed)
    total_num = 11125
    all_img_index = list(range(total_num))
    num_splits = 10
    save_path = f"data/train_split_info/spaq_82_seed{seed}.pkl"

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
    spaq_image_path = "data/spaq/TestImage"
    meta_info_file = "data/meta_info/meta_info_SPAQDataset.csv"
    save_image_path = "data/spaq/TestImage_384"

    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path, exist_ok=True)

    meta_info = pd.read_csv(meta_info_file)

    preprocess = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size=down_size)]
    )

    for i in tqdm(range(meta_info.shape[0])):
        img_name = meta_info.loc[i]["img_name"]
        # PIL.Image.open will rotate jpeg images
        # https://github.com/python-pillow/Pillow/issues/4703
        img = Image.open(os.path.join(spaq_image_path, img_name))
        # There seems to be a bug in this function in pillow 9.4.0
        # but it has been fixed in pillow 9.5.0
        # https://github.com/python-pillow/Pillow/pull/6890
        # Unfortunately pillow 9.5.0 is not compatible with current env
        # Till I write this code, I cannot create an env with compatible
        # pillow 9.5.0 and torchvision by conda, so if this issue still
        # remains, you can use the below code (but there are still some
        # images being rotated) or create a new virtual env by pip then
        # remove try...except
        try:
            img_t = ImageOps.exif_transpose(img)
        except Exception:
            img_t = img
        resized_img = preprocess(img_t)
        img_name = os.path.splitext(img_name)[0] + ".png"
        resized_img.save(os.path.join(save_image_path, img_name))


if __name__ == "__main__":
    get_meta_info()
    get_random_splits()
    downsample_images()
