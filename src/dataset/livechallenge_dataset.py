import os

import pandas as pd
import torch.utils.data as data

from src.utils.dataset_utils import pil_loader

from .dataloader import DataloaderMode


class LIVEChallengeDataset(data.Dataset):
    def __init__(self, cfg, index, transform, mode):
        self.root = cfg.data.root
        meta_info = pd.read_csv(cfg.data.meta_info_file)
        # remove first 7 training images as previous works
        # https://github.com/chaofengc/IQA-PyTorch/blob/fe95923f9c48188c65666930048597b45c9046de/pyiqa/data/livechallenge_dataset.py#L38
        meta_info = meta_info[7:].reset_index()

        if mode is DataloaderMode.train:
            patch_num = cfg.train.patch_num
        elif mode is DataloaderMode.val:
            patch_num = cfg.val.patch_num
        elif mode is DataloaderMode.test:
            patch_num = cfg.test.patch_num
        else:
            raise ValueError(f"invalid dataloader mode {mode}")

        sample = []
        for idx in index:
            img_name = meta_info.loc[idx]["img_name"]
            img_path = os.path.join("Images", img_name)
            label = meta_info.loc[idx]["mos"]
            for _ in range(patch_num):
                sample.append((img_path, label))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = pil_loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        length = len(self.samples)
        return length
