import pickle

import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataloader_mode import DataloaderMode
from .flive_dataset import FLIVE_Dataset
from .kadid10k_dataset import KADID10k_Dataset
from .koniq10k_dataset import KonIQ10k_Dataset
from .live_dataset import LIVEDataset
from .livechallenge_dataset import LIVEChallengeDataset
from .spaq_dataset import SPAQ_Dataset
from .tid2013_dataset import TID2013Dataset


def get_transforms(cfg, mode):
    if (
        cfg.data.name == "livec"
        or cfg.data.name == "koniq10k"
        or cfg.data.name == "spaq"
        or cfg.data.name == "flive"
        or cfg.data.name == "kadid10k"
        or cfg.data.name == "live"
        or cfg.data.name == "tid2013"
    ):
        if mode is DataloaderMode.train:
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomCrop(size=cfg.data.patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        elif mode is DataloaderMode.val or mode is DataloaderMode.test:
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomCrop(size=cfg.data.patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            raise ValueError(f"invalid dataloader mode {mode}")
    else:
        raise ValueError(f"invalid dataset name {cfg.data.name}")

    return transforms


def get_dataset(cfg, mode, split_index=0):
    transforms = get_transforms(cfg=cfg, mode=mode)

    # prepare data index
    split_index = cfg.split_index
    with open(cfg.data.train_test_split_file, "rb") as f:
        split_idx = pickle.load(f)
    train_idx = split_idx[split_index]["train"]
    val_idx = split_idx[split_index]["val"]
    test_idx = split_idx[split_index]["test"]

    if mode is DataloaderMode.train:
        img_idx = train_idx
    elif mode is DataloaderMode.val:
        img_idx = val_idx
    elif mode is DataloaderMode.test:
        img_idx = test_idx
    else:
        raise ValueError(f"invalid dataloader mode {mode}")

    if cfg.data.name == "koniq10k":
        dataset = KonIQ10k_Dataset(
            cfg=cfg, index=img_idx, transform=transforms, mode=mode
        )
    elif cfg.data.name == "livec":
        dataset = LIVEChallengeDataset(
            cfg=cfg, index=img_idx, transform=transforms, mode=mode
        )
    elif cfg.data.name == "spaq":
        dataset = SPAQ_Dataset(cfg=cfg, index=img_idx, transform=transforms, mode=mode)
    elif cfg.data.name == "flive":
        dataset = FLIVE_Dataset(cfg=cfg, index=img_idx, transform=transforms, mode=mode)
    elif cfg.data.name == "kadid10k":
        dataset = KADID10k_Dataset(
            cfg=cfg, index=img_idx, transform=transforms, mode=mode
        )
    elif cfg.data.name == "live":
        dataset = LIVEDataset(cfg=cfg, index=img_idx, transform=transforms, mode=mode)
    elif cfg.data.name == "tid2013":
        dataset = TID2013Dataset(
            cfg=cfg, index=img_idx, transform=transforms, mode=mode
        )
    else:
        raise ValueError(f"invalid dataset name {cfg.data.name}")

    return dataset


def create_dataloader(cfg, mode, rank, split_index=0):
    data_loader = DataLoader
    dataset = get_dataset(cfg=cfg, mode=mode, split_index=split_index)
    train_use_shuffle = True
    sampler = None
    if (
        cfg.dist.device == "cuda"
        and cfg.dist.gpus != 0
        and cfg.data.divide_dataset_per_gpu
    ):
        sampler = DistributedSampler(dataset, cfg.dist.gpus, rank)
        train_use_shuffle = False
    if mode is DataloaderMode.train:
        return (
            data_loader(
                dataset=dataset,
                batch_size=cfg.train.batch_size,
                shuffle=train_use_shuffle,
                sampler=sampler,
                num_workers=cfg.train.num_workers,
                pin_memory=True,
                drop_last=True,
            ),
            sampler,
        )
    elif mode is DataloaderMode.test:
        return (
            data_loader(
                dataset=dataset,
                batch_size=cfg.test.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=cfg.test.num_workers,
                pin_memory=True,
                drop_last=False,
            ),
            sampler,
        )
    else:
        raise ValueError(f"invalid dataloader mode {mode}")
