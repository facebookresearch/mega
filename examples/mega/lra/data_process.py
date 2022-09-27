import os
import sys

from argparse import ArgumentParser
import time
import math
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader


class PathFinderDataset(torch.utils.data.Dataset):
    """Path Finder dataset."""

    # There's an empty file in the dataset
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"data_dir {str(self.data_dir)} does not exist"
        self.transform = transform
        samples = []
        # for diff_level in ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14']:
        for diff_level in ["curv_contour_length_14"]:
            path_list = sorted(
                list((self.data_dir / diff_level / "metadata").glob("*.npy")),
                key=lambda path: int(path.stem),
            )
            assert path_list, "No metadata found"
            for metadata_file in path_list:
                with open(metadata_file, "r") as f:
                    for metadata in f.read().splitlines():
                        metadata = metadata.split()
                        image_path = Path(diff_level) / metadata[0] / metadata[1]
                        if (
                            str(Path(self.data_dir.stem) / image_path)
                            not in self.blacklist
                        ):
                            label = int(metadata[3])
                            samples.append((image_path, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(self.data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")  # Open in grayscale
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def parse_args():
    parser = ArgumentParser(description='CIFAR')
    parser.add_argument('--dataset', choices=['cifar10', 'pathfinder'], required=True)
    parser.add_argument('--resolution', type=int, default=None)
    parser.add_argument('--data_path', help='path for data file.', required=True)
    return parser.parse_args()


def dump_dataset(dataset, img_size, data_path, split, num_classes):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    try:
        os.makedirs(os.path.join(data_path, 'input'))
        os.makedirs(os.path.join(data_path, 'label'))
    except FileExistsError:
        print('File exists')

    src_path = os.path.join(data_path, 'input', split + ".src")
    label_path = os.path.join(data_path, 'label', split + ".label")

    targets = [0] * num_classes
    total = 0
    with open(src_path, 'w') as sf, open(label_path, 'w') as lf:
        for img, y in dataloader:
            img = img.view(img_size).mul(255).int().numpy().tolist()
            y = y.item()
            targets[y] += 1
            total += 1
            pixels = [str(p) for p in img]
            sf.write(' '.join(pixels) + '\n')
            lf.write(str(y) + '\n')
            sf.flush()
            lf.flush()

    print(total)
    print(targets)


def cifar10(data_path):
    dataset = datasets.CIFAR10
    num_classes = 10
    trainset = dataset(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Grayscale(),
                           transforms.ToTensor(),
                       ]))

    train_size = int(len(trainset) * 0.9)
    trainset, valset = torch.utils.data.random_split(
        trainset,
        (train_size, len(trainset) - train_size),
        generator=torch.Generator().manual_seed(42),
    )

    testset = dataset(data_path, train=False, download=False,
                     transform=transforms.Compose([
                         transforms.Grayscale(),
                         transforms.ToTensor(),
                     ]))

    dump_dataset(trainset, 1024, data_path, 'train', num_classes)
    dump_dataset(valset, 1024, data_path, 'valid', num_classes)
    dump_dataset(testset, 1024, data_path, 'test', num_classes)


def pathfinder(data_path, resolution):
    data_dir = Path(data_path) / f"pathfinder{resolution}"
    dataset = PathFinderDataset(data_dir,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))
    num_classes = 2

    val_size = int(len(dataset) * 0.1)
    test_size = int(len(dataset) * 0.1)

    trainset, valset, testset = torch.utils.data.random_split(
        dataset,
        (len(dataset) - val_size - test_size, val_size, test_size),
        generator=torch.Generator().manual_seed(42),
    )

    img_size = resolution ** 2

    dump_dataset(trainset, img_size, str(data_dir), 'train', num_classes)
    dump_dataset(valset, img_size, str(data_dir), 'valid', num_classes)
    dump_dataset(testset, img_size, str(data_dir), 'test', num_classes)


def main(args):
    dataset = args.dataset
    data_path = args.data_path
    resolution = args.resolution
    if dataset == 'cifar10':
        cifar10(data_path)
    elif dataset == 'pathfinder':
        pathfinder(data_path, resolution)
    else:
        pass


if __name__ == "__main__":
    args = parse_args()
    main(args)