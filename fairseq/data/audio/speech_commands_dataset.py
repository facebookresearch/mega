"""
for speech command dataset
Adapted from https://github.com/HazyResearch/state-spaces/blob/main/src/dataloaders/sc.py
which is
adapted from https://github.com/dwromero/ckconv/blob/dc84dceb490cab2f2ddf609c380083367af21890/datasets/speech_commands.py
which is
adapted from https://github.com/patrick-kidger/NeuralCDE/blob/758d3a7134e3a691013e5cc6b7f68f277e9e6b69/experiments/datasets/speech_commands.py
"""


import os
import logging
import numpy as np
import sys

import torch
import torch.nn.functional as F

import pathlib
import tarfile
import urllib.request

import sklearn.model_selection
from fairseq.data import fairseq_dataset
# import torchaudio

from .. import FairseqDataset, BaseWrapperDataset

logger = logging.getLogger(__name__)


def pad(channel, maxlen):
    channel = torch.tensor(channel)
    out = torch.full((maxlen,), channel[-1])
    out[: channel.size(0)] = channel
    return out


def subsample(X, y, subsample_rate):
    if subsample_rate != 1:
        X = X[:, ::subsample_rate, :]
    return X, y


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + ".pt")


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            tensor_name = filename.split(".")[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out

def normalize_all_data(X_train, X_val, X_test):

    for i in range(X_train.shape[-1]):
        mean = X_train[:, :, i].mean()
        std = X_train[:, :, i].std()
        X_train[:, :, i] = (X_train[:, :, i] - mean) / (std + 1e-5)
        X_val[:, :, i] = (X_val[:, :, i] - mean) / (std + 1e-5)
        X_test[:, :, i] = (X_test[:, :, i] - mean) / (std + 1e-5)

    return X_train, X_val, X_test

def minmax_scale(tensor):
    min_val = torch.amin(tensor, dim=(1, 2), keepdim=True)
    max_val = torch.amax(tensor, dim=(1, 2), keepdim=True)
    return (tensor - min_val) / (max_val - min_val)

def mu_law_encode(audio, bits=8):
    """
    Perform mu-law companding transformation.
    """
    mu = torch.tensor(2**bits - 1)

    # Audio must be min-max scaled between -1 and 1
    audio = 2 * minmax_scale(audio) - 1

    # Perform mu-law companding transformation.
    numerator = torch.log1p(mu * torch.abs(audio))
    denominator = torch.log1p(mu)
    encoded = torch.sign(audio) * (numerator / denominator)

    # Quantize signal to the specified number of levels.
    return ((encoded + 1) / 2 * mu + 0.5).to(torch.int32)

def mu_law_decode(encoded, bits=8):
    """
    Perform inverse mu-law transformation.
    """
    mu = 2**bits - 1
    # Invert the quantization
    x = (encoded / mu) * 2 - 1

    # Invert the mu-law transformation
    x = torch.sign(x) * ((1 + mu)**(torch.abs(x)) - 1) / mu
    return x

def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (
        train_tensor,
        testval_tensor,
        train_stratify,
        testval_stratify,
    ) = sklearn.model_selection.train_test_split(
        tensor,
        stratify,
        train_size=0.7,
        random_state=0,
        shuffle=True,
        stratify=stratify,
    )

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(
        testval_tensor,
        train_size=0.5,
        random_state=1,
        shuffle=True,
        stratify=testval_stratify,
    )
    return train_tensor, val_tensor, test_tensor


class SpeechCommandsDataset(FairseqDataset):

    SUBSET_CLASSES = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]
    # SUBSET_CLASSES = [
    #     "yes",
    #     "no",
    # ]
    ALL_CLASSES = [
        "bed",
        "cat",
        "down",
        "five",
        "forward",
        "go",
        "house",
        "left",
        "marvin",
        "no",
        "on",
        "right",
        "sheila",
        "tree",
        "up",
        "visual",
        "yes",
        "backward",
        "bird",
        "dog",
        "eight",
        "follow",
        "four",
        "happy",
        "learn",
        "nine",
        "off",
        "one",
        "seven",
        "six",
        "stop",
        "three",
        "two",
        "wow",
        "zero",
    ]

    def __init__(
            self,
            partition: str,  # `train`, `val`, `test`
            length: int, # sequence length
            mfcc: bool,  # whether to use MFCC features (`True`) or raw features
            sr: int,  # subsampling rate: default should be 1 (no subsampling); keeps every kth sample
            dropped_rate: float,  # rate at which samples are dropped, lies in [0, 100.]
            path: str,
            all_classes: bool = False,
            gen: bool = False,  # whether we are doing speech generation
            discrete_input: bool = False,  # whether we are using discrete inputs
            resolution: int = 1,  # resolution of the input
    ):
        # compatible with fairseq
        if partition == 'valid':
            partition = 'val'

        self.dropped_rate = dropped_rate
        self.all_classes = all_classes
        self.gen = gen
        self.discrete_input = discrete_input
        self.resolution = resolution

        self.root = pathlib.Path(path)  # pathlib.Path("./data")
        base_loc = self.root / "processed_data"

        # import pdb; pdb.set_trace()


        if mfcc:
            data_loc = base_loc / "mfcc"
        elif gen:
            data_loc = base_loc / "gen"
        else:
            data_loc = base_loc / "raw"

            if self.dropped_rate != 0:
                data_loc = pathlib.Path(
                    str(data_loc) + "_dropped{}".format(self.dropped_rate)
                )

        if self.all_classes:
            data_loc = pathlib.Path(str(data_loc) + "_all_classes")

        if self.discrete_input:
            data_loc = pathlib.Path(str(data_loc) + "_discrete")

        # import pdb; pdb.set_trace()

        if not os.path.exists(data_loc):
            self.download()
            if not self.all_classes:
                train_X, val_X, test_X, train_y, val_y, test_y = self._process_data(mfcc)
            else:
                train_X, val_X, test_X, train_y, val_y, test_y = self._process_all(mfcc)

            # import pdb; pdb.set_trace()
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            save_data(
                data_loc,
                train_X=train_X,
                val_X=val_X,
                test_X=test_X,
                train_y=train_y,
                val_y=val_y,
                test_y=test_y,
            )

        X, y = self.load_data(data_loc, partition) # (batch, length, 1)
        if self.gen: y = y.transpose(1, 2)

        if not mfcc and not self.gen:
            X = F.pad(X, (0, 0, 0, length-16000))

        # Subsample
        if not mfcc:
            X, y = subsample(X, y, sr)

        if self.discrete_input:
            X = X.long().squeeze()

        # import pdb; pdb.set_trace()
        self.src = X
        self.tgt = y
        # super(SpeechCommands, self).__init__(X, y)

    def __getitem__(self, index):

        example = {
            "id": index,
            "source": self.src[index],
            "target": self.tgt[index],
        }

        return example

    def __len__(self):
        return len(self.src)

    def num_tokens(self, index):
        return len(self.src[index])

    def size(self, index):
        return len(self.src[index])

    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        targets = [s["target"] for s in samples]
        sizes = [len(s) for s in sources]

        def _collate(batch, resolution=1):
            # From https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            elem = batch[0]
            if isinstance(elem, torch.Tensor):
                out = None
                # if torch.utils.data.get_worker_info() is not None:
                #     # If we're in a background process, concatenate directly into a
                #     # shared memory tensor to avoid an extra copy
                #     numel = sum(x.numel() for x in batch)
                #     storage = elem.storage()._new_shared(numel)
                #     out = elem.new(storage)
                x = torch.stack(batch, dim=0, out=out)
                if resolution is not None:
                    x = x[:, ::resolution] # assume length is first axis after batch
                return x
            else:
                batch = torch.tensor(batch)
                if resolution is not None:
                    batch = batch[:, ::resolution] # assume length is first axis after batch
                return batch

        src_lengths = torch.LongTensor(sizes)
        src_tokens = _collate(sources, resolution=self.resolution)
        src_tokens = src_tokens.squeeze(-1)
        target = _collate(targets, resolution=None)

        ntokens = src_lengths.sum().item()

        id = torch.LongTensor([s["id"] for s in samples])

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,
        }

        return batch

    # def set_epoch(self, epoch):
    #     super().set_epoch(epoch)

    def download(self):
        base_loc = self.root / "SpeechCommands"
        loc = base_loc / "speech_commands.tar.gz"
        if os.path.exists(loc):
            return

        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve(
            "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", loc
        )  # TODO: Add progress bar
        with tarfile.open(loc, "r") as f:
            f.extractall(base_loc)

    def _process_all(self, mfcc):
        assert self.dropped_rate == 0, "Dropped rate must be 0 for all classes"
        base_loc = self.root / "SpeechCommands"

        with open(base_loc / "validation_list.txt", "r") as f:
            validation_list = set([line.rstrip() for line in f])

        with open(base_loc / "testing_list.txt", "r") as f:
            testing_list = set([line.rstrip() for line in f])

        # X = torch.empty(105829, 16000, 1)
        # y = torch.empty(105829, dtype=torch.long)
        train_X, val_X, test_X = [], [], []
        train_y, val_y, test_y = [], [], []

        # val_indices = []
        # test_indices = []

        batch_index = 0
        y_index = 0
        for foldername in self.ALL_CLASSES:
            print(foldername)
            loc = base_loc / foldername
            for filename in os.listdir(loc):
                audio, _ = torchaudio.load(
                    loc / filename, channels_first=False,
                )
                audio = (
                        audio / 2 ** 15
                )
                # Pad
                audio = F.pad(audio, (0, 0, 0, 16000 - audio.shape[0]))
                # # A few samples are shorter than the full length; for simplicity we discard them.
                # if len(audio) != 16000:
                #     continue
                # breakpoint()

                if str(foldername + '/' + filename) in validation_list:
                    val_X.append(audio)
                    val_y.append(y_index)
                elif str(foldername + '/' + filename) in testing_list:
                    test_X.append(audio)
                    test_y.append(y_index)
                else:
                    train_X.append(audio)
                    train_y.append(y_index)

                # X[batch_index] = audio
                # y[batch_index] = y_index

                # if filename in validation_list:
                #     val_indices.append(1)
                # else:
                #     val_indices.append(0)
                # if filename in testing_list:
                #     test_indices.append(1)
                # else:
                #     test_indices.append(0)

                batch_index += 1
            y_index += 1
        # print("Full data: {} samples".format(len(X)))
        train_X = torch.stack(train_X)
        val_X = torch.stack(val_X)
        test_X = torch.stack(test_X)
        train_y = torch.tensor(train_y, dtype=torch.long)
        val_y = torch.tensor(val_y, dtype=torch.long)
        test_y = torch.tensor(test_y, dtype=torch.long)

        # y = torch.tensor(y, dtype=torch.long)
        # val_indices = torch.tensor(val_indices, dtype=torch.bool)
        # test_indices = torch.tensor(test_indices, dtype=torch.bool)

        # If MFCC, then we compute these coefficients.
        if mfcc:
            train_X = torchaudio.transforms.MFCC(
                log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
            )(train_X.squeeze(-1)).detach()

            val_X = torchaudio.transforms.MFCC(
                log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
            )(val_X.squeeze(-1)).detach()

            test_X = torchaudio.transforms.MFCC(
                log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
            )(test_X.squeeze(-1)).detach()
            # X is of shape (batch, channels=20, length=161)
        else:
            train_X = train_X.unsqueeze(1).squeeze(-1)
            val_X = val_X.unsqueeze(1).squeeze(-1)
            test_X = test_X.unsqueeze(1).squeeze(-1)

            # X is of shape (batch, channels=1, length=16000)

        # breakpoint()
        # train_X = torch.masked_select(X, ~(val_indices + test_indices))
        # val_X = torch.masked_select(X, val_indices)
        # test_X = torch.masked_select(X, test_indices)
        # train_y = torch.masked_select(y, ~(val_indices + test_indices))
        # val_y = torch.masked_select(y, val_indices)
        # test_y = torch.masked_select(y, test_indices)

        # Normalize data
        if mfcc:
            train_X, val_X, test_X = normalize_all_data(train_X.transpose(1, 2), val_X.transpose(1, 2), test_X.transpose(1, 2))
            train_X = train_X.transpose(1, 2)
            val_X = val_X.transpose(1, 2)
            test_X = test_X.transpose(1, 2)
        else:
            train_X, val_X, test_X = normalize_all_data(train_X, val_X, test_X)

        # Print the shape of all tensors in one line
        print(
            "Train: {}, Val: {}, Test: {}".format(
                train_X.shape, val_X.shape, test_X.shape
            )
        )

        return (
            train_X,
            val_X,
            test_X,
            train_y,
            val_y,
            test_y,
        )


    def _process_data(self, mfcc):
        base_loc = self.root / "SpeechCommands"
        if self.gen:
            X = torch.empty(35628, 16000, 1)
            y = torch.empty(35628, dtype=torch.long)
        else:
            X = torch.empty(34975, 16000, 1)
            y = torch.empty(34975, dtype=torch.long)

        batch_index = 0
        y_index = 0
        for foldername in self.SUBSET_CLASSES:
            print(foldername)
            loc = base_loc / foldername
            for filename in os.listdir(loc):
                audio, _ = torchaudio.load(
                    loc / filename, channels_first=False,
                )
                # audio, _ = torchaudio.load_wav(
                #     loc / filename, channels_first=False, normalization=False
                # )  # for forward compatbility if they fix it
                audio = (
                        audio / 2 ** 15
                )  # Normalization argument doesn't seem to work so we do it manually.

                # A few samples are shorter than the full length; for simplicity we discard them.
                if len(audio) != 16000:
                    continue

                X[batch_index] = audio
                y[batch_index] = y_index
                batch_index += 1
            y_index += 1
        if self.gen:
            assert batch_index == 35628, "batch_index is {}".format(batch_index)
        else:
            assert batch_index == 34975, "batch_index is {}".format(batch_index)

        # If MFCC, then we compute these coefficients.
        if mfcc:
            X = torchaudio.transforms.MFCC(
                log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
            )(X.squeeze(-1)).detach()
            # X is of shape (batch=34975, channels=20, length=161)
        else:
            X = X.unsqueeze(1).squeeze(-1)
            # X is of shape (batch=34975, channels=1, length=16000)

        # If dropped is different than zero, randomly drop that quantity of data from the dataset.
        if self.dropped_rate != 0:
            generator = torch.Generator().manual_seed(56789)
            X_removed = []
            for Xi in X:
                removed_points = (
                    torch.randperm(X.shape[-1], generator=generator)[
                    : int(X.shape[-1] * float(self.dropped_rate) / 100.0)
                    ]
                        .sort()
                        .values
                )
                Xi_removed = Xi.clone()
                Xi_removed[:, removed_points] = float("nan")
                X_removed.append(Xi_removed)
            X = torch.stack(X_removed, dim=0)

        # Normalize data
        if mfcc:
            X = normalise_data(X.transpose(1, 2), y).transpose(1, 2)
        else:
            X = normalise_data(X, y)

        # Once the data is normalized append times and mask values if required.
        if self.dropped_rate != 0:
            # Get mask of possitions that are deleted
            mask_exists = (~torch.isnan(X[:, :1, :])).float()
            X = torch.where(~torch.isnan(X), X, torch.Tensor([0.0]))
            X = torch.cat([X, mask_exists], dim=1)

        train_X, val_X, test_X = split_data(X, y)
        train_y, val_y, test_y = split_data(y, y)

        if self.gen:
            train_y, val_y, test_y = train_X, val_X, test_X
            train_y, val_y, test_y = mu_law_encode(train_y), mu_law_encode(val_y), mu_law_encode(test_y)
            # train_X, val_X, test_X = train_X[..., :-1], val_X[..., :-1], test_X[..., :-1]
            # # Prepend zero to train_X, val_X, test_X
            # train_X = torch.cat([torch.zeros(train_X.shape[0], 1, train_X.shape[2]), train_X], dim=1)

            # train_X, val_X, test_X = torch.roll(train_X, 1, 2), torch.roll(val_X, 1, 2), torch.roll(test_X, 1, 2)
            if not self.discrete_input:
                train_X, val_X, test_X = torch.roll(mu_law_decode(train_y), 1, 2), torch.roll(mu_law_decode(val_y), 1, 2), torch.roll(mu_law_decode(test_y), 1, 2)
            else:
                train_X, val_X, test_X = torch.roll(train_y, 1, 2), torch.roll(val_y, 1, 2), torch.roll(test_y, 1, 2)
            train_X[..., 0], val_X[..., 0], test_X[..., 0] = 0, 0, 0

            assert(train_y.shape == train_X.shape)

        return (
            train_X,
            val_X,
            test_X,
            train_y,
            val_y,
            test_y,
        )

    @staticmethod
    def load_data(data_loc, partition):

        # import pdb; pdb.set_trace()
        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
            y = tensors["train_y"]
        elif partition == "val":
            X = tensors["val_X"]
            y = tensors["val_y"]
        elif partition == "test":
            X = tensors["test_X"]
            y = tensors["test_y"]
        else:
            raise NotImplementedError("the set {} is not implemented.".format(set))

        return X.transpose(1, 2), y


class SCTruncateDataset(BaseWrapperDataset):
    """Truncate a sequence by returning the first truncation_length tokens
    tailored for speech command dataset
    """

    def __init__(self, dataset, truncation_length):
        super().__init__(dataset)
        assert truncation_length is not None
        self.truncation_length = truncation_length
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        item_len = len(item['source'])
        if item_len > self.truncation_length:
            item['source'] = item['source'][:self.truncation_length]
        return item

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.truncation_length)

    def __len__(self):
        return len(self.dataset)
