import io
import pandas as pd
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir
import torch
import torch.nn as nn

import math
from typing import Tuple

from torch import Tensor
import torchvision
from torchvision.transforms import functional as F
from torchvision import datasets, transforms


# Based on https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/val_format.py
def TINYIMAGENET(
    root,
    train=True,
    transform=None,
    target_transform=None,
    download=True,
):
    def _exists(root, filename):
        return os.path.exists(os.path.join(root, filename))

    def _download(url, root, filename):
        datasets.utils.download_and_extract_archive(
            url=url,
            download_root=root,
            extract_root=root,
            filename=filename,
        )

    def _setup(root, base_folder):
        target_folder = os.path.join(root, base_folder, "val/")

        val_dict = {}
        with open(target_folder + "val_annotations.txt", "r") as f:
            for line in f.readlines():
                split_line = line.split("\t")
                val_dict[split_line[0]] = split_line[1]

        paths = glob.glob(target_folder + "images/*")
        paths[0].split("/")[-1]
        for path in paths:
            file = path.split("/")[-1]
            folder = val_dict[file]
            if not os.path.exists(target_folder + str(folder)):
                os.mkdir(target_folder + str(folder))

        for path in paths:
            file = path.split("/")[-1]
            folder = val_dict[file]
            dest = target_folder + str(folder) + "/" + str(file)
            move(path, dest)

        os.remove(target_folder + "val_annotations.txt")
        rmdir(target_folder + "images")

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    base_folder = "tiny-imagenet-200"

    if download and not _exists(root, filename):
        _download(url, root, filename)
        _setup(root, base_folder)

    folder = os.path.join(root, base_folder, "train" if train else "val")

    return datasets.ImageFolder(
        folder,
        transform=transform,
        target_transform=target_transform,
    )


def dimension(dataset):
    if dataset == "mnist":
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == "cifar10":
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == "cifar100":
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == "tiny_imagenet":
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == "imagenet":
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes


def _get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)


def dataloader(root, dataset, batch_size, train, workers=8, length=None, prune=False):
    if dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
        transform = _get_transform(
            size=28, padding=0, mean=mean, std=std, preprocess=False
        )
        dataset = datasets.MNIST(root, train=train, download=True, transform=transform)
    elif dataset == "cifar10":
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
        transform = _get_transform(
            size=32, padding=4, mean=mean, std=std, preprocess=train
        )
        root = f"{root}/CIFAR10"
        dataset = datasets.CIFAR10(
            root, train=train, download=True, transform=transform
        )
    elif dataset == "cifar100":
        mean, std = (0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
        root = f"{root}/CIFAR100"
        transform = _get_transform(
            size=32, padding=4, mean=mean, std=std, preprocess=train
        )
        dataset = datasets.CIFAR100(
            root, train=train, download=True, transform=transform
        )
    elif dataset == "tiny_imagenet":
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        transform = _get_transform(
            size=64, padding=4, mean=mean, std=std, preprocess=train
        )
        dataset = TINYIMAGENET(root, train=train, download=True, transform=transform)
    elif dataset == "imagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        folder = "{}/ImageNet/{}".format(root, "train" if train else "val")
        Warning(f"\nUsing ImageNet data from folder {folder}\n")
        dataset = datasets.ImageFolder(folder, transform=transform)

    # Dataloader
    use_cuda = torch.cuda.is_available()
    kwargs = {"num_workers": workers, "pin_memory": False} if use_cuda else {}
    shuffle = train is True
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)

    # use Mixup and CutMix only in ImageNet retraining
    if dataset == "imagenet" and train and not prune:
        from torch.utils.data.dataloader import default_collate

        num_classes = 1000
        mixup_alpha = 0.8
        cutmix_alpha = 1.0

        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            num_categories=num_classes,
            use_v2=True,
        )

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )

    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs
        )

    return dataloader


# NLL loss with label smoothing.
class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_mixup_cutmix(*, mixup_alpha, cutmix_alpha, num_categories, use_v2):
    transforms_module = torchvision.transforms.v2 if use_v2 else torchvision.transforms

    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(
            transforms_module.MixUp(alpha=mixup_alpha, num_categories=num_categories)
            if use_v2
            else RandomMixUp(num_classes=num_categories, p=1.0, alpha=mixup_alpha)
        )
    if cutmix_alpha > 0:
        mixup_cutmix.append(
            transforms_module.CutMix(alpha=mixup_alpha, num_categories=num_categories)
            if use_v2
            else RandomCutMix(num_classes=num_categories, p=1.0, alpha=mixup_alpha)
        )
    if not mixup_cutmix:
        return None

    return transforms_module.RandomChoice(mixup_cutmix)


class RandomMixUp(torch.nn.Module):
    """Randomly apply MixUp to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutMix(torch.nn.Module):
    """Randomly apply CutMix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(
                "Please provide a valid positive value for the num_classes."
            )
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        _, H, W = F.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s
