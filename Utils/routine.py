import warnings
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from transformers import ViTForImageClassification

from Utils.misc import func_timer


def accuracy(output, target):
    _, pred = output.topk(5, dim=1)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    correct1 = correct[:, :1].sum().item()
    correct5 = correct[:, :5].sum().item()

    return correct1, correct5


@func_timer
def train(
    model,
    criterion,
    optimizer,
    dataloader,
    device,
    epoch,
    verbose=True,
    log_interval=200,
    n_grad_accum=1,
    n_steps=None,
    grad_clip=None,
    wandb=None,
):
    assert n_grad_accum >= 1

    model.train()
    model.to(device)
    total_loss = 0.0
    correct1, correct5 = 0, 0
    N = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        if n_grad_accum == 1 or (batch_idx + 1) % n_grad_accum == 1:
            optimizer.zero_grad()

        output = (
            model(pixel_values=data)["logits"]
            if isinstance(model, ViTForImageClassification)
            else model(data)
        )
        loss = criterion(output, target) / n_grad_accum
        loss.backward()

        if n_grad_accum == 1 or (batch_idx + 1) % n_grad_accum == 0:
            if grad_clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                grad_norm = np.nan

            optimizer.step()

        total_loss += loss.item() * data.shape[0]
        N += data.size(0)
        correct1_tmp, correct5_tmp = accuracy(output, target)
        correct1 += correct1_tmp
        correct5 += correct5_tmp
        if verbose & ((batch_idx + 1) % log_interval == 0):
            print(
                "Train Epoch: {} [{}/{} ({:.2%})], Avg loss: {:.6f},"
                " Top1-acc: {:2.2f}%, Top5-acc: {:2.2f}%, grad_norm = {}".format(
                    epoch + 1,
                    N,
                    len(dataloader.dataset),
                    N / len(dataloader.dataset),
                    total_loss / N,
                    100.0 * correct1 / N,
                    100.0 * correct5 / N,
                    grad_norm,
                )
            )

            if wandb is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_step_loss": total_loss / N,
                        "train_step_acc1": 100.0 * correct1 / N,
                        "train_step_acc5": 100.0 * correct5 / N,
                        "grad_norm": grad_norm,
                    }
                )

        if n_steps is not None and batch_idx >= n_steps:
            break

    avg_loss = total_loss / N
    acc1 = 100.0 * correct1 / N
    acc5 = 100.0 * correct5 / N

    return avg_loss, acc1, acc5


@func_timer
def prune(model, pruner, criterion, pruneloader, sparsity, device, train=True):
    model.to(device)
    if train:
        model.train()
        Warning("\npruning at model.train() mode ...")
    else:
        model.eval()
        Warning("\nWarning: pruning at model.eval() mode ...")

    pruner.score(criterion, pruneloader, device)
    pruner.mask(sparsity)


@func_timer
@torch.no_grad()
def evaluate(model, criterion, dataloader, device, verbose=True):
    start = time.time()

    model.eval()
    model.to(device)
    total = 0
    correct1, correct5 = 0, 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = (
            model(pixel_values=data)["logits"]
            if isinstance(model, ViTForImageClassification)
            else model(data)
        )
        total += criterion(output, target).item() * data.shape[0]
        correct1_tmp, correct5_tmp = accuracy(output, target)
        correct1 += correct1_tmp
        correct5 += correct5_tmp

    avg_loss = total / len(dataloader.dataset)
    acc1 = 100.0 * correct1 / len(dataloader.dataset)
    acc5 = 100.0 * correct5 / len(dataloader.dataset)

    if verbose:
        print(
            f"Eval: Avg loss: {avg_loss:.4f}, "
            f"Top 1 Acc: {correct1}/{len(dataloader.dataset)} ({acc1:.2f}%), "
            f"Top 5 Acc: {correct5}/{len(dataloader.dataset)} ({acc5:.2f}%), "
            f"time: {time.time() - start:.3f}s.\n"
        )

    return avg_loss, acc1, acc5


def save_dict(
    save_path,
    epoch,
    sparsity,
    model,
    optimizer,
    scheduler,
    val_acc1,
    val_acc5,
    args,
):
    ckpt = dict(
        val_acc={"acc1": val_acc1, "acc5": val_acc5},
        epoch=epoch,
        sparsity=sparsity,
        model=model.state_dict(),
        optimizer=optimizer.state_dict() if optimizer is not None else None,
        scheduler=scheduler.state_dict() if scheduler is not None else None,
        args=args,
    )
    torch.save(ckpt, save_path)
    print(
        f"\nEpoch {epoch} checkpoint saved, "
        f"Sparsity = {sparsity * 100: .3f}%, "
        f"Acc1 = {val_acc1}, "
        f"Acc5 = {val_acc5}.\n"
    )


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        T_warmup,
        T_max,
        eta_max=0.1,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
    ):
        self.T_warmup = T_warmup
        self.T_cosine = T_max - T_warmup
        self.eta_max = eta_max
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.T_warmup:
            return self._get_lr_warmup()
        else:
            return self._get_lr_cosine()

    def _get_lr_warmup(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.T_warmup == 0:
            return self.base_lrs
        else:
            return [
                base_lr + (self.eta_max - base_lr) * (self.last_epoch / self.T_warmup)
                for base_lr in self.base_lrs
            ]

    def _get_lr_cosine(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if (self.last_epoch - self.T_warmup) == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif (self._step_count - self.T_warmup) == 1 and (
            self.last_epoch - self.T_warmup
        ) > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(
                        (self.last_epoch - self.T_warmup) * math.pi / self.T_cosine
                    )
                )
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - self.T_warmup - 1 - self.T_cosine) % (
            2 * self.T_cosine
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_cosine)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / self.T_cosine))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.T_warmup - 1) / self.T_cosine
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
