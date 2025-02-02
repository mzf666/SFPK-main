import sys
import random
from copy import deepcopy

import torch
from torch.nn import Module
import numpy as np


def randomize(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gpu_monitor(device):
    GB = 1024**3
    used = torch.cuda.memory_allocated(device) / GB
    total = torch.cuda.get_device_properties("cuda:0").total_memory / GB
    print(f"GPU usage = {used / total: .2%} ([{used: .2f}]/[{total: .2f}] GB)")


def func_timer(function):
    from functools import wraps

    @wraps(function)
    def function_timer(*args, **kwargs):
        import time

        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print(
            "[{name}() finished, time elapsed: {time:.5f}s]".format(
                name=function.__name__, time=t1 - t0
            )
        )
        return result

    return function_timer


def parse_second(second):
    hours = second // 3600
    second = second % 3600
    minutes = second // 60
    second = second % 60
    hms_str = f"{hours: .1f} hr {minutes: .1f} min {second: .1f} sec"
    return hms_str


@func_timer
@torch.no_grad()
def update_bn(loader, model, device=None, batch_num=1):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    print(
        f"BatchNorm running_mean and running_var updating with batch_num =  {batch_num} ..."
    )
    while batch_num >= 0:
        _, input = next(enumerate(loader))
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)
        batch_num -= 1

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class AveragedModel(Module):
    def __init__(self, model, device=None, discount=0):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer(
            "n_averaged", torch.tensor(0, dtype=torch.long, device=device)
        )
        self.register_buffer("discount", torch.tensor(discount, device=device))

        if discount == 0:

            def avg_fn(
                averaged_model_parameter, model_parameter, num_averaged, discount
            ):
                return averaged_model_parameter + (
                    model_parameter - averaged_model_parameter
                ) / (num_averaged + 1)

        else:

            def avg_fn(
                averaged_model_parameter, model_parameter, num_averaged, discount
            ):
                return averaged_model_parameter + discount * (
                    model_parameter - averaged_model_parameter
                )

        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(
                    self.avg_fn(
                        p_swa.detach(),
                        p_model_,
                        self.n_averaged.to(device),
                        self.discount,
                    )
                )
        self.n_averaged += 1


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
