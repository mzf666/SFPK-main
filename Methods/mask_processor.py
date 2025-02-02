from functools import partial
import time
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm as Gaussian

sns.set_style("darkgrid")

from Utils.misc import func_timer

from Layers import layers
from Methods.utils import is_masked_module, masked_params


# ======================================= Helper functions ============================================ #


def _expand_struct_weight_mask(mask, param, mask_dim=1):
    if param.dim() == 4:
        view_shape = (-1, 1, 1, 1) if mask_dim == 0 else (1, -1, 1, 1)
    else:
        view_shape = (-1, 1) if mask_dim == 0 else (1, -1)

    mask_expanded = mask.view(view_shape).expand_as(param)

    return mask_expanded


def _calc_struct_param(param, mask_dim=1):
    if param.dim() == 4:
        unmasked_dims = (1, 2, 3) if mask_dim == 0 else (0, 2, 3)
        param = param.abs().sum(unmasked_dims)
    elif param.dim() == 2:
        unmasked_dims = (1) if mask_dim == 0 else (0)
        param = param.abs().sum(unmasked_dims)
    else:
        pass

    return param


def _calc_global_mask_vec(model, mxp=True, structural=False, mask_dim=1):
    if mxp:  # mask * param as pruning score
        if structural:
            # (mxp, struct)
            global_mask_vec = []
            for _, m, p in masked_params(model):
                p.detach_()
                p_struct = _calc_struct_param(p, mask_dim)
                global_mask_vec += [(m * p_struct).flatten().abs()]

            global_mask_vec = torch.cat(global_mask_vec)

        else:
            # (mxp, un-struct)
            global_mask_vec = torch.cat(
                [(m * p.detach()).flatten() for _, m, p in masked_params(model)]
            )

    else:
        # (m, struct) or (m, un-struct), only use mask as pruning score
        global_mask_vec = torch.cat([m.flatten() for _, m, _ in masked_params(model)])

    return global_mask_vec.abs()


def _calc_local_mask_vec(mask, param, mxp=True, structural=False, mask_dim=1):
    param.detach_()

    if mxp:  # mask * param as pruning score
        if structural:
            # (mxp, struct)
            param_struct = _calc_struct_param(param, mask_dim)
            local_mask_vec = (mask * param_struct).flatten()

        else:
            # (mxp, un-struct)
            local_mask_vec = (mask * param).flatten()

    else:
        # (m, un-struct) or (m, struct), only use mask as pruning score
        local_mask_vec = mask.flatten()

    return local_mask_vec.abs()


# ======================================= Mask processors ============================================ #

# mask polarizers, see Algorithm 2, 3, and 4 in SpODE paper, https://proceedings.mlr.press/v202/mo23c/mo23c.pdf


class MaskProcessor(nn.Module):
    def __init__(
        self, one_x_m=False, mask_dim=None, local=False, eps=0.99, ratio=0.99, mxp=False
    ):
        super(MaskProcessor, self).__init__()
        self.one_x_m = one_x_m
        self.mask_dim = mask_dim
        self.structural = mask_dim in [0, 1]
        self.local = local
        self.eps = eps
        self.ratio = ratio
        self.mxp = mxp

    def _calc_ctx_vec(self, mask, param, global_mask_vec):
        if self.local:
            ctx_vec = _calc_local_mask_vec(
                mask,
                param,
                mxp=self.mxp,
                structural=self.structural,
                mask_dim=self.mask_dim,
            )
        else:
            ctx_vec = global_mask_vec

        return ctx_vec

    def _calc_ctx_mask(self, mask, param):
        # start_time = time.time()
        if self.mxp:
            param = param.detach()
            if self.structural:
                param_struct = _calc_struct_param(param, mask_dim=self.mask_dim)
                ctx_mask = mask * param_struct
            else:
                ctx_mask = mask * param
        else:
            ctx_mask = mask

        return ctx_mask.abs()

    def forward(self, itr, mask, param, sparsity, global_mask_vec):
        raise NotImplementedError


class Identity(MaskProcessor):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def forward(self, itr, mask, param, sparsity, global_mask_vec):
        return mask


class OneHot(MaskProcessor):
    def __init__(
        self, one_x_m=False, mask_dim=None, local=False, eps=0.99, ratio=0.99, mxp=False
    ):
        super(OneHot, self).__init__(one_x_m, mask_dim, local, eps, ratio, mxp)

    def forward(
        self, itr, mask, param, sparsity, global_mask_vec, global_threshold=None
    ):
        if self.local:
            ctx_vec = self._calc_ctx_vec(mask, param, global_mask_vec)
            # print("\tFinish calc ctx vec in ", time.time() - start_time, "s")
            # start_time = time.time()

            prune_num = int((1 - sparsity) * ctx_vec.numel())
            if prune_num <= 1:
                return mask

            threshold, _ = ctx_vec.kthvalue(k=prune_num)
            # print("\tFinish calc threshold in ", time.time() - start_time, "s")
            # start_time = time.time()
        else:
            if global_threshold is None:
                return mask

            threshold = global_threshold
            # print("\tFinish calc threshold in ", time.time() - start_time, "s")
            # start_time = time.time()

        ctx_mask = self._calc_ctx_mask(mask, param)
        mask_proced = ctx_mask.ge(threshold).float()
        mask_proced = mask_proced * mask

        if not self.one_x_m:
            mask_proced = mask_proced / mask.data.clone().detach()

        return mask_proced


class OneHotHard(MaskProcessor):
    def __init__(
        self, one_x_m=False, mask_dim=None, local=False, eps=0.99, ratio=0.99, mxp=False
    ):
        super(OneHotHard, self).__init__(one_x_m, mask_dim, local, eps, ratio, mxp)

    def forward(
        self, itr, mask, param, sparsity, global_mask_vec, global_threshold=None
    ):
        # start_time = time.time()
        if self.local:
            ctx_vec = self._calc_ctx_vec(mask, param, global_mask_vec)
            # print("\tFinish calc ctx vec in ", time.time() - start_time, "s")
            # start_time = time.time()

            prune_num = int((1 - sparsity) * ctx_vec.numel())
            if prune_num <= 1:
                return mask

            threshold, _ = ctx_vec.kthvalue(k=prune_num)
            # print("\tFinish calc threshold in ", time.time() - start_time, "s")
            # start_time = time.time()
        else:
            if global_threshold is None:
                return mask

            threshold = global_threshold
            # print("\tFinish calc threshold in ", time.time() - start_time, "s")
            # start_time = time.time()

        ctx_mask = self._calc_ctx_mask(mask, param)
        mask_proced = ctx_mask.ge(threshold).float()
        # print("\tFinish calc mask_proced in ", time.time() - start_time, "s")
        # print(
        #     "\terror = ",
        #     (mask_proced - torch.where(ctx_mask.le(threshold), zero, one)).norm(),
        # )
        # start_time = time.time()

        if self.one_x_m:
            mask.data = mask.data * mask_proced
        else:
            mask.data = mask_proced

        # print("\tFinish calc final mask in ", time.time() - start_time, "s")
        # start_time = time.time()

        return mask


class QuantSigmoid(MaskProcessor):
    def __init__(
        self, one_x_m=False, mask_dim=None, local=False, eps=0.99, ratio=0.99, mxp=False
    ):
        super(QuantSigmoid, self).__init__(one_x_m, mask_dim, local, eps, ratio, mxp)

    def forward(
        self,
        itr,
        mask,
        param,
        sparsity,
        global_mask_vec,
        global_ub_orig=None,
        global_lb_orig=None,
    ):
        # start_time = time.time()
        if self.local:
            ## [ctx_vec.(ratio * qt), ctx_vec.qt] --> [logit(1 - eps), logit(eps)]
            qt = 1 - sparsity
            eps = torch.Tensor([self.eps])
            ub = eps.logit().to(mask)
            lb = (1 - eps).logit().to(mask)

            ctx_vec = self._calc_ctx_vec(mask, param, global_mask_vec)
            ub_orig = ctx_vec.quantile(q=qt)
            lb_orig = ctx_vec.quantile(q=qt * self.ratio)

        else:
            qt = 1 - sparsity
            eps = torch.Tensor([self.eps])
            ub = eps.logit().to(mask)
            lb = (1 - eps).logit().to(mask)

            ub_orig = global_ub_orig
            lb_orig = global_lb_orig

        ctx_mask = self._calc_ctx_mask(mask, param)

        normalized_mask = (ctx_mask - lb_orig + 1e-12) / (ub_orig - lb_orig + 1e-12)
        mask_proced = (normalized_mask * (ub - lb) + lb).sigmoid()

        if self.one_x_m:
            mask_proced = mask_proced.detach() * mask

        return mask_proced


class GaussSigmoid(MaskProcessor):

    def __init__(
        self, one_x_m=False, mask_dim=None, local=False, eps=0.99, ratio=0.99, mxp=False
    ):
        super(GaussSigmoid, self).__init__(one_x_m, mask_dim, local, eps, ratio, mxp)

    def forward(
        self,
        itr,
        mask,
        param,
        sparsity,
        global_mask_vec,
        global_mean=None,
        global_std=None,
    ):

        if self.local:
            ctx_vec = self._calc_ctx_vec(mask, param, global_mask_vec)
            mean, std = ctx_vec.mean(), ctx_vec.std()

        else:
            mean = global_mean
            std = global_std

        ## [mean + Gauss-qt(ratio * qt) * std, mean + Gauss-qt(qt) * std]
        ## --> [logit(1 - eps), logit(eps)]
        qt = 1 - sparsity + 1e-12
        eps = torch.Tensor([self.eps])

        # qt_l, qt_u = torch.Tensor(Gaussian.ppf(q=[qt.cpu() * self.ratio, qt.cpu()])).to(mask)
        qt_l, qt_u = torch.Tensor(
            Gaussian.ppf(q=[qt.cpu().item() * self.ratio, qt.cpu().item()])
        ).to(mask)
        qt_l, qt_u = max(-1e12, qt_l), max(-1e12, qt_u)

        ub_orig = mean + qt_u * std
        lb_orig = mean + qt_l * std
        ub = eps.logit().to(mask)
        lb = (1 - eps).logit().to(mask)

        ctx_mask = self._calc_ctx_mask(mask, param)

        normalized_mask = (ctx_mask - lb_orig + 1e-12) / (ub_orig - lb_orig + 1e-12)
        mask_proced = (normalized_mask * (ub - lb) + lb).sigmoid()

        if self.one_x_m:
            mask_proced = mask_proced.detach() * mask

        return mask_proced


# ==================================================================================================== #


def load_mask_procer(mask_proc_option, **kwargs):
    mask_procer_zoo = {
        "Id": Identity,
        "qt": partial(QuantSigmoid, one_x_m=False),
        "qtm": partial(QuantSigmoid, one_x_m=True),
        "gau": partial(GaussSigmoid, one_x_m=False),
        "gaum": partial(GaussSigmoid, one_x_m=True),
        "oh": partial(OneHot, one_x_m=False),
        "ohm": partial(OneHot, one_x_m=True),
        "ohh": partial(
            OneHotHard, one_x_m=False
        ),  # NOTE: recommended, see SpODE paper, https://proceedings.mlr.press/v202/mo23c/mo23c.pdf
        "ohhm": partial(OneHotHard, one_x_m=True),
    }
    mask_procer = mask_procer_zoo[mask_proc_option](**kwargs)

    return mask_procer


# register hooks to execute mask polarization
def reg_mask_proc_hooks(itr, sparsity, model, mxp, structural, mask_dim, mask_procer):
    global_mask_vec = _calc_global_mask_vec(model, mxp, structural, mask_dim)

    if isinstance(mask_procer, (OneHotHard, OneHot)):
        prune_num = int((1 - sparsity) * global_mask_vec.numel())
        if prune_num >= 1:
            global_threshold, _ = global_mask_vec.kthvalue(k=prune_num)
        else:
            global_threshold = None

        global_kwargs = {"global_threshold": global_threshold}

    elif isinstance(mask_procer, QuantSigmoid):
        qt = 1 - sparsity
        ub_orig = global_mask_vec.quantile(q=qt)
        lb_orig = global_mask_vec.quantile(q=qt * mask_procer.ratio)
        global_kwargs = {
            "global_ub_orig": ub_orig,
            "global_lb_orig": lb_orig,
        }

    elif isinstance(mask_procer, GaussSigmoid):
        mean = global_mask_vec.mean()
        std = global_mask_vec.std()
        global_kwargs = {
            "global_mean": mean,
            "global_std": std,
        }

    else:
        raise NotImplementedError(
            "Invalid mask processor. Only support OneHot, OneHotHard, GaussSigmoid, and QuantSigmoid."
        )

    def _mask_proc(module, input, output):

        if hasattr(module, "weight_mask"):
            if structural:
                mask_proced = mask_procer(
                    itr,
                    module.weight_mask,
                    module.weight,
                    sparsity,
                    global_mask_vec,
                    **global_kwargs
                )
                weight_mask_expanded = _expand_struct_weight_mask(
                    mask_proced, module.weight, mask_dim
                )
                w = module.weight * weight_mask_expanded
            else:
                mask_proced = mask_procer(
                    itr,
                    module.weight_mask,
                    module.weight,
                    sparsity,
                    global_mask_vec,
                    **global_kwargs
                )
                w = module.weight * mask_proced
        else:
            w = module.weight

        if hasattr(module, "bias_mask"):
            mask_proced = mask_procer(
                itr,
                module.bias_mask,
                module.bias,
                sparsity,
                global_mask_vec,
                **global_kwargs
            )
            b = module.bias * mask_proced
        else:
            b = module.bias

        input = input[0]

        if isinstance(module, (nn.Linear, layers.Linear)):
            output = F.linear(input, w, b)

        elif isinstance(module, (nn.Conv2d, layers.Conv2d)):
            if module.padding_mode != "zeros":
                from torch.nn.modules.utils import _pair

                output = F.conv2d(
                    F.pad(
                        input, module._padding_repeated_twice, mode=module.padding_mode
                    ),
                    w,
                    b,
                    module.stride,
                    _pair(0),
                    module.dilation,
                    module.groups,
                )
            else:
                output = F.conv2d(
                    input,
                    w,
                    b,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                )

        return output

    for m in model.modules():
        if isinstance(
            m, (nn.Linear, nn.Conv2d, layers.Linear, layers.Conv2d)
        ) and is_masked_module(m):
            if hasattr(m, "fwd_hook"):
                m.fwd_hook.remove()
            m.fwd_hook = m.register_forward_hook(_mask_proc)


# ==================================================================================================== #
