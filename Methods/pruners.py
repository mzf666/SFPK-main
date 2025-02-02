import copy
import os
import pickle
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from Utils import routine
from Utils.misc import func_timer, update_bn
from Methods.utils import masked_params, reg_fwd_hooks


class Pruner:
    def __init__(self, model):
        self.model = model
        self.total_param = sum(m.numel() for _, m, _ in masked_params(model))
        self.scores = {}
        for key, m, _ in masked_params(self.model):
            self.scores[key] = torch.ones_like(m).to(m)

    def _global_mask(self, sparsity):
        for key, mask, _ in masked_params(self.model):
            self.scores[key] *= mask

        prune_num = int((1 - sparsity) * self.total_param)
        global_scores = torch.cat([s.flatten() for s in self.scores.values()])
        if not prune_num < 1:
            threshold, _ = global_scores.kthvalue(k=prune_num)
            for key, mask, _ in masked_params(self.model):
                score = self.scores[key]
                zero = torch.tensor([0.0]).to(mask)
                one = torch.tensor([1.0]).to(mask)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        for key, mask, _ in masked_params(self.model):
            self.scores[key] *= mask

        for (key, mask, _), s in zip(masked_params(self.model), sparsity):
            score = self.scores[key]
            prune_num = int((1 - s) * score.numel())
            if not prune_num < 1:
                threshold, _ = score.flatten().kthvalue(k=prune_num)
                zero = torch.tensor([0.0]).to(mask)
                one = torch.tensor([1.0]).to(mask)
                mask.copy_(torch.where(score <= threshold, zero, one))

    @func_timer
    def mask(self, sparsity):
        if isinstance(sparsity, list):
            self._local_mask(sparsity)
        else:
            self._global_mask(sparsity)

        for _, m, p in masked_params(self.model):
            p.data *= (m != 0).float()

        update_bn(self.dataloader, self.model, self.device)

    @func_timer
    def score(self, criterion, dataloader, device):
        raise NotImplementedError

    def stats(self, model):
        sparsities = []
        total = 0
        remain = 0

        for _, m, p in masked_params(model):
            total += p.numel()
            remain += m.data.count_nonzero().item()
            sparsities += [remain / total]

        return remain, total, remain / total


class Randn(Pruner):
    def __init__(self, model):
        super(Randn, self).__init__(model)

    @func_timer
    def score(self, criterion, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        for key, _, w in masked_params(self.model):
            self.scores[key] = torch.randn_like(w).to(w)


class Mag(Pruner):
    def __init__(self, model):
        super(Mag, self).__init__(model)

    @func_timer
    def score(self, criterion, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        for key, _, w in masked_params(self.model):
            self.scores[key] = w.detach().abs()


class MagRand(Pruner):
    def __init__(self, model):
        super(MagRand, self).__init__(model)

    @func_timer
    def score(self, criterion, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        for key, _, p in masked_params(self.model):
            self.scores[key] = (p * torch.randn_like(p).to(p)).detach().abs()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, model):
        super(SNIP, self).__init__(model)

    def score(self, criterion, dataloader, device):

        self.dataloader = dataloader
        self.device = device

        # allow masks to have gradient
        for _, m, _ in masked_params(self.model):
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = self.model(data)
            criterion(output, target).backward()

        # calculate score |g * theta|
        for key, m, p in masked_params(self.model):
            self.scores[key] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for key, _, _ in masked_params(self.model):
            self.scores[key].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, model):
        super(GraSP, self).__init__(model)
        self.temp = 200
        self.eps = 1e-10

    def score(self, criterion, dataloader, device):

        self.dataloader = dataloader
        self.device = device

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = self.model(data) / self.temp
            L = criterion(output, target)

            grads = torch.autograd.grad(
                L, [p for _, _, p in masked_params(self.model)], create_graph=False
            )
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = self.model(data) / self.temp
            L = criterion(output, target)

            grads = torch.autograd.grad(
                L, [p for _, _, p in masked_params(self.model)], create_graph=True
            )
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for key, _, p in masked_params(self.model):
            self.scores[key] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for key, _, p in masked_params(self.model):
            self.scores[key].div_(norm)


# Based on https://github.com/ganguli-lab/Synaptic-Flow
class SynFlow(Pruner):
    def __init__(self, model):
        super(SynFlow, self).__init__(model)

    def score(self, criterion, dataloader, device):

        self.dataloader = dataloader
        self.device = device

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(self.model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(
            device
        )  # , dtype=torch.float64).to(device)
        self.model.train()
        output = self.model(input)
        self.model.zero_grad()
        torch.sum(output).backward()

        for key, _, p in masked_params(self.model):
            if torch.isnan(p.grad.data.sum()):
                print(p.grad.data)
            self.scores[key] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(self.model, signs)


# Sparsity-indexed ODE, see https://proceedings.mlr.press/v202/mo23c/mo23c.pdf
class SpODE(Pruner):
    def __init__(
        self,
        model,
        N=15,
        r=1,
        ode_scope="global",
        E="CE",
        G="l1",
        score_option="m",
        mask_option="one",
        mask_proc_kwargs={},
        schedule="lin",
        rt_schedule="fix",
        momentum=0,
        start=None,
        save_dir=None,
        save_ckpt=False,
    ):
        super(SpODE, self).__init__(model)

        from Methods.spode import SparsityIndexedODE

        self.ode = SparsityIndexedODE(
            None,
            N,
            r,
            ode_scope,
            E,
            G,
            score_option,
            mask_proc_kwargs,
            schedule,
            rt_schedule,
            momentum,
            save_dir,
            save_ckpt,
        )
        self.score_option = score_option
        self.mask_option = mask_option
        self.start = start

    @func_timer
    def score(self, criterion, dataloader, device):
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def _global_mask(self, sparsity):
        for key, _, p in masked_params(self.model):
            self.scores[key] *= (p != 0).float()

        prune_num = int((1 - sparsity) * self.total_param)
        global_scores = torch.cat([s.flatten() for s in self.scores.values()])
        if not prune_num < 1:
            threshold, _ = global_scores.kthvalue(k=prune_num)
            for key, mask, _ in masked_params(self.model):
                score = self.scores[key]
                zero = torch.tensor([0.0]).to(mask)

                if self.mask_option == "one":
                    one = torch.tensor([1.0]).to(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, one))

                elif self.mask_option == "mask":
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy))

                elif self.mask_option == "sign":
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy.sign()))

    def _local_mask(self, sparsity):
        for key, _, p in masked_params(self.model):
            self.scores[key] *= (p != 0).float()

        for (key, mask, _), s in zip(masked_params(self.model), sparsity):
            score = self.scores[key]
            prune_num = int((1 - s) * score.numel())
            if not prune_num < 1:
                threshold, _ = score.flatten().kthvalue(k=prune_num)
                zero = torch.tensor([0.0]).to(mask)

                if self.mask_option == "one":
                    one = torch.tensor([1.0]).to(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, one))

                elif self.mask_option == "mask":
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy))

                elif self.mask_option == "sign":
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy.sign()))

    @func_timer
    def mask(self, sparsity):
        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # allow masks to have gradient
        for _, m, _ in masked_params(self.model):
            m.requires_grad = True
            m.grad = None

        # run Sparsity-Indexed ODE
        if self.start is None:
            if isinstance(sparsity, list):
                self.start = [1.0] * len(sparsity)
            else:
                self.start = 1.0

        self.end = sparsity
        self.ode.discretization(
            self.start, self.end, self.model, self.dataloader, self.device
        )
        self.start = sparsity

        # scores <-- abs(optimized masks)
        for key, _, _ in masked_params(self.model):
            self.scores[key] = self.ode.scores[key]

        # reset other parameters
        for p in self.model.parameters():
            p.requires_grad = True

        # freeze masks to be constant
        for _, m, _ in masked_params(self.model):
            m.requires_grad = False

        # get 0-1 / soft masks
        if isinstance(sparsity, list):
            self._local_mask(sparsity)
        else:
            self._global_mask(sparsity)

        for _, m, p in masked_params(self.model):
            p.data *= (m != 0).float()
        print("Model pruned successfully.")

        update_bn(self.dataloader, self.model, self.device)

        # # apply masks
        # for _, m, p in masked_params(self.model):
        #     p.data *= (m != 0).float()


# Sparsity evolutionary Fokker-Planck-Kolmogorov Equation Pruner
class SFPK(Pruner):
    def __init__(
        self,
        model,
        N=15,
        r=1,
        ode_scope="global",
        E="CE",
        G="l1",
        score_option="m",
        mask_option="one",
        mask_proc_kwargs={},
        schedule="lin",
        rt_schedule="fix",
        momentum=0,
        n_mc=1,
        repl=0,
        repl_weighted=False,
        repl_mode="exp",
        vote_ratio=0,
        vote_mode=0,
        start=None,
        save_dir=None,
        save_ckpt=False,
    ):
        super(SFPK, self).__init__(model)

        from Methods.sfpk import SFPKSystem

        self.sfpk = SFPKSystem(
            None,
            N,
            r,
            ode_scope,
            E,
            G,
            score_option,
            mask_proc_kwargs,
            schedule,
            rt_schedule,
            momentum,
            n_mc=n_mc,
            repl_mode=repl_mode,
            repl=repl,
            repl_weighted=repl_weighted,
            vote_ratio=vote_ratio,
            vote_mode=vote_mode,
            save_dir=save_dir,
            save_ckpt=save_ckpt,
        )

        self.score_option = score_option
        self.mask_option = mask_option
        self.start = start

    def _global_mask(self, sparsity):
        for key, _, p in masked_params(self.model):
            self.scores[key] *= (p != 0).float()

        prune_num = int((1 - sparsity) * self.total_param)
        global_scores = torch.cat([s.flatten() for s in self.scores.values()])
        if not prune_num < 1:
            threshold, _ = global_scores.kthvalue(k=prune_num)
            for key, mask, _ in masked_params(self.model):
                score = self.scores[key]
                zero = torch.tensor([0.0]).to(mask)

                if self.mask_option == "one":
                    one = torch.tensor([1.0]).to(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, one))

                elif self.mask_option == "mask":
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy))

                elif self.mask_option == "sign":
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy.sign()))

    def _local_mask(self, sparsity):
        for key, _, p in masked_params(self.model):
            self.scores[key] *= (p != 0).float()

        for (key, mask, _), s in zip(masked_params(self.model), sparsity):
            score = self.scores[key]
            prune_num = int((1 - s) * score.numel())
            if not prune_num < 1:
                threshold, _ = score.flatten().kthvalue(k=prune_num)
                zero = torch.tensor([0.0]).to(mask)

                if self.mask_option == "one":
                    one = torch.tensor([1.0]).to(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, one))

                elif self.mask_option == "mask":
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy))

                elif self.mask_option == "sign":
                    mask_copy = copy.deepcopy(mask)
                    mask.copy_(torch.where(score.le(threshold), zero, mask_copy.sign()))

    @func_timer
    def score(self, criterion, dataloader, device):
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    @func_timer
    def mask(self, sparsity):
        print(f"\nSparisty renormalized skipped\n")

        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # allow masks to have gradient
        for _, m, _ in masked_params(self.model):
            m.requires_grad = True
            m.grad = None

        if self.start is None:
            if isinstance(sparsity, list):
                self.start = [1.0] * len(sparsity)
            else:
                self.start = 1.0

        if not hasattr(self, "mask_sys_ckpt"):
            self.mask_sys_ckpt = None

        # run SFPK simulation
        self.sfpk.simulation(
            sparsity,
            self.model,
            self.dataloader,
            self.device,
            mask_sys_ckpt=self.mask_sys_ckpt,
        )

        self.mask_sys_ckpt = copy.deepcopy(self.sfpk.mask_sys)
        print("\n SFPK mask system checkpoint saved for successive pruning ... \n")

        # scores <-- abs(optimized masks)
        for key, _, _ in masked_params(self.model):
            self.scores[key] = self.sfpk.scores[key]

        # reset other parameters
        for p in self.model.parameters():
            p.requires_grad = True

        # freeze masks to be constant
        for _, m, _ in masked_params(self.model):
            m.requires_grad = False

        # get 0-1 / soft masks
        if isinstance(sparsity, list):
            self._local_mask(sparsity)
        else:
            self._global_mask(sparsity)

        update_bn(self.dataloader, self.model, self.device)
