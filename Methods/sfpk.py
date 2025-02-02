import copy
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification

from Layers import layers
from Utils import load, routine, datasets
from Utils.misc import func_timer, gpu_monitor
from Methods.utils import reg_fwd_hooks, is_masked_module, masked_params
from Methods.mask_processor import (
    load_mask_procer,
    reg_mask_proc_hooks,
    _calc_global_mask_vec,
)
from Methods.spode import StatTracker, calc_norm, calc_inner_prod, calc_cosine


def apply_mask(mask_list, model):
    for layer, (_, m, _) in enumerate(masked_params(model)):
        m.data = mask_list[layer]
        m.requires_grad_(True)


def get_expand_mask(mask, param, mask_dim=0):
    if param.dim() == 4:
        view_shape = (-1, 1, 1, 1) if mask_dim == 0 else (1, -1, 1, 1)
        mask = mask.view(view_shape).expand_as(param)
        return mask

    elif param.dim() == 2:
        view_shape = (-1, 1) if mask_dim == 0 else (1, -1)
        mask = mask.view(view_shape).expand_as(param)
        return mask

    elif param.dim() == 1:  # bias mask
        return mask

    else:
        raise ValueError(
            f"Invalid param. shape = {param.shape}. for mask expansion. Only 2D or 4D allowed."
        )


def get_expand_mask_reverse(param, mask_dim=0):
    if param.dim() == 4:
        view_shape = (-1, 1, 1, 1) if mask_dim == 0 else (1, -1, 1, 1)
        mask = param.sum(dim=[i for i in range(4) if i != mask_dim])
        assert mask.dim() == 1, mask.dim()
        return mask

    elif param.dim() == 2:
        view_shape = (-1, 1) if mask_dim == 0 else (1, -1)
        mask = param.sum(dim=[i for i in range(2) if i != mask_dim])
        assert mask.dim() == 1, mask.dim()
        return mask

    elif param.dim() == 1:  # bias mask
        return param

    else:
        raise ValueError(
            f"Invalid param. shape = {param.shape}. for mask expansion. Only 2D or 4D allowed."
        )


class inf_iter:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataiter = iter(self.dataloader)

    def __next__(self):
        try:
            # prit("i m here, next")
            x, y = next(self.dataiter)
            return x, y
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            x, y = next(self.dataiter)
            return x, y


class MaskSystem:
    def __init__(self, n, model):
        self.n = n
        self.itr = 0
        self.mask_list = [None] * n  # storing updated mask
        self.energy_list = [None] * n  # storing the energy of updated mask
        self.cache = [None] * n  # storing latest mask
        for i in range(n):
            self.mask_list[i] = [
                m.data.clone().detach() for _, m, _ in masked_params(model)
            ]
            self.cache[i] = [
                m.data.clone().detach() for _, m, _ in masked_params(model)
            ]

    def __len__(self):
        return len(self.mask_list)

    def _calc_distance(self, mask1, mask2):
        if len(mask1) == 0 or len(mask2) == 0:
            return 0

        norm_sq = 0
        for m1, m2 in zip(mask1, mask2):
            assert m1.shape == m2.shape, "Error: shape mismatch!"
            norm_sq += (m1 - m2).pow(2).sum().detach()

        return norm_sq.pow(0.5)

    def _calc_norm(self, mask):
        if len(mask) == 0:
            return 0

        norm_sq = 0
        for m in mask:
            norm_sq += m.pow(2).sum().detach()

        return norm_sq.pow(0.5)

    # @func_timer
    # def calc_pairwise_distance(self):
    #     results = {}
    #     for i in range(self.n):
    #         for j in range(i + 1, self.n):
    #             tmp = self._calc_distance(self.mask_list[i], self.mask_list[j])
    #             results[f"mask_dist({i},{j})"] = tmp.cpu().item()

    #     return results

    # def calc_all_norm(self):
    #     results = {}
    #     for i in range(self.n):
    #         tmp = self._calc_norm(self.mask_list[i])
    #         results[f"mask_norm({i})"] = tmp.cpu().item()

    #     return results

    def calc_pairwise_distance(self):
        return {}

    def calc_all_norm(self):
        return {}

    @func_timer
    def _calc_similarity_exp_layer(self, model, weighted=False):
        # exp(- |mask - mask_i|^2)'
        # exp( - |mask - mask_i|^2 / |mask_i|^2) * (2) (mask - mask_i) / |mask_i|^2
        # - g = - d/dm |m|_2 / normalizer =  - 2 m / (|m|_2 * normalizer)
        if self.__len__() == 0:
            return 0.0
        else:
            similarity = 0.0
            num_layers = 0.0
            for layer, (_, m, p) in enumerate(masked_params(model)):
                num_layers += 1
                if not weighted:
                    sq_sum = m.pow(2).sum().detach()
                else:
                    p_abs = p.clone().abs().detach()
                    sq_sum = (m.pow(2) * p_abs).sum().detach()

                for i in range(self.n):
                    mask_prev = self.mask_list[i]
                    m_prev = mask_prev[layer]
                    if not weighted:  # not weighted with params
                        diff = (m - m_prev).pow(2).sum()
                        sq_sum_prev = m_prev.pow(2).sum().detach()
                    else:
                        diff = ((m - m_prev) * p_abs).pow(2).sum()
                        sq_sum_prev = (m_prev.pow(2) * p_abs).sum().detach()
                    sim_exp = torch.exp(
                        -diff / torch.sqrt(sq_sum * sq_sum_prev + 1e-12)
                    )
                    similarity = similarity + sim_exp
            return similarity / self.n / num_layers

    ##TODO: not fixed yet
    @func_timer
    def _calc_similarity_exp(self, model, weighted=False):
        # exp(- |mask - mask_i|^2)'
        # exp( - |mask - mask_i|^2 / |mask_i|^2) * (2) (mask - mask_i) / |mask_i|^2
        # - g = - d/dm |m|_2 / normalizer =  - 2 m / (|m|_2 * normalizer)
        if self.__len__() == 0:
            return 0.0
        else:
            similarity = 0.0
            for i in range(self.n):
                diff = 0.0
                sq_sum = 0.0
                sq_sum_prev = 0.0
                mask_prev = self.mask_list[i]
                for layer, (_, m, p) in enumerate(masked_params(model)):
                    m_prev = mask_prev[layer]

                    if not weighted:  # not weighted with params
                        diff = diff + (m - m_prev).pow(2).sum()
                        sq_sum = sq_sum + m.pow(2).sum().detach()
                        sq_sum_prev = sq_sum_prev + m_prev.pow(2).sum().detach()
                    else:
                        p_abs = p.clone().abs().detach()
                        diff = diff + ((m - m_prev) * p_abs).pow(2).sum()
                        sq_sum = sq_sum + (m.pow(2) * p_abs).sum().detach()
                        sq_sum_prev = (
                            sq_sum_prev + (m_prev.pow(2) * p_abs).sum().detach()
                        )

                sim_exp = torch.exp(-diff / torch.sqrt(sq_sum * sq_sum_prev + 1e-12))
                # sim_exp = torch.exp(-diff)
                similarity = similarity + sim_exp

            similarity = similarity / self.n

            return similarity

    @func_timer
    def _calc_similarity_cos(self, model, weighted=False):
        # cos(mask, mask_i)
        if self.__len__() == 0:
            return 0.0
        else:
            similarity = 0.0
            for i in range(self.n):
                diff = 0.0
                sq_sum = 0.0
                sq_sum_prev = 0.0
                mask_prev = self.mask_list[i]
                for layer, (_, m, p) in enumerate(masked_params(model)):
                    m_prev = mask_prev[layer]

                    if not weighted:  # weighted with params
                        diff = diff + (m * m_prev).sum()
                        sq_sum = sq_sum + m.pow(2).sum().detach()
                        sq_sum_prev = sq_sum_prev + m_prev.pow(2).sum().detach()
                    else:
                        p_abs = p.clone().abs().detach()
                        diff = diff + (m * m_prev * p_abs).sum()
                        sq_sum = sq_sum + (m.pow(2) * p_abs).sum().detach()
                        sq_sum_prev = (
                            sq_sum_prev + (m_prev.pow(2) * p_abs).sum().detach()
                        )

                cosine = diff / torch.sqrt(sq_sum * sq_sum_prev + 1e-12)
                similarity = similarity + cosine

            similarity = similarity / self.n

            return similarity

    def calc_similarity(self, model, weighted=False, mode="exp"):
        if mode == "exp":
            return self._calc_similarity_exp(model, weighted)
        elif mode == "cos":
            return self._calc_similarity_cos(model, weighted)
        elif mode == "exp_layer":
            return self._calc_similarity_exp_layer(model, weighted)
        # elif mode == "cos":
        # return self._calc_similarity_cos(model, weighted)
        else:
            raise NotImplementedError(f"Error: {mode} not supported!")

    @func_timer
    def _calc_similarity_struct_cos(self, model, mask_dim, weighted=False):
        # cos(mask, mask_i)
        if self.__len__() == 0:
            return 0.0
        else:
            similarity = 0.0
            for i in range(self.n):
                diff = 0.0
                sq_sum = 0.0
                sq_sum_prev = 0.0
                mask_prev = self.mask_list[i]
                for layer, (_, m, p) in enumerate(masked_params(model)):
                    m_prev = mask_prev[layer]

                    m_prev = get_expand_mask(m_prev, p, mask_dim)
                    m = get_expand_mask(m, p, mask_dim)

                    if not weighted:  # weighted with params
                        diff = diff + (m * m_prev).sum()
                        sq_sum = sq_sum + m.pow(2).sum().detach()
                        sq_sum_prev = sq_sum_prev + m_prev.pow(2).sum().detach()
                    else:
                        p_abs = p.clone().abs().detach()
                        diff = diff + (m * m_prev * p_abs).sum()
                        sq_sum = sq_sum + (m.pow(2) * p_abs).sum().detach()
                        sq_sum_prev = (
                            sq_sum_prev + (m_prev.pow(2) * p_abs).sum().detach()
                        )

                cosine = diff / torch.sqrt(sq_sum * sq_sum_prev + 1e-12)
                similarity = similarity + cosine

            similarity = similarity / self.n

            return similarity

    ##TODO: not fixed yet
    @func_timer
    def _calc_similarity_struct_exp(self, model, mask_dim, weighted=False):
        # exp(||mask - mask_i||^2)
        if self.__len__() == 0:
            return 0.0
        else:
            similarity = 0.0
            for i in range(self.n):
                diff = 0.0
                sq_sum = 0.0
                sq_sum_prev = 0.0
                mask_prev = self.mask_list[i]
                for layer, (_, m, p) in enumerate(masked_params(model)):
                    m_prev = mask_prev[layer]

                    m_prev = get_expand_mask(m_prev, p, mask_dim)
                    m = get_expand_mask(m, p, mask_dim)

                    if not weighted:  # weighted with params
                        diff = diff + (m - m_prev).pow(2).sum()
                        sq_sum = sq_sum + m.pow(2).sum().detach()
                        sq_sum_prev = sq_sum_prev + m_prev.pow(2).sum().detach()
                    else:
                        p_abs = p.clone().abs().detach()
                        diff = diff + ((m - m_prev) * p_abs).pow(2).sum()
                        sq_sum = sq_sum + (m.pow(2) * p_abs).sum().detach()
                        sq_sum_prev = (
                            sq_sum_prev + (m_prev.pow(2) * p_abs).sum().detach()
                        )

                exp_sim = torch.exp(-diff / (sq_sum_prev + 1e-12))
                # exp_sim = torch.exp(-diff)
                similarity = similarity + exp_sim

            similarity = similarity / self.n

            return similarity

    @func_timer
    def _calc_similarity_struct_exp_layer(self, model, mask_dim, weighted=False):
        if self.__len__() == 0:
            return 0.0
        else:
            # layer_m_p_list = list(masked_params(model))
            # @torch.jit.script
            # def inner_f(layer_m_p_list):
            similarity = 0.0
            num_layers = 0.0
            for layer, (_, m, p) in enumerate(masked_params(model)):
                num_layers += 1
                if not weighted:
                    sq_sum = m.pow(2).sum().detach()
                else:
                    p_abs = get_expand_mask_reverse(p, mask_dim)
                    p_abs = p_abs.abs().detach()

                    sq_sum = (m.pow(2) * p_abs).sum().detach()
                for i in range(self.n):
                    mask_prev = self.mask_list[i]
                    m_prev = mask_prev[layer]
                    if not weighted:  # not weighted with params
                        diff = (m - m_prev).pow(2).sum()
                        sq_sum_prev = m_prev.pow(2).sum().detach()
                    else:
                        diff = ((m - m_prev) * p_abs).pow(2).sum()
                        sq_sum_prev = (m_prev.pow(2) * p_abs).sum().detach()
                    sim_exp = torch.exp(
                        -diff / torch.sqrt(sq_sum * sq_sum_prev + 1e-12)
                    )
                    similarity = similarity + sim_exp
            return similarity / self.n / num_layers
            # return inner_f(layer_m_p_list)

    def calc_similarity_struct(self, model, mask_dim, weighted=False, mode="exp"):
        if mode == "exp":
            return self._calc_similarity_struct_exp(model, mask_dim, weighted)
        elif mode == "cos":
            return self._calc_similarity_struct_cos(model, mask_dim, weighted)
        elif mode == "exp_layer":
            return self._calc_similarity_struct_exp_layer(model, mask_dim, weighted)
        else:
            raise NotImplementedError(f"Error: {mode} not supported!")

    def apply(self, i, model):
        for layer, (_, m, _) in enumerate(masked_params(model)):
            assert (
                m.data.dtype == self.mask_list[i][layer].dtype
            ), "Apply Error: dtype of model and masksys mismatch!"
            assert (
                m.data.shape == self.mask_list[i][layer].shape
            ), "Apply Error: shape of model and masksys mismatch!"
            m.data = self.mask_list[i][layer].data.detach().clone()
            m.requires_grad_(True)

    @torch.no_grad()
    def update_to_cache(self, i, delta, dt):
        for layer in range(len(self.mask_list[i])):
            if isinstance(dt, list):
                # local SFPK
                self.cache[i][layer].data = (
                    self.mask_list[i][layer].data
                    + delta[layer].data.detach().clone() * dt[layer]
                )
            else:
                # global SFPK
                self.cache[i][layer].data = (
                    self.mask_list[i][layer].data
                    + delta[layer].data.detach().clone() * dt
                )

    def update(self):
        self.itr += 1
        for i in range(self.n):
            for layer in range(len(self.mask_list[i])):
                self.mask_list[i][layer].data = (
                    self.cache[i][layer].data.detach().clone()
                )

    def replace(self, i, mask_new):
        assert len(mask_new) == len(self.mask_list[i])
        for layer in range(len(self.mask_list[i])):
            self.mask_list[i][layer].data = mask_new[layer].data.detach().clone()


class SFPKSystem:

    def __init__(
        self,
        mask_dim=None,
        N=15,
        r=1.1,
        ode_scope="global",
        E="CE",
        G="l2",
        score_option="mp",
        mask_proc_kwargs={},
        schedule="lin",
        rt_schedule="fix",
        momentum=0.0,
        n_mc=1,
        repl=0.01,
        repl_weighted=False,
        repl_mode="exp",
        vote_ratio=None,
        vote_mode="hard",
        save_dir=None,
        save_ckpt=False,
    ):
        self.N = int(N)  # num. of bins
        assert r > 1, f"Error: r_init = {r} <= 1!"
        self.r_init = r
        self.r_prev = self.r_init  # for auto / autor search
        self.r_min = 1.01
        self.r_max = 4
        self.n_mc = n_mc  # Size of mask particle system
        self.repl = repl
        self.repl_weighted = repl_weighted
        self.repl_mode = repl_mode
        self.evol_score_top_k = int(np.ceil(vote_ratio * n_mc))
        self.vote_mode = vote_mode
        self.criterion = nn.CrossEntropyLoss()  # label smoothing is not good
        print("\n\n\tWarning: Using vanilla cross-entropy energy function ...\n\n")

        self.E = E  # Energy to preserve
        self.G = G  # sparsity to reduce
        self.score_option = score_option
        self.mask_proc_option = mask_proc_kwargs["mask_proc_option"]
        self.mask_proc_eps = mask_proc_kwargs["mask_proc_eps"]
        self.mask_proc_ratio = mask_proc_kwargs["mask_proc_ratio"]
        self.mask_proc_score_option = mask_proc_kwargs["mask_proc_score_option"]
        self.mask_proc_mxp = mask_proc_kwargs["mask_proc_mxp"]

        self.schedule = schedule
        self.rt_schedule = rt_schedule

        if rt_schedule == "auto":
            self.multipliers = [0.99**i for i in [-3, -2, -1, 0, 1, 2, 3]]
            print(
                f"Auto r_t seach: r_init = {self.r_prev}, alphas = {self.multipliers}."
            )
        elif rt_schedule == "autor":
            self.multipliers = [0.9**i for i in [-3, -2, -1, 0, 1, 2, 3]]
            print(
                f"Auto-r r_t seach: r_init = {self.r_prev}, alphas = {self.multipliers}."
            )

        self.momentum = momentum
        self.local_ode = ode_scope == "local"

        self.structural = mask_dim in [0, 1]
        self.mask_dim = mask_dim
        if self.structural:
            print("\tUsing structural ProbODE-sys...")
        else:
            print("\tUsing unstructural ProbODE-sys ...")

        self.save_dir = save_dir
        self.save_ckpt = save_ckpt
        if save_dir:
            print("\tWarning: tracking SFPK stats ...")
            if save_ckpt:
                self.G_mile_stones = list(
                    reversed([0.5, 0.28, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.02])
                )
                print(
                    f"\n\nWarning: saving intermediate model ckpts sp in {self.G_mile_stones} ...\n\n"
                )

        # initialize pruning scores
        self.scores = {}

    # ======================================= General helpers =================================================== #

    def _save_ckpt(self, itr, model, G_stats, G_ideal, E_stats):
        model_tmp = copy.deepcopy(model)

        i_argmin = torch.cat([e.view(1, 1) for e in E_stats]).argmin()
        self.mask_sys.apply(i_argmin, model_tmp)
        self._permanent_polarization(model_tmp, G_stats[i_argmin])

        ckpt = dict(
            model=model_tmp.state_dict(),
            mask_sys=self.mask_sys,
            scores=self.scores,
            sparsity=G_stats[i_argmin],
            itr=itr + 1,
        )
        torch.save(ckpt, f"{self.save_dir}/sp{G_ideal}.pkl")
        print(
            f"\tSpODE checkpoint of sparsity = {G_ideal} with reset fwd-hooks saved successfully ...\n"
        )

    def _update_stat_tracker(
        self,
        i,
        itr,
        dt,
        r_t,
        G,
        E,
        penalty,
        delta,
        neg_dE,
        neg_dG,
        mask_stat_norm=None,
        mask_stat_dist=None,
        verbose=True,
    ):

        dt = dt[0] if isinstance(dt, list) else dt

        stat_vals = {
            "dt": dt,
            "r_t": r_t,
            "sparsity": G.item(),
            "energy": E.item(),
            "penalty": penalty.item(),
            "cos(delta, - dE)": calc_cosine(delta, neg_dE).item(),
            "cos(delta, - dG)": calc_cosine(delta, neg_dG).item(),
            "cos(dG, dE)": calc_cosine(neg_dE, neg_dG).item(),
            "norm(dG)": calc_norm(neg_dG).item(),
            "norm(dE)": calc_norm(neg_dE).item(),
        }

        if mask_stat_norm is not None:
            stat_vals.update(mask_stat_norm)
        if mask_stat_dist is not None:
            stat_vals.update(mask_stat_dist)

        self.stat_trackers[i].update_stats(stat_vals)
        if verbose:
            self.stat_trackers[i].show_stats(itr, self.N, stat_vals, ncols=4)

    # @func_timer
    def get_fmt_stat(self, i, itr, dt, r_t, G, E, penalty, delta, neg_dE, neg_dG):
        dt = dt[0] if isinstance(dt, list) else dt

        msg = f"itr = {itr}, MC-num = {i + 1} | dt = {dt: .4f}, r_t = {r_t: .4f} | "
        msg += f"repl = {self.repl: .4f} wp = {self.repl_weighted} | "
        msg += f"G = {G.item(): .2%}, E = {E.item(): .4f}, K_{self.repl_mode} = {penalty.item(): .10f} | "
        # msg += f"<delta, -dE> = {calc_cosine(delta, neg_dE).item(): .4f} | "
        # msg += f"<delta, -dG> = {calc_cosine(delta, neg_dG).item(): .4f} | "
        # msg += f"<dG> = {calc_norm(neg_dG).item(): .4f} | "
        # msg += f"<dE> = {calc_norm(neg_dE).item(): .4f} | "
        # msg += f"<delta> = {calc_norm(delta).item(): .4f} | "

        return msg

    def energy_func(self, model, x, y):
        # E = F.cross_entropy(model(x), y)
        # start_time = time.time()
        output = (
            model(pixel_values=x)["logits"]
            if isinstance(model, ViTForImageClassification)
            else model(x)
        )
        # print(f"Time taken for model forward pass: {time.time() - start_time}")

        E = self.criterion(output, y)

        return E

    def _permanent_polarization(self, model, sparsity):
        ## load polarizor
        mask_procer_kwargs = dict(
            mask_dim=self.mask_dim,
            local=self.local_prune,
            eps=self.mask_proc_eps,
            ratio=self.mask_proc_ratio,
            mxp=self.mask_proc_mxp,
        )
        mask_procer = load_mask_procer(self.mask_proc_option, **mask_procer_kwargs)

        ## polarize mask values
        global_mask_vec = _calc_global_mask_vec(
            model, self.mask_proc_mxp, self.structural, self.mask_dim
        )
        for _, m, p in masked_params(model):
            mask_proced = mask_procer(None, m, p, sparsity, global_mask_vec)
            m.data = mask_proced

        ## polarizor hook --> mask mulitplicative hook
        reg_fwd_hooks(model, self.structural, self.mask_dim)

        print("\tMasks permanently polarized successfully ...")

    # =========================================== Calc. SFPK-Stats =============================================== #

    def _calc_r_t(self, itr, neg_dE):
        if self.rt_schedule == "fix":
            r_t = self.r_init

        elif self.rt_schedule == "invexp":
            if self.local_ode:
                raise NotImplementedError
            else:
                # itr starts from 0
                ratio = self.r_init / self.r_max
                r_t = self.r_max * ratio ** (1 - (itr + 1) / self.N)

        elif self.rt_schedule == "hess":
            # r_t = const * norm(dE)
            if self.local_ode:
                raise NotImplementedError
            else:
                dE_norm = calc_norm(neg_dE)
                # r_t = 1 + (self.r_max - 1) * dE_norm
                ## Initialize const s.t. 1 + const * dE_norm_0 = r_init
                if itr == 0:
                    self.r_t_normalizer = (self.r_init - 1) / (dE_norm + 1e-12)
                    # self.r_t_normalizer = self.r_init / (dE_norm + 1e-12)
                    print(f"\nr_init_const = {self.r_t_normalizer}\n")

                # r_t = 1 + (self.r_max - 1) * dE_norm # w/o normalization
                r_t = 1 + self.r_t_normalizer * dE_norm
                # r_t = self.r_t_normalizer * dE_norm
                # r_t = max(self.r_min, r_t.item())
                r_t = max(self.r_min, r_t.item())
                r_t = min(self.r_max, r_t)

        else:
            raise NotImplementedError

        return r_t

    def _calc_dt(self, itr, start, end, G=None, neg_dE=None, neg_dG=None, G_local=None):
        self.local_prune = isinstance(start, list)

        if self.schedule == "lin":
            if self.local_prune:
                dt = [(s - e) / self.N for s, e in zip(start, end)]
            else:
                dt = (start - end) / self.N

        elif self.schedule == "exp":
            if self.local_prune:
                dt = [
                    e
                    * (
                        (s / e) ** (1 - itr / self.N)
                        - (s / e) ** (1 - (itr + 1) / self.N)
                    )
                    for s, e in zip(start, end)
                ]
            else:
                # itr starts from 0
                ratio = start / end
                dt = end * (
                    ratio ** (1 - itr / self.N) - ratio ** (1 - (itr + 1) / self.N)
                )

        elif self.schedule == "invexp":
            if self.local_prune:
                dt = [
                    s
                    * (
                        (e / s) ** (1 - (itr + 1) / self.N)
                        - (e / s) ** (1 - itr / self.N)
                    )
                    for s, e in zip(start, end)
                ]
            else:
                # itr starts from 0
                ratio = end / start
                dt = start * (
                    ratio ** (1 - (itr + 1) / self.N) - ratio ** (1 - itr / self.N)
                )

        elif self.schedule == "hess":

            if self.local_prune:
                # dt = const / norm(dE)
                dE_norm = calc_norm(neg_dE).item()
                dt_exp_max = (
                    end[0] * (start[0] / end[0]) ** 1 / (self.N / 5)
                )  # cifar100-resnet20, vgg16_bn
                # dt_exp_max = 1 # cifar100-wrn20/
                if itr == 0:
                    # calibration: dt_lin_init / norm(dE_0) * dt_normalizer = dt_exp_init
                    # self.dt_normalizer = 1
                    dt_lin_init = (start[0] - end[0]) / self.N
                    self.dt_normalizer = dE_norm * dt_exp_max / (dt_lin_init + 1e-12)

                dt = [
                    min(
                        (G.item() - e)
                        / ((self.N - itr) * dE_norm + 1e-12)
                        * self.dt_normalizer,
                        dt_exp_max,
                    )
                    for e in end
                ]
            else:
                ## dt = const / norm(dE)
                dE_norm = calc_norm(neg_dE).item()
                dt_exp_max = (
                    end * (start / end) ** 1 / (self.N / 5)
                )  # cifar100-resnet20, vgg16_bn
                ## dt_exp_max = 1 # cifar100-wrn20/
                if itr == 0:
                    # calibration: dt_lin_init / norm(dE_0) * dt_normalizer = dt_exp_init
                    # self.dt_normalizer = 1
                    dt_lin_init = (start - end) / self.N
                    self.dt_normalizer = dE_norm * dt_exp_max / (dt_lin_init + 1e-12)

                dt = (
                    (G.item() - end)
                    / ((self.N - itr) * dE_norm + 1e-12)
                    * self.dt_normalizer
                )
                dt = min(dt, dt_exp_max)

        elif self.schedule == "hessexp":
            # dt = const / norm(dE)
            dE_norm = calc_norm(neg_dE).item()
            start = [G.item()] * len(start) if self.local_prune else G.item()
            if itr == 0:
                # calibration: dt_exp / norm(dE_0) * dt_normalizer = 1
                # dt_init = end * (start / end) ** (1 / self.N)
                self.dt_normalizer = dE_norm

            if self.local_prune:
                dt = [
                    e * (s / e) ** (1 / (self.N - itr)) * self.dt_normalizer
                    for s, e in zip(start, end)
                ]
            else:
                dt = end * (start / end) ** (1 / (self.N - itr)) * self.dt_normalizer

        elif self.schedule == "adaLin":
            # adaptive
            if self.local_prune:
                dt = [(G.item() - e) / (self.N - itr) for e in end]
            else:
                dt = (G.item() - end) / (self.N - itr)

        elif self.schedule == "adaExp":
            start = [G.item()] * len(start) if self.local_prune else G.item()
            if self.local_prune:
                dt = [e * (s / e) ** (1 / (self.N - itr)) for s, e in zip(start, end)]
            else:
                # adaptive
                ratio = start / end
                dt = end * ratio ** (1 / (self.N - itr))

        else:
            raise NotImplementedError

        return dt

    def _calc_G(self, model, eps=1e-12):
        G_local = []

        ## Local SFPK ##
        if self.local_ode:
            if self.G == "l1":
                # sparsity = mask l1-norm
                if self.structural:
                    G = []
                    for _, m, p in masked_params(model):
                        if p.dim() in [2, 4]:
                            node_size = (
                                p.data[0, :].numel()
                                if self.mask_dim == 0
                                else p.data[:, 0].numel()
                            )
                            G += [m.norm(p=1) * node_size / p.numel()]
                        else:
                            G += [m.norm(p=1) / p.numel()]

                else:
                    G = [m.norm(p=1) / p.numel() for _, m, p in masked_params(model)]

            elif self.G == "l2":
                # sparsity = mask l2-norm
                if self.structural:
                    G = []
                    for _, m, p in masked_params(model):
                        if p.dim() in [2, 4]:
                            node_size = (
                                p.data[0, :].numel()
                                if self.mask_dim == 0
                                else p.data[:, 0].numel()
                            )
                            G += [m.norm(p=2) * (node_size**0.5) / p.numel() ** 0.5]
                        else:
                            G += [m.norm(p=2) / p.numel() ** 0.5]

                else:
                    G = [
                        m.norm(p=2) / p.numel() ** 0.5
                        for _, m, p in masked_params(model)
                    ]

            else:
                raise NotImplementedError

            # print(f'G = \n {[g.item() for g in G]}\n')
            return sum(G), G

        ## Global SFPK ##
        else:
            if self.G == "l1":
                # sparsity = mask l1-norm
                normalizer = sum([p.numel() for _, _, p in masked_params(model)])
                if self.structural:
                    G = 0.0
                    for _, m, p in masked_params(model):
                        if p.dim() in [2, 4]:
                            node_size = (
                                p.data[0, :].numel()
                                if self.mask_dim == 0
                                else p.data[:, 0].numel()
                            )
                            G_tmp = m.norm(p=1) * node_size
                        else:
                            G_tmp = m.norm(p=1)

                        G += G_tmp
                        G_local.append(G_tmp / p.numel())

                    G /= normalizer
                else:
                    G = (
                        sum([m.norm(p=1) for _, m, _ in masked_params(model)])
                        / normalizer
                    )

            elif self.G == "l2":
                # sparsity = mask l2-norm
                normalizer = sum([p.numel() for _, _, p in masked_params(model)]) ** 0.5
                if self.structural:
                    G = 0.0
                    for _, m, p in masked_params(model):
                        if p.dim() in [2, 4]:
                            node_size = (
                                p.data[0, :].numel()
                                if self.mask_dim == 0
                                else p.data[:, 0].numel()
                            )
                            # G_tmp = (m.norm(p=2) ** 2) * node_size
                            G_tmp = m.pow(2).sum() * node_size
                        else:
                            # G_tmp = m.norm(p=2) ** 2
                            G_tmp = m.pow(2).sum()

                        G += G_tmp
                        G_local.append(G_tmp**0.5 / p.numel() ** 0.5)

                    G = torch.sqrt(G + eps) / normalizer
                else:
                    # G = torch.sqrt(
                    #     sum([m.norm(p=2) ** 2 for _, m, _ in masked_params(model)])
                    #     + eps
                    # ) / (normalizer + eps)
                    G = torch.sqrt(
                        sum([m.pow(2).sum() for _, m, _ in masked_params(model)]) + eps
                    ) / (normalizer + eps)

            elif self.G[0] == "l":
                # sparsity = mask lp-norm
                Lp = float(self.G[1:])
                normalizer = sum([p.numel() for _, _, p in masked_params(model)]) ** (
                    1 / Lp
                )
                if self.structural:
                    G = 0.0
                    for _, m, p in masked_params(model):
                        if p.dim() in [2, 4]:
                            node_size = (
                                p.data[0, :].numel()
                                if self.mask_dim == 0
                                else p.data[:, 0].numel()
                            )
                            G_tmp = (m.norm(p=Lp) ** Lp) * node_size
                        else:
                            G_tmp = m.norm(p=Lp) ** Lp

                        G += G_tmp
                        G_local.append(G_tmp ** (1 / Lp) / p.numel() ** (1 / Lp))

                    G = (G + eps) ** (1 / Lp) / normalizer
                else:
                    G = (
                        sum([m.norm(p=Lp) ** Lp for _, m, _ in masked_params(model)])
                        + eps
                    ) ** (1 / Lp) / (normalizer + eps)

            else:
                raise NotImplementedError

            return G, G_local

    def _calc_neg_dG(self, model):
        G, G_local = self._calc_G(model)
        neg_dG = torch.autograd.grad(
            -G, [m for _, m, _ in masked_params(model)], allow_unused=True
        )
        G.detach_()
        return G, neg_dG, G_local

    @func_timer
    def _calc_E(self, itr, G, model, x, y):
        ## Mark down un-polarized mask values
        if self.mask_proc_option in ["ohh", "ohhm"]:
            masks_prev = [m.data.clone() for _, m, _ in masked_params(model)]

        ## Polarize masks
        mask_procer_kwargs = dict(
            mask_dim=self.mask_dim,
            local=self.local_prune,
            eps=self.mask_proc_eps,
            ratio=self.mask_proc_ratio,
            mxp=self.mask_proc_mxp,
        )
        self.mask_procer = load_mask_procer(self.mask_proc_option, **mask_procer_kwargs)
        reg_mask_proc_hooks(
            itr,
            G,
            model,
            self.mask_proc_mxp,
            self.structural,
            self.mask_dim,
            self.mask_procer,
        )

        E = self.energy_func(model, x, y)

        ## Un-polarize masks for SFPK update
        if self.mask_proc_option in ["ohh", "ohhm"]:
            for i, (_, m, _) in enumerate(masked_params(model)):
                m.data.copy_(masks_prev[i])

        return E

    @func_timer
    def _calc_neg_dE(self, itr, G, model, x, y):
        ## Mark down un-polarized mask values
        start_time = time.time()
        if self.mask_proc_option in ["ohh", "ohhm"]:
            masks_prev = [m.data.clone() for _, m, _ in masked_params(model)]
        # print(f"end mask bkup in {time.time() - start_time}s")
        start_time = time.time()

        for p in model.parameters():
            p.requires_grad_(False)

        ## Polarize masks
        mask_procer_kwargs = dict(
            mask_dim=self.mask_dim,
            local=self.local_prune,
            eps=self.mask_proc_eps,
            ratio=self.mask_proc_ratio,
            mxp=self.mask_proc_mxp,
        )
        self.mask_procer = load_mask_procer(self.mask_proc_option, **mask_procer_kwargs)
        reg_mask_proc_hooks(
            itr,
            G,
            model,
            self.mask_proc_mxp,
            self.structural,
            self.mask_dim,
            self.mask_procer,
        )
        # print(f"end mask processor registration in {time.time() - start_time}s")
        # start_time = time.time()

        # print("Data size = ", x.shape)
        ## Calc. energy with porlarized model
        E = self.energy_func(model, x, y)
        # print(f"end forward in {time.time() - start_time}s")
        # start_time = time.time()

        neg_dE = torch.autograd.grad(
            -E,
            [m for _, m, _ in masked_params(model)],
            allow_unused=True,
        )
        # print(f"end backward in {time.time() - start_time}s")
        # start_time = time.time()

        E.detach_()

        ## Un-polarize masks for SFPK update
        if self.mask_proc_option in ["ohh", "ohhm"]:
            for i, (_, m, _) in enumerate(masked_params(model)):
                m.data.copy_(masks_prev[i])

        # print(f"end mask reset in {time.time() - start_time}s")

        return E, neg_dE

        # Must be computed outside the polarization environment

    def _calc_K(self, model):
        if self.structural:
            penalty = self.mask_sys.calc_similarity_struct(
                model,
                self.mask_dim,
                self.repl_weighted,
                self.repl_mode,
            )
        else:
            penalty = self.mask_sys.calc_similarity(
                model, self.repl_weighted, self.repl_mode
            )

        return penalty

    @func_timer
    def _calc_neg_dK(self, model):
        K = self._calc_K(model)
        neg_dK = torch.autograd.grad(-K, [m for _, m, _ in masked_params(model)])
        return K, neg_dK

    #
    def _calc_delta(self, r_t, vec_G, vec_E, eps=1e-12):
        if self.local_ode:
            delta = [None] * len(vec_G)
            for layer, (g, e) in enumerate(zip(vec_G, vec_E)):
                g_norm = g.norm()
                e_norm = e.norm()
                gxe = (g * e).sum()

                x = torch.sqrt(
                    (r_t**2 - 1) / ((g_norm * e_norm) ** 2 - gxe**2 + eps) + eps
                )
                y = (1 - x * gxe) / (g_norm**2 + eps)
                delta[layer] = (x * e + y * g).clone()

        else:
            G_norm = calc_norm(vec_G)
            E_norm = calc_norm(vec_E)
            GxE = calc_inner_prod(vec_G, vec_E)

            x = torch.sqrt(
                (r_t**2 - 1 + eps) / ((G_norm * E_norm) ** 2 - GxE**2 + eps) + eps
            )
            y = (1 - x * GxE) / (G_norm**2 + eps)
            delta = [(x * e + y * g) for g, e in zip(vec_G, vec_E)]

        return delta

    # ============================================= Core functions ============================================= #

    @torch.no_grad()
    def _calc_scores(self, itr, i, model):

        ## Apply i-th mask
        self.mask_sys.apply(i, model)

        assert (
            self.mask_proc_score_option == "Id"
        ), "importance scores are based on un-polarized (i.e. Id-polarization) masks"
        mask_procer_kwargs = dict(
            mask_dim=self.mask_dim,
            local=self.local_prune,
            eps=self.mask_proc_eps,
            ratio=self.mask_proc_ratio,
            mxp=self.mask_proc_mxp,
        )
        mask_procer_for_score = load_mask_procer(
            self.mask_proc_score_option, **mask_procer_kwargs
        )

        def _unstructural_score(itr, mask, param, sparsity, score_option, mask_procer):
            if score_option == "m":
                score_tmp = (
                    (mask_procer(itr, mask, param, sparsity, model)).abs().clone()
                )
            elif score_option == "mp":
                score_tmp = (
                    (mask_procer(itr, mask, param, sparsity, model) * param)
                    .abs()
                    .clone()
                )
            elif score_option == "mpz":
                score_tmp = (
                    (
                        mask_procer(itr, mask, param, sparsity, model)
                        * (param.abs() + 1e-4)
                    )
                    .abs()
                    .clone()
                )
            else:
                raise NotImplementedError

            return score_tmp

        def _structural_score(
            itr, mask, param, sparsity, score_option, mask_procer, mask_dim
        ):
            ## Conv2d
            if param.dim() == 4:
                view_shape = (-1, 1, 1, 1) if mask_dim == 0 else (1, -1, 1, 1)
                unmasked_dims = (1, 2, 3) if mask_dim == 0 else (0, 2, 3)
                mask_proced = (
                    mask_procer(itr, mask, param, sparsity, model)
                    .view(view_shape)
                    .expand_as(p)
                )

            ## FC
            elif param.dim() == 2:
                view_shape = (-1, 1) if mask_dim == 0 else (1, -1)
                unmasked_dims = (1) if mask_dim == 0 else (0)
                mask_proced = (
                    mask_procer(itr, mask, param, sparsity, model)
                    .view(view_shape)
                    .expand_as(p)
                )

            ## FC-bias
            else:
                unmasked_dims = None
                mask_proced = mask_procer(itr, mask, param, sparsity, model)

            if score_option == "m":
                score_tmp = (mask_proced).abs().clone()
            elif score_option == "mp":
                score_tmp = (mask_proced * param).abs().clone()
            elif score_option == "mpz":
                score_tmp = (mask_proced * (param.abs() + 1e-4)).abs().clone()
            else:
                raise NotImplementedError

            if unmasked_dims is not None:
                score_tmp = score_tmp.sum(unmasked_dims)

            return score_tmp

        scores = {}
        sparsity, _ = self._calc_G(model)
        for key, m, p in masked_params(model):
            if self.structural:
                scores[key] = _structural_score(
                    itr,
                    m,
                    p,
                    sparsity,
                    self.score_option,
                    mask_procer_for_score,
                    self.mask_dim,
                )
            else:
                scores[key] = _unstructural_score(
                    itr, m, p, sparsity, self.score_option, mask_procer_for_score
                )

        return scores

    def _update_scores(self, itr, idxs_sort, model, target_sparsity=None):
        ## Initialize scores
        if len(self.scores) == 0:
            self.scores = {
                key: torch.zeros_like(m).to(m) for key, m, _ in masked_params(model)
            }

        idxs_select = (
            idxs_sort[: self.evol_score_top_k] if self.evol_score_top_k else idxs_sort
        )
        print(f"\nUpdaing score, idxs selected = {idxs_select}\n")
        scores_list = [self._calc_scores(itr, i, model) for i in idxs_select]

        if self.vote_mode == "soft":
            for key in self.scores.keys():
                score_new_avg = 0.0
                for scores_new in scores_list:
                    ## Update scores with momentum
                    score_new_avg += scores_new[key]

                score_new_avg /= len(scores_list)

                self.scores[key] = score_new_avg + self.momentum * self.scores[key]

        elif self.vote_mode == "hard":
            ## based on one-hot voting
            assert target_sparsity is not None
            for score_new in scores_list:
                for key, _, p in masked_params(model):
                    score_new[key] *= (p != 0).float()

                global_scores = torch.cat([s.flatten() for s in score_new.values()])
                prune_num = int((1 - target_sparsity) * global_scores.numel())
                assert prune_num >= 1
                threshold, _ = global_scores.kthvalue(k=prune_num)

                zero = torch.tensor([0.0]).to(self.device)
                for key, _, _ in masked_params(model):
                    score_new[key] = torch.where(
                        score_new[key].le(threshold), zero, score_new[key]
                    )

            for key in self.scores.keys():
                score_new_avg = 0.0
                for score_new in scores_list:
                    ## Update scores with momentum
                    score_new_avg += score_new[key]
                score_new_avg /= len(scores_list)

                self.scores[key] = score_new_avg + self.momentum * self.scores[key]

        else:
            raise NotImplementedError

    @func_timer
    def _one_step_ode(self, itr, start, end, model, x, y):

        G, neg_dG, G_local = self._calc_neg_dG(model)
        E, neg_dE = self._calc_neg_dE(itr, G, model, x, y)

        if self.repl != 0:
            K, neg_dK = self._calc_neg_dK(model)
            neg_dE_aug = []
            for i in range(len(neg_dE)):
                neg_dE_aug.append(neg_dE[i] + self.repl * neg_dK[i])

            neg_dE = neg_dE_aug

        else:
            with torch.no_grad():
                # K = self._calc_K(model)
                K = torch.Tensor([np.nan]).to(self.device)

        dt = self._calc_dt(
            itr, start, end, G=G, neg_dE=neg_dE, neg_dG=neg_dG, G_local=G_local
        )

        if self.rt_schedule in ["auto", "autor"]:
            ## Adaptive r_t seach: r_t = best_among( r_t * alpha for alpha in self.multipliers )
            E_best = torch.inf
            for alpha in self.multipliers:
                model_tmp = copy.deepcopy(model)

                if self.rt_schedule == "auto":
                    r_tmp = max(self.r_min, alpha * self.r_prev)
                elif self.rt_schedule == "autor":
                    r_tmp = 1 + (self.r_prev - 1) * alpha

                delta_tmp = self._calc_delta(r_tmp, neg_dG, neg_dE)

                ## Adjust dt according to r_tmp
                # scaler = r_tmp / self.r_init
                # dt_tmp = [dt_tmp_i / scaler for dt_tmp_i in dt] if self.local_prune else dt / scaler
                dt_tmp = dt

                # self._update_masks(model_tmp, delta_tmp, dt_tmp)
                for layer, (_, m, _) in enumerate(masked_params(model_tmp)):
                    if self.local_prune:
                        m.data += delta_tmp[layer] * dt_tmp[layer]
                    else:
                        m.data += delta_tmp[layer] * dt_tmp

                with torch.no_grad():
                    G_tmp, _ = self._calc_G(model_tmp)
                    E_tmp, penalty_tmp = self._calc_E(itr, G_tmp, model_tmp, x, y)
                    del model_tmp

                ## Store best model (prefer smaller r_tmp)
                G_descent_ideal = dt_tmp[0] if isinstance(dt_tmp, list) else dt_tmp
                if E_tmp <= E_best:
                    if (G - G_tmp) > 0.9 * G_descent_ideal:
                        E_best = E_tmp
                        dt_best, r_best, delta_best = dt_tmp, r_tmp, delta_tmp

            ## Update search result
            dt, r_t, delta = dt_best, r_best, delta_best
            self.r_prev = r_best

        else:
            ## Direct update
            r_t = self._calc_r_t(itr, neg_dE)
            delta = self._calc_delta(r_t, neg_dG, neg_dE)

        return dt, r_t, G, E, K, delta, neg_dE, neg_dG

    @func_timer
    def one_step_evol(self, itr, start, end, model, dataloader, eps=1e-12):

        dt_stats = []
        r_t_stats = []
        E_stats = []
        pen_stats = []
        G_stats = []
        delta_stats = []
        neg_dE_stats = []
        neg_dG_stats = []

        ## mask system update via one-step SpODE

        # calc all SpODE delta
        G_avg = 0.0
        pbar = tqdm(range(len(self.mask_sys)))
        for i in pbar:
            x, y = next(dataloader)
            x, y = x.to(self.device), y.to(self.device)
            # print("load data time",time.time()-tttt)
            self.mask_sys.apply(i, model)
            dt, r_t, G, E, penalty, delta, neg_dE, neg_dG = self._one_step_ode(
                itr, start, end, model, x, y
            )
            self.mask_sys.update_to_cache(i, delta, dt)
            self.mask_sys.energy_list[i] = E.item()  ##TODO: debug, comparison
            G_avg += G.item() / self.n_mc  ##TODO: debug, comparison

            fmt_stat_str = self.get_fmt_stat(
                i, itr, dt, r_t, G, E, penalty, delta, neg_dE, neg_dG
            )
            pbar.set_postfix_str(fmt_stat_str)

            dt_stats.append(dt)
            r_t_stats.append(r_t)
            E_stats.append(E)
            pen_stats.append(penalty)
            G_stats.append(G)
            delta_stats.append(delta)
            neg_dE_stats.append(neg_dE)
            neg_dG_stats.append(neg_dG)

        # update all mask particles
        self.mask_sys.update()

        # print masks with k-lowest energy
        idxs_sort = np.argsort(self.mask_sys.energy_list, kind="stable")

        return (
            idxs_sort,
            dt_stats,
            r_t_stats,
            G_stats,
            E_stats,
            pen_stats,
            delta_stats,
            neg_dE_stats,
            neg_dG_stats,
        )

    @func_timer
    def simulation(self, end, model, dataloader, device, mask_sys_ckpt=None):
        self.model_size = sum([p.numel() for p in model.parameters()])
        self.local_prune = isinstance(end, list)
        self.device = device

        if mask_sys_ckpt is not None:
            self.mask_sys = MaskSystem(self.n_mc, model)
            for i in range(len(self.mask_sys)):
                self.mask_sys.mask_list[i] = mask_sys_ckpt.mask_list[i]

            if self.local_prune:
                G = []
                for i in range(len(self.mask_sys)):
                    self.mask_sys.apply(i, model)
                    _, G_local_tmp = self._calc_G(model)
                    G.append([g.item() for g in G_local_tmp])

                G = np.mean(np.array(G), axis=0)
                start = list(G)

            else:
                G = []
                for i in range(len(self.mask_sys)):
                    self.mask_sys.apply(i, model)
                    G_tmp, _ = self._calc_G(model)
                    G.append(G_tmp.item())
                print("Resuming mask system sparsity = ", G)
                start = np.mean(np.array(G))

        else:
            self.mask_sys = MaskSystem(self.n_mc, model)

            G_start, G_start_local = self._calc_G(model)

            if self.local_prune:
                start = [g.item() for g in G_start_local]
            else:
                start = G_start.item()

        print(f"ODE sparsity jounrney = {start} --> {end}")

        for _, m, _ in masked_params(model):
            m.requires_grad = True

        stat_names = [
            "dt",
            "r_t",
            "sparsity",
            "energy",
            "penalty",
            "cos(delta, - dG)",
            "cos(delta, - dE)",
            "cos(dG, dE)",
            "norm(dG)",
            "norm(dE)",
        ]

        mask_stat_norm = self.mask_sys.calc_all_norm()  # layer-wise norm
        mask_stat_dist = self.mask_sys.calc_pairwise_distance()

        stat_names += list(mask_stat_norm.keys()) + list(mask_stat_dist.keys())

        self.stat_trackers = [
            StatTracker(stat_names, save_dir=self.save_dir, postfix=i + 1)
            for i in range(len(self.mask_sys))
        ]
        data_iter = inf_iter(dataloader)

        for itr in range(self.N):
            (
                idxs_sort,
                dt_stats,
                r_t_stats,
                G_stats,
                E_stats,
                pen_stats,
                delta_stats,
                neg_dE_stats,
                neg_dG_stats,
            ) = self.one_step_evol(itr, start, end, model, data_iter)

            num_fwd_hook = 0
            for k, module in model.named_modules():
                if hasattr(module, "fwd_hook"):
                    num_fwd_hook += 1
            print(f"\n# fwd hook = {num_fwd_hook}\n")

            ##TODO: active revision
            self._update_scores(itr, idxs_sort, model, target_sparsity=end)

            mask_stat_norm = self.mask_sys.calc_all_norm()
            mask_stat_dist = self.mask_sys.calc_pairwise_distance()

            for i in range(len(self.mask_sys)):
                self._update_stat_tracker(
                    i,
                    itr,
                    dt_stats[i],
                    r_t_stats[i],
                    G_stats[i],
                    E_stats[i],
                    pen_stats[i],
                    delta_stats[i],
                    neg_dE_stats[i],
                    neg_dG_stats[i],
                    mask_stat_norm,
                    mask_stat_dist,
                    verbose=False,
                )
                self.stat_trackers[i].save_stats()

            df_values = np.concatenate(
                [
                    stat_tracker.df.values[None, ...]
                    for stat_tracker in self.stat_trackers
                ],
                axis=0,
            )
            df_mean = np.mean(df_values, axis=0)
            df_std = np.std(df_values, axis=0)
            df_summary = []
            n_row, n_col = df_mean.shape
            for i in range(n_row):
                row = []
                for j in range(n_col):
                    row.append(f"{df_mean[i, j]: .5f} ({df_std[i, j]: .5f})")
                df_summary.append(row)

            df_summary = pd.DataFrame(df_summary, columns=stat_names)
            df_summary.to_csv(f"{self.save_dir}/stats_summary.csv")

            if self.save_ckpt:
                G = sum(G_stats) / len(G_stats)
                # first time reach a sparsity milestone
                if len(self.G_mile_stones) >= 1 and G <= self.G_mile_stones[-1]:
                    G_ideal = self.G_mile_stones.pop()
                    self._save_ckpt(itr, model, G_stats, G_ideal, E_stats)

            # early breaking
            G_avg = np.mean([g.item() for g in G_stats])
            if G_avg <= np.mean(end):
                print(
                    f"\nEARLY STOPPING: itr = {itr}, current sparsity = {G_avg: .2%} <= target sparsity {end: .2%}. \n"
                )
                break

        if self.save_ckpt and len(self.G_mile_stones) >= 1:
            G_ideal = self.G_mile_stones.pop()
            self._save_ckpt(itr, model, G_stats, G_ideal, E_stats)

        for _, m, _ in masked_params(model):
            m.requires_grad_(False)

        reg_fwd_hooks(model, self.structural, self.mask_dim)
        print("\n Reset polarizor hook --> mask mulitplicative hook successfully ...\n")
