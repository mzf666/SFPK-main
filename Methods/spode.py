import os
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

sns.set_style("darkgrid")
import matplotlib
import copy

matplotlib.use("Agg")

from Utils.misc import func_timer
from Methods.utils import masked_params, reg_fwd_hooks
from Methods.mask_processor import (
    load_mask_procer,
    reg_mask_proc_hooks,
    _calc_global_mask_vec,
)


def calc_norm(x):
    x_norm = torch.sqrt(sum([xi.norm() ** 2 for xi in x]) + 1e-12)
    return x_norm


def calc_inner_prod(x, y):
    x_dot_y = sum([(xi * yi).sum() for xi, yi in zip(x, y)])
    return x_dot_y


def calc_cosine(x, y):
    # assert isinstance(x, list) and isinstance(y, list)
    x_norm = calc_norm(x)
    y_norm = calc_norm(y)
    x_dot_y = calc_inner_prod(x, y)
    # assert x_dot_y.abs() <= (
    #         x_norm * y_norm + 1e-12), f'Warning: <x, y> = {x_dot_y.abs()} > x_norm ({x_norm}) * y_norm ({y_norm})'

    return x_dot_y / (x_norm * y_norm + 1e-12)


class StatTracker:
    def __init__(self, stat_names, save_dir=None, postfix=None):
        self.save_dir = save_dir
        self.stat_names = stat_names
        self.postfix = postfix
        self.stats = {stat_name: [] for stat_name in stat_names}
        self.save_tensor = []
        print("\n\tODE statistic tracker initialized successfully ...")
        print(f"\tstats-to-track = {stat_names}.\n")

    def update_stats(self, stat_vals):
        for key in self.stats.keys():
            self.stats[key].append(stat_vals[key])

    def save_stats(self):
        self.df = pd.DataFrame(data=self.stats)
        if self.save_dir:
            if self.postfix:
                save_path = f"{self.save_dir}/stats_{self.postfix}.csv"
            else:
                save_path = f"{self.save_dir}/stats.csv"

            self.df.to_csv(save_path)
        else:
            print("Warning: save_dir does not exists. Skip saving ...")

    def show_stats(self, itr, N, stat_vals, ncols=3):
        print(f"\nitr = [{itr + 1}/{N}]")
        msg = []
        for i, key in enumerate(self.stats.keys()):
            if stat_vals[key] is not None:
                msg.append("{0:<18} = {1:.5f}".format(key, stat_vals[key]))
            if (i + 1) % ncols == 0 or (i + 1) == len(self.stats.keys()):
                print("\t" + ", ".join(msg))
                msg = []

    @torch.no_grad()
    def show_dist(self, tensor, sparsity, milestones, save_dist):
        assert isinstance(tensor, torch.Tensor)
        print("\nShow distribution:")

        print(f"\t Overall [mu = {tensor.mean(): .2f}, std = {tensor.std(): .9f}]")

        indicator = tensor.le(milestones[0])
        ratio = indicator.sum() / tensor.numel()
        mu = tensor[indicator].mean()
        std = tensor[indicator].std()
        print(f"\t {ratio: .2%} < {milestones[0]} [mu = {mu: .2f}, std = {std: .9f}]")

        for i in range(len(milestones) - 1):
            leps, reps = milestones[i], milestones[i + 1]
            indicator = tensor.ge(leps) * tensor.le(reps)
            ratio = indicator.sum() / tensor.numel()
            mu = tensor[indicator].mean()
            std = tensor[indicator].std()
            print(
                f"\t {ratio: .2%} in [{leps}, {reps}) [mu = {mu: .2f}, std = {std: .9f}]"
            )

        indicator = tensor.ge(milestones[-1])
        ratio = indicator.sum() / tensor.numel()
        mu = tensor[indicator].mean()
        std = tensor[indicator].std()
        print(
            f"\t {ratio: .2%} >= [{milestones[-1]}) [mu = {mu: .2f}, std = {std: .9f}]"
        )

        if save_dist:
            self.save_tensor.append({"phi": tensor, "sparsity": sparsity})
            path = f"{self.save_dir}/track_phi.pt"
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(self.save_tensor, path)
            print(
                f"\t Phi (sparsity = {sparsity: .2%}) saved to {path} successfully ..."
            )


class SparsityIndexedODE:

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
        save_dir=None,
        save_ckpt=False,
    ):
        self.N = int(N)  # num. of bins
        assert r > 1, f"Error: r_init = {r} <= 1!"
        self.r_init = r
        self.r_prev = self.r_init  # for auto / autor search
        self.r_min = 1.01
        self.r_max = 4

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
        self.init_funcs()

        self.structural = mask_dim in [0, 1]
        self.mask_dim = mask_dim
        if self.structural:
            print("\tUsing structural ODE ...")
        else:
            print("\tUsing unstructural ODE ...")

        self.save_dir = save_dir
        self.save_ckpt = save_ckpt
        if save_dir:
            print("\tWarning: tracking ODE stats ...")
            if save_ckpt:
                self.G_mile_stones = list(
                    reversed([0.5, 0.28, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.02])
                )
                print(
                    f"\n\nWarning: saving intermediate model ckpts sp in {self.G_mile_stones} ...\n\n"
                )

    # ======================================= General helpers =================================================== #

    def _save_ckpt(self, itr, model, G, G_ideal):
        model_tmp = copy.deepcopy(model)
        self._permanent_polarization(model_tmp, G)

        ckpt = dict(
            model=model_tmp.state_dict(),
            scores=self.scores,
            sparsity=G,
            itr=itr + 1,
        )
        torch.save(ckpt, f"{self.save_dir}/sp{G_ideal}.pkl")
        print(
            f"\tSpODE checkpoint of sparsity = {G_ideal} with reset fwd-hooks saved successfully ...\n"
        )

    def _update_stat_tracker(
        self, itr, dt, r_t, G, E, delta, neg_dE, neg_dG, verbose=True
    ):

        dt = dt[0] if isinstance(dt, list) else dt

        stat_vals = {
            "dt": dt,
            "r_t": r_t,
            "sparsity": G.item(),
            "energy": E.item(),
            "cos(delta, - dE)": calc_cosine(delta, neg_dE).item(),
            "cos(delta, - dG)": calc_cosine(delta, neg_dG).item(),
            "cos(dG, dE)": calc_cosine(neg_dE, neg_dG).item(),
            "norm(dG)": calc_norm(neg_dG).item(),
            "norm(dE)": calc_norm(neg_dE).item(),
        }
        self.stat_tracker.update_stats(stat_vals)
        if verbose:
            self.stat_tracker.show_stats(itr, self.N, stat_vals, ncols=3)

    def init_funcs(self):
        if self.E == "CE":
            self.energy_func = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

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

    # =========================================== Calc. ODE-Stats =============================================== #

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
            # dt = const / norm(dE)
            dE_norm = calc_norm(neg_dE).item()
            dt_exp_max = (
                end * (start / end) ** 1 / (self.N / 5)
            )  # cifar100-resnet20, vgg16_bn
            # dt_exp_max = 1 # cifar100-wrn20/
            if itr == 0:
                # calibration: dt_lin_init / norm(dE_0) * dt_normalizer = dt_exp_init
                # self.dt_normalizer = 1
                dt_lin_init = (start - end) / self.N
                self.dt_normalizer = dE_norm * dt_exp_max / (dt_lin_init + 1e-12)

            if self.local_prune:
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

        ## Local ODE ##
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

        ## Global ODE ##
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
                            G_tmp = (m.norm(p=2) ** 2) * node_size
                        else:
                            G_tmp = m.norm(p=2) ** 2

                        G += G_tmp
                        G_local.append(G_tmp**0.5 / p.numel() ** 0.5)

                    G = torch.sqrt(G + eps) / normalizer
                else:
                    G = torch.sqrt(
                        sum([m.norm(p=2) ** 2 for _, m, _ in masked_params(model)])
                        + eps
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

        E = self.energy_func(model(x), y)

        ## Un-polarize masks for ODE update
        if self.mask_proc_option in ["ohh", "ohhm"]:
            for i, (_, m, _) in enumerate(masked_params(model)):
                m.data.copy_(masks_prev[i])

        return E

    def _calc_neg_dE(self, itr, G, model, x, y):
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

        ## Calc. energy with porlarized model
        E = self.energy_func(model(x), y)
        neg_dE = torch.autograd.grad(
            -E, [m for _, m, _ in masked_params(model)], allow_unused=True
        )
        E.detach_()

        ## Un-polarize masks for ODE update
        if self.mask_proc_option in ["ohh", "ohhm"]:
            for i, (_, m, _) in enumerate(masked_params(model)):
                m.data.copy_(masks_prev[i])

        return E, neg_dE

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
    def _update_scores(self, itr, model):

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
        print(f"Using {self.mask_proc_score_option} score")

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
            else:
                raise NotImplementedError

            if unmasked_dims is not None:
                score_tmp = score_tmp.sum(unmasked_dims)

            return score_tmp

        ## Initialize scores
        if not hasattr(self, "scores"):
            self.scores = {
                key: torch.zeros_like(m).to(m) for key, m, _ in masked_params(model)
            }

        sparsity, _ = self._calc_G(model)
        for key, m, p in masked_params(model):
            if self.structural:
                score_tmp = _structural_score(
                    itr,
                    m,
                    p,
                    sparsity,
                    self.score_option,
                    mask_procer_for_score,
                    self.mask_dim,
                )
            else:
                score_tmp = _unstructural_score(
                    itr, m, p, sparsity, self.score_option, mask_procer_for_score
                )

            ## Update scores with momentum
            self.scores[key] = score_tmp + self.momentum * self.scores[key]

    def _update_masks(self, model, delta, dt):
        for layer, (_, m, _) in enumerate(masked_params(model)):
            if self.local_prune:
                m.data += delta[layer] * dt[layer]
            else:
                m.data += delta[layer] * dt

    @func_timer
    def _one_step_ode(self, itr, start, end, model, x, y):

        G, neg_dG, G_local = self._calc_neg_dG(model)
        E, neg_dE = self._calc_neg_dE(itr, G, model, x, y)
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

                self._update_masks(model_tmp, delta_tmp, dt_tmp)
                with torch.no_grad():
                    G_tmp, _ = self._calc_G(model_tmp)
                    E_tmp = self._calc_E(itr, G_tmp, model_tmp, x, y)

                ## Store best model (prefer smaller r_tmp)
                G_descent_ideal = dt_tmp[0] if isinstance(dt_tmp, list) else dt_tmp
                if E_tmp <= E_best:
                    if (G - G_tmp) > 0.9 * G_descent_ideal:
                        E_best = E_tmp
                        dt_best, r_best, delta_best = dt_tmp, r_tmp, delta_tmp
                        model_best = copy.deepcopy(model_tmp)

            ## Update search result
            dt, r_t, delta = dt_best, r_best, delta_best
            self.r_prev = r_best
            model.load_state_dict(model_best.state_dict())
            del model_best

        else:
            ## Direct update
            r_t = self._calc_r_t(itr, neg_dE)
            delta = self._calc_delta(r_t, neg_dG, neg_dE)
            self._update_masks(model, delta, dt)

        return dt, r_t, G, E, delta, neg_dE, neg_dG

    @func_timer
    def discretization(self, start, end, model, dataloader, device):
        self.model_size = sum([p.numel() for p in model.parameters()])
        self.local_prune = isinstance(start, list)

        for _, m, _ in masked_params(model):
            m.requires_grad = True

        stat_names = [
            "dt",
            "r_t",
            "sparsity",
            "energy",
            "cos(delta, - dG)",
            "cos(delta, - dE)",
            "cos(dG, dE)",
            "norm(dG)",
            "norm(dE)",
        ]
        self.stat_tracker = StatTracker(stat_names, save_dir=self.save_dir)

        for itr in range(self.N):

            _, (x, y) = next(enumerate(dataloader))
            x, y = x.to(device), y.to(device)

            dt, r_t, G, E, delta, neg_dE, neg_dG = self._one_step_ode(
                itr, start, end, model, x, y
            )

            self._update_scores(itr, model)

            self._update_stat_tracker(itr, dt, r_t, G, E, delta, neg_dE, neg_dG)
            self.stat_tracker.save_stats()

            if self.save_ckpt:
                G, _ = self._calc_G(model)
                # first time reach a sparsity milestone
                if len(self.G_mile_stones) >= 1 and G <= self.G_mile_stones[-1]:
                    G_ideal = self.G_mile_stones.pop()
                    self._save_ckpt(itr, model, G, G_ideal)

        if self.save_ckpt and len(self.G_mile_stones) >= 1:
            G_ideal = self.G_mile_stones.pop()
            self._save_ckpt(itr, model, G, G_ideal)

        ## Permanent mask polarization
        self._permanent_polarization(model, G)


class NaiveODE(SparsityIndexedODE):
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
        save_dir=None,
        save_ckpt=False,
        eq_constrain=False,
        normalized=True,
        lam=1e4,
    ):

        super(NaiveODE, self).__init__(
            mask_dim,
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
        self.eq_constrain = eq_constrain
        self.normalized = normalized
        self.lam = lam
        print(f"Warning: using naive ODE, normalized = {normalized} ...")
        if eq_constrain:
            print(f"\t Obj = E + {lam} * 1/2 * (G - t)^2 \n\n")
        else:
            print(f"\t Obj = E + {lam} * G \n\n")

    def _calc_naive_delta(self, r_t, G, t, neg_dG, neg_dE, eps=1e-12):
        if self.eq_constrain:
            # Obj = E + lam * 1/2 * (G - t)^2
            # t = target sparsity, i.e. 'end'
            delta = [de + self.lam * (G - t) * dg for de, dg in zip(neg_dE, neg_dG)]
        else:
            # Obj = E + lam * G
            delta = [de + self.lam * dg for de, dg in zip(neg_dE, neg_dG)]

        if self.normalized:
            delta_norm = torch.sqrt(sum([d.norm() ** 2 for d in delta]) + eps)
            G_norm = torch.sqrt(sum([g.norm() ** 2 for g in neg_dG]) + eps)

            # normalize to r / norm(G)
            normalizer = r_t / (delta_norm * G_norm + eps)
            delta = [d * normalizer for d in delta]

        return delta

    @func_timer
    def _one_step_ode(self, itr, start, end, model, x, y):

        G, neg_dG, G_local = self._calc_neg_dG(model)
        E, neg_dE = self._calc_neg_dE(itr, G, model, x, y)

        if self.rt_schedule in ["auto"]:
            raise NotImplementedError
        else:
            r_t = self._calc_r_t(itr, neg_dE)
            # delta = self._calc_delta(r_t, neg_dG, neg_dE)
            delta = self._calc_naive_delta(r_t, G, end, neg_dG, neg_dE)
            dt = self._calc_dt(
                itr, start, end, G=G, neg_dE=neg_dE, neg_dG=neg_dG, G_local=G_local
            )

            self._update_masks(model, delta, dt)

        return dt, r_t, G, E, delta, neg_dE, neg_dG

    @func_timer
    def discretization(self, start, end, model, dataloader, device, testloader=None):
        self.model_size = sum([p.numel() for p in model.parameters()])
        self.local_prune = isinstance(start, list)

        for _, m, _ in masked_params(model):
            m.requires_grad = True

        stat_names = [
            "dt",
            "r_t",
            "sparsity",
            "energy",
            "cos(delta, - dG)",
            "cos(delta, - dE)",
            "cos(dG, dE)",
            "norm(dG)",
            "norm(dE)",
        ]
        self.stat_tracker = StatTracker(stat_names, save_dir=self.save_dir)

        ## early quit as sparsity met
        itr = 0
        meet_sparsity = False

        while itr < self.N and not meet_sparsity:

            _, (x, y) = next(enumerate(dataloader))
            x, y = x.to(device), y.to(device)

            dt, r_t, G, E, delta, neg_dE, neg_dG = self._one_step_ode(
                itr, start, end, model, x, y
            )

            self._update_scores(itr, model)

            self._update_stat_tracker(itr, dt, r_t, G, E, delta, neg_dE, neg_dG)
            self.stat_tracker.save_stats()

            if self.save_ckpt:
                G, _ = self._calc_G(model)
                # first time reach a sparsity milestone
                if len(self.G_mile_stones) >= 1 or G <= self.G_mile_stones[-1]:
                    G_ideal = self.G_mile_stones.pop()
                    self._save_ckpt(itr, model, G, G_ideal)

            itr = itr + 1
            G, _ = self._calc_G(model)
            meet_sparsity = G <= end

        ## Permanent mask polarization
        self._permanent_polarization(model, G)
