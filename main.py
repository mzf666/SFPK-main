import argparse
import json
import os
import socket
import sys
import time
import warnings

import numpy as np
import torch
import torchvision

from Experiments import (
    oneshot,
    iterative,
    pretrain,
)
from Utils import misc


def build_args():
    HOST_NAME = socket.gethostname()

    DATA_DICT = {
        "<host_name>": "<data_dir>",
    }

    MODEL_DICT = {
        "<host_name>": "<model_dir>",
    }
    OUT_DICT = {
        "<host_name>": "<root>/SFPK-main/Results",
    }
    WORK_DICT = {
        "<host_name>": "<root>/SFPK-main",
    }

    parser = argparse.ArgumentParser(description="Pruning with CV data.")
    parser.add_argument(
        "--exp",
        type=str,
        default="oneshot",
        help="oneshot | oneshot_rebuttal | oneshot_struct | iter | pretrain | tune | track | tune_track | sparse_train",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--resume_time_stamp", type=str, default=None)
    parser.add_argument("--force_re_prune", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=124)
    parser.add_argument("--cuda_idx", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default=DATA_DICT[HOST_NAME])
    parser.add_argument(
        "--data",
        type=str,
        default="imagenet",
        help="cifar10 | cifar100 | tiny_imagenet | imagenet",
    )
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--model_dir", type=str, default=MODEL_DICT[HOST_NAME])
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--use_init", action="store_true", default=False)
    parser.add_argument("--use_baseline", action="store_true", default=False)

    parser.add_argument("--opt", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_scheduler", type=str, default="cos")
    parser.add_argument("--lr_min", type=float, default=None)

    parser.add_argument("--lr_drop_rate", type=float, default=0.1)  # for pretrain
    parser.add_argument(
        "--tune_per_prn", type=str, default="3,4,5,10,15,15,20,20,30"
    )  # for iterative
    parser.add_argument(
        "--itr_lr", type=str, default="3,4,5,10,15,15,20,20,30"
    )  # for iterative
    parser.add_argument(
        "--itr_lr_end", type=str, default="5e-4,2e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4"
    )  # for iterative
    parser.add_argument(
        "--lr_milestones", type=str, default="50,80,120"
    )  # for pretrain

    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.875)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--n_grad_accum", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--pre_epochs", type=int, default=2)  # for pretrain
    parser.add_argument("--prn_epochs", type=int, default=10)  # for oneshot / iterative
    parser.add_argument("--ft_epochs", type=int, default=100)  # for oneshot / iterative
    parser.add_argument(
        "--fixed_lr_epochs", type=int, default=0
    )  # for oneshot / iterative
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--cold_start_lr", type=float, default=1e-4)

    parser.add_argument(
        "--pruner",
        type=str,
        default="SFPK",
        help="Rand | Mag | SNIP | SynFlow | GraSP | SpODE | SFPK",
    )
    parser.add_argument("--structural", action="store_true", default=False)
    parser.add_argument(
        "--mask_dim",
        type=int,
        default=1,
        help="for structural pruning: 0 = out_chn | 1 = in_chn",
    )
    parser.add_argument(
        "--prn_scope", type=str, default="global", help="local | global"
    )
    parser.add_argument("--prn_schedule", type=str, default="exponential")
    parser.add_argument(
        "--sparsity", type=float, default=0.7
    )  # target sparsity = remain / total

    # keep modules unpruned / unmasked
    parser.add_argument("--free_bn", action="store_true", default=True)
    parser.add_argument("--free_Id", action="store_true", default=True)
    parser.add_argument("--free_bias", action="store_true", default=True)
    parser.add_argument("--free_conv1", action="store_true", default=False)
    parser.add_argument("--free_lastfc", action="store_true", default=False)

    # ========================== SpODE & SFPK hyper-parameters ======================== #

    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--r", type=float, default=1.4)
    parser.add_argument(
        "--ode_scope", type=str, default="global", help="global | local"
    )
    parser.add_argument(
        "--energy_option", type=str, default="CE", help="CE | P1 | CEP1"
    )
    parser.add_argument(
        "--sparsity_option", type=str, default="l2", help="l1 | l2 | l1p"
    )
    parser.add_argument("--score_option", type=str, default="mp")
    parser.add_argument("--mask_option", type=str, default="one")
    parser.add_argument(
        "--schedule", type=str, default="exp", help="lin | exp | invexp | hess"
    )
    parser.add_argument(
        "--rt_schedule", type=str, default="fix", help="fix | invexp | hess | auto"
    )
    parser.add_argument(
        "--mom",
        type=float,
        default=0.1,
        help="momentum coeff for SpODE / SFPK score update",
    )

    parser.add_argument(
        "--mask_proc_option",
        type=str,
        default="ohh",
        help="Id | qt | oh | ohh | gau",
    )
    parser.add_argument("--mask_proc_eps", type=float, default=0.9)
    parser.add_argument("--mask_proc_ratio", type=float, default=0.9)
    parser.add_argument("--mask_proc_score_option", type=str, default="Id")
    parser.add_argument("--mask_proc_mxp", action="store_true", default=True)

    # ========================== SFPK hyper-parameters ======================== #

    parser.add_argument("--sfpk_n_mask", type=int, default=10)
    parser.add_argument("--sfpk_repl_mode", type=str, default="exp", help="exp | cos")
    parser.add_argument("--sfpk_repl_lam", type=float, default=0.1)
    parser.add_argument(
        "--sfpk_repl_weighted", action="store_true", default=True
    )  # use mxp (mask * param) or m (mask) when calculating repulsive convex penalty
    parser.add_argument(
        "--sfpk_vote_ratio", type=float, default=0.0, help="activates iff > 0"
    )  # ratio of voting masking / pruning score wthin the n mask system
    parser.add_argument(
        "--sfpk_vote_mode", type=str, default="hard", help="hard | soft"
    )  # mode of voting masking / pruning score wthin the n mask system

    # ========================== Saving Configs ======================== #

    parser.add_argument("--out_dir", type=str, default=OUT_DICT[HOST_NAME])
    parser.add_argument("--ckpt_freq", type=int, default=300)
    parser.add_argument("--save_ckpt", action="store_true", default=False)
    parser.add_argument("--save_dist", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)

    args = parser.parse_args()

    args.device = f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu"
    args.pretrained = not args.use_init
    args.lr_milestones = [int(i) for i in args.lr_milestones.split(",")]

    # ================================ Pruner Config ================================= #

    import yaml
    from easydict import EasyDict as edict

    if args.pruner in ["SpODE", "SFPK"]:

        config_path = f"{WORK_DICT[HOST_NAME]}/Methods/Configs/{args.pruner}.yaml"
        args.prn_kwargs = edict(
            yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        )
        args.prn_kwargs["N"] = args.N
        args.prn_kwargs["r"] = args.r
        args.prn_kwargs["ode_scope"] = args.ode_scope
        args.prn_kwargs["E"] = args.energy_option
        args.prn_kwargs["G"] = args.sparsity_option
        args.prn_kwargs["score_option"] = args.score_option
        args.prn_kwargs["schedule"] = args.schedule
        args.prn_kwargs["rt_schedule"] = args.rt_schedule
        args.prn_kwargs["momentum"] = args.mom
        args.prn_kwargs["save_ckpt"] = args.save_ckpt
        args.prn_kwargs["mask_option"] = args.mask_option
        args.prn_kwargs["mask_proc_kwargs"] = {
            "mask_proc_option": args.mask_proc_option,
            "mask_proc_eps": args.mask_proc_eps,
            "mask_proc_ratio": args.mask_proc_ratio,
            "mask_proc_score_option": args.mask_proc_score_option,
            "mask_proc_mxp": args.mask_proc_mxp,
        }

        if args.pruner in ["SFPK"]:
            args.prn_kwargs["n_mc"] = args.sfpk_n_mask
            args.prn_kwargs["repl_mode"] = args.sfpk_repl_mode
            args.prn_kwargs["repl"] = args.sfpk_repl_lam
            args.prn_kwargs["repl_weighted"] = args.sfpk_repl_weighted
            args.prn_kwargs["vote_ratio"] = args.sfpk_vote_ratio
            args.prn_kwargs["vote_mode"] = args.sfpk_vote_mode

        # ===================== Add descriptions =================== #

        for key in args.prn_kwargs.keys():
            if key in ["N", "r", "score_option"]:
                args.description += f"_{key[:3]}{args.prn_kwargs[key]}"
            if key in ["G", "mask_option", "schedule", "rt_schedule"]:
                args.description += f"_{args.prn_kwargs[key]}"

        if args.pruner in ["SpODE", "SFPK"]:
            args.description += f"_{args.mask_proc_option}"
            if args.mask_proc_option in ["qt", "gau", "qtm", "gaum"]:
                args.description += f"{args.mask_proc_eps}{args.mask_proc_ratio}"

        if args.pruner in ["SFPK"]:
            args.description += f"_mom{args.mom}_nmask{args.sfpk_n_mask}_repl_{args.sfpk_repl_mode}{args.sfpk_repl_lam}"
            if args.sfpk_vote_ratio:
                args.description += f"_esrt{args.sfpk_vote_ratio}_{args.sfpk_vote_mode}"
            if args.sfpk_repl_weighted:
                args.description = args.description.replace("_repl", "_replwp")

    else:
        args.prn_kwargs = {}

    if args.structural:
        args.prn_scope = "local"

    if args.free_conv1:
        args.description += "_conv1"
    if args.free_lastfc:
        args.description += "_lastfc"
    if args.structural:
        args.description += f"_struct{args.mask_dim}"
    if args.use_init:
        args.description += "_init"

    if args.exp == "pretrain":
        args.save_dir = f"{args.model_dir}/pretrained/{args.data}"
    else:
        # oneshot / iterative
        if args.use_init:
            args.save_dir = f"{args.out_dir}/{args.exp}_init_{args.data}/{args.model}_sp{args.sparsity}/{args.pruner}{args.description}_{args.prn_scope}/seed{args.seed}"
        else:
            args.save_dir = f"{args.out_dir}/{args.exp}_{args.data}/{args.model}_sp{args.sparsity}/{args.pruner}{args.description}_{args.prn_scope}/seed{args.seed}"

    os.makedirs(args.save_dir, exist_ok=True)

    if args.pruner in ["SpODE", "SFPK"]:
        args.prn_kwargs["save_dir"] = args.save_dir

    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    args.time_stamp = time_stamp
    sys.stdout = misc.Tee(f"{args.save_dir}/{time_stamp}_log.txt")
    sys.stderr = misc.Tee(f"{args.save_dir}/{time_stamp}_err.txt")

    print("Environment:")
    print("\tTime: {}".format(time_stamp))
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))

    print("Args:")
    for k, v in sorted(vars(args).items()):
        print("\t{}: {}".format(k, v))

    with open(args.save_dir + "/args.json", "w") as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=4)

    args.target_sparsity = args.sparsity

    if args.structural and args.exp in ["oneshot", "iterative"]:
        # estimated channel sparsity for structural (layer-wise) pruning
        args.sparsity = args.sparsity**0.5

    return args


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    args = build_args()
    misc.randomize(args.seed)

    if args.use_wandb and os.path.exists("wandb_api.py"):
        from wandb_api import wandb_key

        args.wandb_key = wandb_key
    else:
        args.wandb_key = None
        Warning("\nWandb is not activated!\n")

    if args.exp == "iterative":
        iterative.run(args)

    elif args.exp == "oneshot":
        oneshot.run(args)

    elif args.exp == "pretrain":
        pretrain.run(args)

    else:
        raise NotImplementedError
