import time
import numpy as np
import pandas as pd
import torch

from Utils import load, routine, datasets, misc
from Methods.utils import free_modules, init_masks_hooks


def run(args):

    # prepare wandb logger
    if args.wandb_key is not None:
        import wandb

        wandb.login(key=args.wandb_key, relogin=True)
        project_name = f"oneshot_{args.data}_{args.model}_sp{args.target_sparsity}"
        if args.structural:
            project_name += "_struct"

        entity_name = (
            f"{args.pruner}{args.description}_{args.prn_scope}_seed{args.seed}"
        )
        wandb.init(
            project=project_name,
            name=entity_name,
            config=vars(args),
            dir=args.save_dir,
            root_dir=args.out_dir,
        )
    else:
        wandb = None

    # load dataset and loss for evaluation and post-pruning retraining
    train_loader, test_loader = load.load_data(
        args.data_dir, args.data, args.bsz, args.num_workers
    )

    # load model and apply mask
    model = load.load_model(
        args.model_dir,
        args.model,
        args.data,
        args.device,
        args.pretrained,
    )
    init_masks_hooks(model, args.structural, args.mask_dim)
    free_modules(
        model,
        args.free_bn,
        args.free_Id,
        args.free_bias,
        args.free_conv1,
        args.free_lastfc,
    )

    if args.data in ["imagenet", "tiny_imagenet"]:
        example_inputs = torch.randn(1, 3, 224, 224).to(args.device)
    elif args.data in ["cifar10", "cifar100"]:
        example_inputs = torch.randn(1, 3, 32, 32).to(args.device)
    else:
        raise NotImplementedError

    # load loss for pruning and post-training
    if args.data in ["imagenet"]:
        criterion_train = datasets.LabelSmoothing(args.label_smoothing)
        criterion_eval = torch.nn.CrossEntropyLoss()
        print(f"\nUsing label smoothing, alpha = {args.label_smoothing}.\n")

    else:
        criterion_train = torch.nn.CrossEntropyLoss()
        criterion_eval = torch.nn.CrossEntropyLoss()
        print("\nUsing CrossEntropyLoss.\n")

    # load pruner
    pruning_start_time = time.time()
    print(
        f"\nOne-shot pruning on {args.data} with {args.pruner} for {args.prn_epochs} epochs.\n"
    )
    pruner = load.load_pruner(
        model,
        args.pruner,
        args.structural,
        args.mask_dim,
        **args.prn_kwargs,
    )

    # NOTE: sparsity = # remain / # total across the whole code base (a.k.a density)
    # sparsity = List[float] for layer-wise local pruning
    sparsity = load.load_sparsity(model, args.sparsity, args.prn_scope)

    # args.prn_epochs can be greater than 1 in one-shot pruning as long as no intermediate fine-tuning is involved
    sparsity_schedule = load.load_sparsity_schedule(
        sparsity, args.prn_epochs, args.prn_schedule
    )

    # init stat results to save
    results = []
    columns = [
        "sparsity",
        "train_loss",
        "train_acc1",
        "train_acc5",
        "val_loss",
        "val_acc1",
        "val_acc5",
    ]
    sparsity_tmp = 1

    # pruning loop
    for epoch in range(args.prn_epochs):

        # load dataset for pruning
        prn_bsz = 128 if args.data in ["imagenet", "tiny_imagenet"] else 512
        prune_loader = datasets.dataloader(
            args.data_dir,
            args.data,
            prn_bsz,
            True,
            args.num_workers,
            prune=True,
        )

        print(
            f"\nPruning [{epoch + 1}/{args.prn_epochs}] (bsz = {prn_bsz}), "
            f"sparsity BEFORE prune = {sparsity_tmp: .2%}%, "
            f"NEXT ideal sparsity = {sparsity_schedule[epoch]: .2%}."
        )

        # prune model
        routine.prune(
            model,
            pruner,
            criterion_eval,
            prune_loader,
            sparsity_schedule[epoch],
            args.device,
            train=True,
        )

        # evaluate pruned model
        remainings, total, sparsity_tmp = pruner.stats(model)
        print(f"sparsity AFTER prune = {sparsity_tmp: .2%} [{remainings}/{total}].")

        val_loss, val_acc1, val_acc5 = routine.evaluate(
            model,
            criterion_eval,
            test_loader,
            args.device,
        )

        # save stat results
        row = [
            sparsity_tmp,
            np.nan,
            np.nan,
            np.nan,
            val_loss,
            val_acc1,
            val_acc5,
        ]
        results.append(row)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f"{args.save_dir}/results.csv")

    # save pruned model
    if args.save_ckpt or args.data in ["imagenet"]:
        optimizer, scheduler = None, None
        save_path = f"{args.save_dir}/model_pruned.pt"
        routine.save_dict(
            save_path,
            epoch + 1,
            sparsity_tmp,
            model,
            optimizer,
            scheduler,
            val_acc1,
            val_acc5,
            args,
        )

    prune_time = time.time() - pruning_start_time
    tune_start_time = time.time()

    print("\n", "=" * 80)

    print(f"\n Retrain pruned model for {args.ft_epochs} epochs.\n")
    best_acc = 0.0

    # load retraining optimizer
    opt_class, opt_kwargs = load.load_optimizer(args.opt)

    opt_kwargs["weight_decay"] = args.weight_decay
    if args.opt not in ["adamw", "adam"]:
        opt_kwargs["momentum"] = args.momentum

    print(f"Retraining optimizer: {opt_class.__name__}")
    for k, v in opt_kwargs.items():
        print(f"{k} = {v}")

    print("\n", "=" * 80)

    lr_warmup = args.warmup_epochs > 0
    lr_init = args.cold_start_lr if lr_warmup else args.lr
    optimizer = opt_class(model.parameters(), lr=lr_init, **opt_kwargs)

    if lr_warmup:
        scheduler = routine.WarmupCosineAnnealingLR(
            optimizer,
            T_warmup=args.warmup_epochs,
            T_max=args.ft_epochs - args.fixed_lr_epochs,
            eta_max=args.lr,
            eta_min=1e-8,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.ft_epochs - args.fixed_lr_epochs,
            eta_min=1e-8,
        )

    start_epoch = 0

    if args.data in ["imagenet", "tinyimagenet"]:
        log_interval = 50 * args.n_grad_accum
    else:
        log_interval = 100 * args.n_grad_accum

    # retraining loop
    for epoch in range(start_epoch, args.ft_epochs):
        train_loss, train_acc1, train_acc5 = routine.train(
            model,
            criterion_train,
            optimizer,
            train_loader,
            args.device,
            epoch,
            verbose=True,
            log_interval=log_interval,
            n_grad_accum=args.n_grad_accum,
            grad_clip=args.grad_clip,
        )

        if epoch + 1 <= args.ft_epochs - args.fixed_lr_epochs:
            if scheduler is not None:
                scheduler.step()

        # track retraining stats
        val_loss, val_acc1, val_acc5 = routine.evaluate(
            model, criterion_eval, test_loader, args.device
        )

        _, _, sparsity_tmp = pruner.stats(model)
        print(
            f"[{epoch + 1}/{args.ft_epochs}], sparsity = {sparsity_tmp * 100 : .2f}%, n_grad_accum = {args.n_grad_accum}."
        )

        row = [
            sparsity_tmp,
            train_loss,
            train_acc1,
            train_acc5,
            val_loss,
            val_acc1,
            val_acc5,
        ]
        results.append(row)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f"{args.save_dir}/results.csv")

        if wandb is not None:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc1": train_acc1,
                    "train_acc5": train_acc5,
                    "val_loss": val_loss,
                    "val_acc1": val_acc1,
                    "val_acc5": val_acc5,
                    "sparsity": sparsity_tmp,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )

        if val_acc1 >= best_acc:
            best_acc = val_acc1
            save_path = f"{args.save_dir}/model_tuned_{args.time_stamp}.pt"
            routine.save_dict(
                save_path,
                epoch,
                sparsity_tmp,
                model,
                optimizer,
                scheduler,
                val_acc1,
                val_acc5,
                args,
            )
            print(
                f"\nModel ({val_acc1: .2f}% >= {best_acc: .2f}%) saved to {save_path}.\n"
            )

        if (epoch + 1) % 5 == 0 and (
            args.save_ckpt or args.data in ["imagenet", "tiny_imagenet"]
        ):
            save_path = (
                f"{args.save_dir}/model_tuned_{args.time_stamp}_ep{epoch + 1}.pt"
            )
            routine.save_dict(
                save_path,
                epoch,
                sparsity_tmp,
                model,
                optimizer,
                scheduler,
                val_acc1,
                val_acc5,
                args,
            )

    tune_time = time.time() - tune_start_time

    print(
        f"\nPruning time = {prune_time: .2f} sec, Tune time = {tune_time: .2f} sec.\n"
        f"\nPruning time = {misc.parse_second(prune_time)}.\n"
        f"\nTune time = {misc.parse_second(tune_time)}.\n"
        f"\nOne-shot pruning ends with best acc = {best_acc: .2f}%\n"
    )
