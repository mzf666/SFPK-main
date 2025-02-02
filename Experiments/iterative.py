import numpy as np
import pandas as pd
import torch

from Utils import load, routine, datasets
from Methods.utils import free_modules, init_masks_hooks


def run(args):
    if args.wandb_key is not None:
        import wandb

        wandb.login(key=args.wandb_key, relogin=True)
        project_name = f"iter_{args.data}_{args.model}_sp{args.target_sparsity}"
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
        )
    else:
        wandb = None

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

    # load dataset and loss for evaluation and post-pruning retraining
    train_loader, test_loader = load.load_data(
        args.data_dir, args.data, args.bsz, args.num_workers
    )
    if args.data in ["imagenet"]:
        criterion_train = datasets.LabelSmoothing(args.label_smoothing)
        criterion_eval = torch.nn.CrossEntropyLoss()
        print(f"\nUsing label smoothing with factor {args.label_smoothing}.\n")
    else:
        criterion_train = torch.nn.CrossEntropyLoss()
        criterion_eval = torch.nn.CrossEntropyLoss()
        print("\nUsing CrossEntropyLoss.\n")

    # load pruner and sparsity schedule
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
    sparsity_tmp = 1.0

    # prepare learning rate for iterative pruning
    tune_epoch_schedule = args.tune_per_prn.split(",")
    tune_lr_list_start = str(args.itr_lr).split(",")

    if args.itr_lr_end is None:
        tune_lr_list_end = [float(x) * 0.1 for x in tune_lr_list_start]
    else:
        tune_lr_list_end = str(args.itr_lr_end).split(",")

    # prepare epoch schedule for iterative pruning
    if len(tune_epoch_schedule) == 1:
        tune_epoch_schedule = [int(tune_epoch_schedule[0])] * args.prn_epochs
        tune_lr_list_start = [float(tune_lr_list_start[0])] * args.prn_epochs
        tune_lr_list_end = [float(tune_lr_list_end[0])] * args.prn_epochs

    elif len(tune_epoch_schedule) == args.prn_epochs - 1:
        tune_epoch_schedule = [int(x) for x in tune_epoch_schedule]
        tune_lr_list_start = [float(x) for x in tune_lr_list_start]
        tune_lr_list_end = [float(x) for x in tune_lr_list_end]

    else:
        raise ValueError("Invalid tune_per_prn schedule format.")

    print(
        f"Warning: {tune_epoch_schedule} retraining epoch per each pruning epoch.\n"
        f"Iterative retraining lr = {tune_lr_list_start} --> {tune_lr_list_end}.\n"
        f"Finall retraining lr = {args.lr} --> {args.lr_min}.\n"
        f"Sparsity journey = {sparsity_schedule}.\n"
        f"Iterative pruning on {args.data} with {args.pruner} for {args.prn_epochs} epochs.\n"
    )

    # iterative pruning loop (prune + retrain)
    for epoch in range(args.prn_epochs):

        if args.data in ["imagenet", "tiny_imagenet"]:
            prn_bsz = 256
            prn_bsz = min(prn_bsz, args.bsz)
        else:
            prn_bsz = 512

        print(
            f"Epoch [{epoch + 1}/{args.prn_epochs}] (prn_bsz = {prn_bsz}), "
            f"BEFORE prune = {sparsity_tmp * 100: .2f}%, "
            f"NEXT ideal sparsity = {sparsity_schedule[epoch] * 100: .2f}%."
        )

        # load dataset for pruning
        prune_loader = datasets.dataloader(
            args.data_dir, args.data, prn_bsz, True, args.num_workers
        )

        # one-step pruning
        routine.prune(
            model,
            pruner,
            criterion_eval,
            prune_loader,
            sparsity_schedule[epoch],
            args.device,
            train=True,
        )

        remainings, total, sparsity_tmp = pruner.stats(model)
        print(f"AFTER prune = {sparsity_tmp * 100 : .2f}% [{remainings}/{total}].")

        val_loss, val_acc1, val_acc5 = routine.evaluate(
            model, criterion_eval, test_loader, args.device
        )
        row = [sparsity_tmp, np.nan, np.nan, np.nan, val_loss, val_acc1, val_acc5]
        results.append(row)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f"{args.save_dir}/results.csv")

        # skip short-finetuning if it's the last epoch
        if epoch == args.prn_epochs - 1:
            break

        n_tune_epochs = tune_epoch_schedule[epoch]
        ft_lr_start = tune_lr_list_start[epoch]
        ft_lr_end = tune_lr_list_end[epoch]
        Warning(f"Retrain without lr warmup: {ft_lr_start} --> {ft_lr_end}")

        # load optimizer for retraining phases in iterative pruning
        opt_class, opt_kwargs = load.load_optimizer(args.opt)
        optimizer = opt_class(model.parameters(), lr=ft_lr_start, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, eta_min=ft_lr_end, T_max=n_tune_epochs
        )

        best_acc = 0.0
        if val_acc1 >= best_acc:
            best_acc = val_acc1
            save_path = f"{args.save_dir}/model_pruned_sp{sparsity_tmp: .2f}_best.pt"
            routine.save_dict(
                save_path,
                epoch + 1,
                sparsity_tmp,
                model,
                optimizer,
                None,
                val_acc1,
                val_acc5,
                args,
            )
            print(
                f"\nSaved best pruned model to {save_path}, Acc1 = {val_acc1}, Sparsity = {sparsity_tmp: .2%}.\n"
            )

        # retrain loop after each pruning shot
        for tune_epoch in range(n_tune_epochs):
            train_loss, train_acc1, train_acc5 = routine.train(
                model,
                criterion_train,
                optimizer,
                train_loader,
                args.device,
                tune_epoch,
                grad_clip=args.grad_clip,
                n_grad_accum=args.n_grad_accum,
            )
            scheduler.step()

            val_loss, val_acc1, val_acc5 = routine.evaluate(
                model,
                criterion_eval,
                test_loader,
                args.device,
            )
            _, _, sparsity_tmp = pruner.stats(model)
            print(f"Current sparsity = {sparsity_tmp: .2%}.")

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
                        "sparsity": sparsity_tmp,
                        "val_loss": val_loss,
                        "val_acc1": val_acc1,
                        "val_acc5": val_acc5,
                        "train_loss": train_loss,
                        "train_acc1": train_acc1,
                        "train_acc5": train_acc5,
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

            if val_acc1 >= best_acc:
                best_acc = val_acc1
                save_path = (
                    f"{args.save_dir}/model_pruned_sp{sparsity_tmp: .2f}_best.pt"
                )
                routine.save_dict(
                    save_path,
                    epoch + 1,
                    sparsity_tmp,
                    model,
                    optimizer,
                    None,
                    val_acc1,
                    val_acc5,
                    args,
                )
                print(
                    f"\nBest model saved to {save_path}, Acc1 = {val_acc1}, Sparsity = {sparsity_tmp: .2%}.\n"
                )

    # final retraining after all iterative pruning
    ft_lr_start = args.lr
    ft_lr_end = args.lr_min if args.lr_min is not None else 0.1 * args.lr
    optimizer = opt_class(model.parameters(), lr=ft_lr_start, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=ft_lr_end, T_max=args.ft_epochs - args.fixed_lr_epochs
    )
    Warning(
        f"\nFinal retraining for {args.ft_epochs} epochs."
        f"lr = {ft_lr_start} --> {ft_lr_end}\n"
    )

    best_acc = 0.0

    # final retraining loop
    for epoch in range(args.ft_epochs):
        train_loss, train_acc1, train_acc5 = routine.train(
            model,
            criterion_train,
            optimizer,
            train_loader,
            args.device,
            epoch,
            grad_clip=args.grad_clip,
            n_grad_accum=args.n_grad_accum,
        )
        if epoch + 1 <= args.ft_epochs - args.fixed_lr_epochs:
            scheduler.step()

        val_loss, val_acc1, val_acc5 = routine.evaluate(
            model, criterion_eval, test_loader, args.device
        )
        _, _, sparsity_tmp = pruner.stats(model)
        print(
            f"[{epoch + 1}/{args.ft_epochs}] Current sparsity = {sparsity_tmp * 100 : .2f}%."
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
        wandb.log(
            {
                "sparsity": sparsity_tmp,
                "val_loss": val_loss,
                "val_acc1": val_acc1,
                "val_acc5": val_acc5,
                "train_loss": train_loss,
                "train_acc1": train_acc1,
                "train_acc5": train_acc5,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        )

        if val_acc1 >= best_acc:
            best_acc = val_acc1
            save_path = f"{args.save_dir}/model_pruned_sp{sparsity_tmp: .2f}_best.pt"
            routine.save_dict(
                save_path,
                epoch + 1,
                sparsity_tmp,
                model,
                optimizer,
                None,
                val_acc1,
                val_acc5,
                args,
            )
            print(
                f"\nBest model saved to {save_path}, Acc1 = {val_acc1}, Sparsity = {sparsity_tmp: .2%}.\n"
            )
