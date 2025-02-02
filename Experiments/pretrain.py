import os
import numpy as np
import pandas as pd
import torch

from Utils import load, routine


def train_from_scratch_configs(args):
    exp_type = f"{args.data}_{args.model}"

    ## CIFAR-10 / 100 ##
    if exp_type in ["cifar10_vgg11", "cifar100_vgg11"]:
        args.opt = "momentum"
        args.pre_epochs = 200
        args.bsz = 64
        args.lr = 0.05
        args.lr_milestones = "60,120,160"
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    elif exp_type in [
        "cifar10_vgg16",
        "cifar100_vgg16",
        "cifar10_vgg16_bn",
        "cifar100_vgg16_bn",
    ]:
        args.opt = "momentum"
        args.pre_epochs = 200
        args.bsz = 64
        args.lr = 0.05
        args.lr_milestones = "60,120,160"
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    elif exp_type in [
        "cifar10_resnet20",
        "cifar100_resnet20",
        "cifar10_resnet56",
        "cifar100_resnet56",
    ]:
        args.opt = "momentum"
        args.pre_epochs = 200
        args.bsz = 64
        args.lr = 0.05
        args.lr_milestones = "60,120,160"
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    elif exp_type in ["cifar10_wrn20", "cifar100_wrn20"]:
        args.opt = "momentum"
        args.pre_epochs = 200
        args.bsz = 64
        args.lr = 0.05
        args.lr_milestones = "60,120,160"
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    elif exp_type in [
        "cifar10_resnet32",
        "cifar100_resnet32",
    ]:
        args.opt = "sgd"
        args.pre_epochs = 300
        args.bsz = 128
        args.lr = 0.1
        args.weight_decay = 5e-4
        args.momentum = 0.9
        args.lr_list = [0.1, 0.02, 0.004, 0.0008, 0.00016, 0.000032]
        args.lr_epochs = [60, 60, 40, 40, 50, 50]
        args.lr_milestones = np.cumsum(args.lr_epochs).tolist()
        args.grad_clip = 3.0

        print("lr_list:", args.lr_list)
        print("lr_milestones:", args.lr_milestones)

    elif exp_type in ["cifar10_wrn32", "cifar100_wrn32"]:
        args.opt = "momentum"
        args.pre_epochs = 300
        args.bsz = 128
        args.lr = 0.1
        args.weight_decay = 5e-4
        args.momentum = 0.9
        args.grad_clip = 3.0

    elif exp_type in ["cifar100_wrn32x4", "cifar100_resnet18"]:
        args.opt = "momentum"
        args.pre_epochs = 300
        args.bsz = 128
        args.lr = 0.1
        args.weight_decay = 5e-4
        args.momentum = 0.9
        warmup_list = [
            0.0001,
            0.0113,
            0.0226,
            0.0338,
            0.0451,
            0.0563,
            0.0675,
            0.0788,
            0.09,
        ]
        warmup_epochs = [1] * len(warmup_list)
        args.lr_list = warmup_list + [0.1, 0.02, 0.004, 0.0008, 0.00016, 0.000032]
        args.lr_epochs = warmup_epochs + [60 - len(warmup_list), 60, 40, 40, 50, 50]
        args.lr_milestones = np.cumsum(args.lr_epochs).tolist()

        print("lr_list:", args.lr_list)
        print("lr_milestones:", args.lr_milestones)

    elif exp_type in [
        "cifar10_wrn28x2",
    ]:
        args.opt = "momentum"
        args.pre_epochs = 300
        args.bsz = 128
        args.lr = 0.1
        args.weight_decay = 5e-4
        args.momentum = 0.9
        warmup_list = [
            0.0001,
            0.0113,
            0.0226,
            0.0338,
            0.0451,
            0.0563,
            0.0675,
            0.0788,
            0.09,
        ]
        warmup_epochs = [1] * len(warmup_list)
        args.lr_list = warmup_list + [0.1, 0.02, 0.004, 0.0008, 0.00016, 0.000032]
        args.lr_epochs = warmup_epochs + [60 - len(warmup_list), 60, 40, 40, 50, 50]
        args.lr_milestones = np.cumsum(args.lr_epochs).tolist()

        print("lr_list:", args.lr_list)
        print("lr_milestones:", args.lr_milestones)

    elif exp_type in [
        "cifar10_vgg19_bn",
        "cifar100_vgg19_bn",
    ]:
        args.opt = "sgd"
        args.pre_epochs = 300
        args.bsz = 64
        args.lr = 0.01
        args.weight_decay = 5e-4
        args.momentum = 0.9
        args.grad_clip = 5.0

    ## Tiny-ImagNet ##

    elif exp_type in ["tiny_imagenet_vgg19_bn"]:
        args.opt = "momentum"
        args.pre_epochs = 100
        args.bsz = 64
        args.lr = 0.01
        args.lr_milestones = "60,120,160"
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    elif exp_type in ["tiny_imagenet_resnet50"]:
        args.opt = "momentum"
        args.pre_epochs = 100
        args.bsz = 64
        args.lr = 0.01
        args.lr_milestones = "60,120,160"
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    elif exp_type in ["tiny_imagenet_wrn34"]:
        args.opt = "momentum"
        args.pre_epochs = 100
        args.bsz = 64
        args.lr = 0.01
        args.lr_milestones = "60,120,160"
        args.lr_drop_rate = 0.2
        args.weight_decay = 5e-4

    else:
        raise NotImplementedError

    print(f"Updated to {exp_type} configuration.")

    return args


def run(args):
    args = train_from_scratch_configs(args)
    for k, v in vars(args).items():
        print(f"{k} = {v}")

    if args.wandb_api is not None:
        import wandb

        wandb.login(key=args.wandb_api, relogin=True)
        project_name = f"pretrain_{args.data}_{args.model}"
        entity_name = (
            f"{args.data}_{args.model}_ep{args.pre_epochs}_bsz{args.bsz}_lr{args.lr}"
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

    save_dir = f"{args.model_dir}/pretrained/{args.data}"
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = f"{save_dir}/{args.model}.pt"

    train_loader, test_loader = load.load_data(
        args.data_dir, args.data, args.bsz, args.num_workers
    )

    model = load.load_model(
        args.model_dir, args.model, args.data, args.device, args.pretrained
    )
    criterion = torch.nn.CrossEntropyLoss()

    opt_class, opt_kwargs = load.load_optimizer(args.opt)
    opt_kwargs["weight_decay"] = args.weight_decay
    opt_kwargs["momentum"] = args.momentum
    optimizer = opt_class(model.parameters(), lr=args.lr, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pre_epochs, eta_min=1e-6
    )

    if hasattr(args, "lr_list") and len(args.lr_list) > 0:
        print("Using lr_list for lr scheduling.")
        print("lr_list:", args.lr_list)
        print("lr_milestones:", args.lr_milestones)

    results = []
    columns = [
        "sparsity",
        "lr",
        "train_loss",
        "train_acc1",
        "train_acc5",
        "val_loss",
        "val_acc1",
        "val_acc5",
    ]

    sparsity_tmp = 1.0
    val_loss, val_acc1, val_acc5 = routine.evaluate(
        model, criterion, test_loader, args.device
    )
    row = [
        sparsity_tmp,
        optimizer.param_groups[0]["lr"],
        np.nan,
        np.nan,
        np.nan,
        val_loss,
        val_acc1,
        val_acc5,
    ]
    results.append(row)
    best_acc = val_acc1

    print(
        f"\nPretrain model on {args.data} with {args.model} for {args.pre_epochs} epochs.\n"
    )

    for epoch in range(args.pre_epochs):

        if hasattr(args, "lr_list") and len(args.lr_list) > 0:
            for i in range(len(args.lr_milestones)):
                if epoch <= args.lr_milestones[i]:
                    break

            optimizer.param_groups[0]["lr"] = args.lr_list[i]
            print(
                f"Set lr to {args.lr_list[i]} at epoch {epoch}. lr milestone = {args.lr_milestones}, lr list = {args.lr_list}"
            )

        train_loss, train_acc1, train_acc5 = routine.train(
            model,
            criterion,
            optimizer,
            train_loader,
            args.device,
            epoch,
            grad_clip=args.grad_clip,
            n_grad_accum=args.n_grad_accum,
        )

        if scheduler is not None:
            scheduler.step()

        val_loss, val_acc1, val_acc5 = routine.evaluate(
            model, criterion, test_loader, args.device
        )

        row = [
            sparsity_tmp,
            optimizer.param_groups[0]["lr"],
            train_loss,
            train_acc1,
            train_acc5,
            val_loss,
            val_acc1,
            val_acc5,
        ]
        results.append(row)
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(f"{save_dir}/{args.model}.csv")

        wandb.log(
            {
                "val_acc1": val_acc1,
                "val_acc5": val_acc5,
                "val_loss": val_loss,
                "sparsity": sparsity_tmp,
                "train_loss": train_loss,
                "train_acc1": train_acc1,
                "train_acc5": train_acc5,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        )

        if val_acc1 >= best_acc:
            best_acc = val_acc1

            if os.path.exists(ckpt_path):
                existing_ckpt = torch.load(ckpt_path)
                existing_acc = existing_ckpt["acc"]
                print(
                    f"Current Acc = {best_acc: .2f}%.\n"
                    f"Existeing Acc = {existing_acc: .2f}%."
                )
                if best_acc < existing_acc:
                    print("Model skipped.\n")
                    continue

            ckpt = {
                "epoch": epoch,
                "acc": best_acc,
                "args": args,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(ckpt, ckpt_path)
            print(
                f"Ckeckpoint at epoch [{epoch + 1}/{args.pre_epochs}] saved.\n"
                f"Acc = {best_acc: .2f}%.\n"
            )
