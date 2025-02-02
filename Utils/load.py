import os
import torch
import torch.nn as nn
from Utils import datasets
from Methods import utils


def load_data(root, dataset, batch_size, num_workers):
    print(f"Loading {dataset} dataset.")
    train_loader = datasets.dataloader(
        root,
        dataset,
        batch_size,
        True,
        workers=num_workers,
        prune=False,
    )
    test_loader = datasets.dataloader(
        root,
        dataset,
        batch_size,
        False,
        workers=num_workers,
        prune=False,
    )

    return train_loader, test_loader


def load_model(
    root,
    model_type,
    dataset,
    device,
    pretrained=True,
    load_checkpoint=None,
):
    from Models import (
        mlp,
        lottery_resnet,
        lottery_vgg,
        tinyimagenet_vgg,
        tinyimagenet_resnet,
        imagenet_vgg,
        imagenet_resnet,
        imagenet_str_resnet,
        imagenet_str_mobilenetv1,
        imagenet_wf_resnet,
        imagenet_wf_mobilenetv1,
        probmask_cifar_resnet,
        probmask_cifar_vgg_bn,
    )

    default_models = {
        "fc": mlp.fc,
        "conv": mlp.conv,
    }
    lottery_models = {
        "vgg11": lottery_vgg.vgg11,
        "vgg11_bn": lottery_vgg.vgg11_bn,
        "vgg13": lottery_vgg.vgg13,
        "vgg13_bn": lottery_vgg.vgg13_bn,
        "vgg16": lottery_vgg.vgg16,
        "vgg16_bn": lottery_vgg.vgg16_bn,
        # "vgg19": lottery_vgg.vgg19,
        # "vgg19_bn": lottery_vgg.vgg19_bn,
        "vgg19_bn": probmask_cifar_vgg_bn.vgg19_bn,
        "resnet18": imagenet_str_resnet.ResNet18,
        "resnet20": lottery_resnet.resnet20,
        # "resnet32": lottery_resnet.resnet32,
        "resnet32": probmask_cifar_resnet.resnet32,
        "resnet44": lottery_resnet.resnet44,
        "resnet56": lottery_resnet.resnet56,
        "resnet110": lottery_resnet.resnet110,
        "resnet1202": lottery_resnet.resnet1202,
        "wrn20": lottery_resnet.wide_resnet20,
        "wrn28x2": lottery_resnet.wide_resnet28x2,
        "wrn28x4": lottery_resnet.wide_resnet28x4,
        "wrn32": lottery_resnet.wide_resnet32,
        "wrn32x4": lottery_resnet.wide_resnet32x4,
        "wrn44": lottery_resnet.wide_resnet44,
        "wrn56": lottery_resnet.wide_resnet56,
        "wrn110": lottery_resnet.wide_resnet110,
        "wrn1202": lottery_resnet.wide_resnet1202,
    }
    tinyimagenet_models = {
        "vgg11": tinyimagenet_vgg.vgg11,
        "vgg11_bn": tinyimagenet_vgg.vgg11_bn,
        "vgg13": tinyimagenet_vgg.vgg13,
        "vgg13_bn": tinyimagenet_vgg.vgg13_bn,
        "vgg16": tinyimagenet_vgg.vgg16,
        "vgg16_bn": tinyimagenet_vgg.vgg16_bn,
        "vgg19": tinyimagenet_vgg.vgg19,
        "vgg19_bn": tinyimagenet_vgg.vgg19_bn,
        "resnet18": tinyimagenet_resnet.resnet18,
        "resnet34": tinyimagenet_resnet.resnet34,
        "resnet50": tinyimagenet_resnet.resnet50,
        "resnet101": tinyimagenet_resnet.resnet101,
        "resnet152": tinyimagenet_resnet.resnet152,
        "wrn18": tinyimagenet_resnet.wide_resnet18,
        "wrn34": tinyimagenet_resnet.wide_resnet34,
        "wrn50": tinyimagenet_resnet.wide_resnet50,
        "wrn101": tinyimagenet_resnet.wide_resnet101,
        "wrn152": tinyimagenet_resnet.wide_resnet152,
    }
    imagenet_models = {
        "vgg11": imagenet_vgg.vgg11,
        "vgg11_bn": imagenet_vgg.vgg11_bn,
        "vgg13": imagenet_vgg.vgg13,
        "vgg13_bn": imagenet_vgg.vgg13_bn,
        "vgg16": imagenet_vgg.vgg16,
        "vgg16_bn": imagenet_vgg.vgg16_bn,
        "vgg19": imagenet_vgg.vgg19,
        "vgg19_bn": imagenet_vgg.vgg19_bn,
        # "resnet18": imagenet_resnet.resnet18,
        "resnet34": imagenet_resnet.resnet34,
        # "resnet50": imagenet_resnet.resnet50,
        "resnet101": imagenet_resnet.resnet101,
        "resnet152": imagenet_resnet.resnet152,
        "wrn50": imagenet_resnet.wide_resnet50_2,
        "wrn101": imagenet_resnet.wide_resnet101_2,
        # NOTE: using ResNet-50 checkpoint in STR, https://github.com/RAIVNLab/STR
        "resnet50": imagenet_str_resnet.ResNet50,
        # NOTE: using MobileNet checkpoint in STR, https://github.com/RAIVNLab/STR
        "mobilenetv1": imagenet_str_mobilenetv1.MobileNetV1,
        "resnet50_wf": imagenet_wf_resnet.resnet50,
        "mobilenetv1_wf": imagenet_wf_mobilenetv1.mobilenet,
    }
    models = {
        "default": default_models,
        "lottery": lottery_models,
        "tiny_imagenet": tinyimagenet_models,
        "imagenet": imagenet_models,
    }

    model_class = "lottery" if dataset in ["cifar10", "cifar100"] else dataset
    if dataset == "imagenet":
        print("WARNING: ImageNet models do not implement `dense_classifier`.")

    model_kwargs = dict(
        cifar10={
            "input_shape": (3, 32, 32),
            "num_classes": 10,
        },
        cifar100={
            "input_shape": (3, 32, 32),
            "num_classes": 100,
        },
        tiny_imagenet={
            "input_shape": (3, 64, 64),
            "num_classes": 200,
        },
        imagenet={
            "input_shape": (3, 224, 224),
            "num_classes": 1000,
            "pretrained": True,
        },
    )

    if dataset != "imagenet":
        model = models[model_class][model_type](**model_kwargs[dataset]).to(device)

        ckpt_path = f"{root}/pretrained/{dataset}/{model_type}.pt"
        if os.path.exists(ckpt_path) and pretrained:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            print("\n\n")
            for key in ckpt.keys():
                if key not in ["model", "optimizer", "scheduler"]:
                    print(f"ckpt  {key} = {ckpt[key]}")
            print(f"Pretrained {dataset} {model_type} loaded successfully.\n\n")
        else:
            if dataset == "tiny_imagenet":
                import torchvision.models as imagenet_models

                Warning(
                    f"\nNO avaible checkpoint in {ckpt_path},  pretrained = {pretrained}, training from scratch ...\n"
                )
                if hasattr(imagenet_models, model_type):
                    imagenet_ckpt = getattr(imagenet_models, model_type)(
                        pretrained=True
                    ).state_dict()
                    imagenet_ckpt = {
                        k: imagenet_ckpt[k]
                        for k in imagenet_ckpt.keys()
                        if "fc" not in k and "classifier" not in k
                    }
                    model.load_state_dict(imagenet_ckpt, strict=False)
                    Warning(
                        "\nLoading ImageNet pretrained model (except FC) instead ...\n"
                    )

            elif dataset in ["cifar10", "cifar100"] and model_type == "resnet18":
                import timm

                model = timm.create_model("resnet18", pretrained=False)
                num_classes = 10 if dataset == "cifar10" else 100

                # override model
                model.conv1 = nn.Conv2d(
                    3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
                )
                model.maxpool = nn.Identity()  # type: ignore
                model.fc = nn.Linear(512, num_classes)

                model.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        f"https://huggingface.co/edadaltocg/resnet18_{dataset}/resolve/main/pytorch_model.bin",
                        map_location="cpu",
                        file_name=f"resnet18_{dataset}.pth",
                    )
                )
                model = model.to(device)
            else:
                print(
                    f"\n\tWarning: NO avaible checkpoint in {ckpt_path}, pretrained = {pretrained}, training from scratch ...\n"
                )

    if pretrained and dataset == "imagenet":

        if model_type in [
            "deit_tiny",
            "deit_small",
            "deit_base",
            "deit_tiny_distilled",
            "deit_small_distilled",
            "deit_base_distilled",
        ]:

            model_config = {
                "deit_tiny": "deit_tiny_patch16_224",
                "deit_small": "deit_small_patch16_224",
                "deit_base": "deit_base_patch16_224",
                "deit_tiny_distilled": "deit_tiny_distilled_patch16_224",
                "deit_small_distilled": "deit_small_distilled_patch16_224",
                "deit_base_distilled": "deit_base_distilled_patch16_224",
            }
            model = torch.hub.load(
                "facebookresearch/deit:main", model_config[model_type], pretrained=True
            )
            model = model.to(device)
            return model

        elif model_type in ["swin_t", "swin_s"]:
            import torchvision

            model = getattr(torchvision.models, model_type)(weights="IMAGENET1K_V1")
            model = model.to(device)
            return model

        elif model_type in [
            "vgg19_bn",
            "resnet50",
            "mobilenetv1",
            "resnet50_wf",
            "mobilenetv1_wf",
        ]:

            model = models[model_class][model_type]().to(device)
            # https://github.com/RAIVNLab/STR/tree/master
            str_ckpt_paths = {
                "resnet50": f"{root}/pretrained/{dataset}/ResNet50-Dense.pth",
                "mobilenetv1": f"{root}/pretrained/{dataset}/MobileNetV1-Dense.pth",
                "resnet50_wf": "<path_to_ckpt>",
                "mobilenetv1_wf": "<path_to_ckpt>",
            }
            ckpt_path = str_ckpt_paths[model_type]

            if os.path.exists(ckpt_path):
                print(f"Loading local model checkpoints from {ckpt_path} ...")
                ckpt = torch.load(ckpt_path, map_location=device)
                ckpt_state_dict = ckpt["state_dict"]
                ckpt_state_dict = {
                    k.replace("module.", ""): v
                    for k, v in ckpt_state_dict.items()
                    if "mask" not in k
                }

                model.load_state_dict(ckpt_state_dict, strict=True)
                print(f"Pretrained {dataset} {model_type} loaded successfully.")
                return model
            else:
                from torchvision.models import (
                    vgg19_bn as imvgg19_bn,
                    resnet50 as imresnet50,
                    mobilenet_v2 as immobilenet_v2,
                )

                models_image = {
                    "resnet50": imresnet50,
                    "vgg19_bn": imvgg19_bn,
                    "mobilenet_v2": immobilenet_v2,
                }
                models_checkpoint = {
                    "resnet50": "IMAGENET1K_V1",
                    "vgg19_bn": "IMAGENET1K_V1",
                    "mobilenetv1": "IMAGENET1K_V1",
                }
                model = models_image[model_type](
                    weights=models_checkpoint[model_type]
                ).to(device)
                print(f"Pretrained {dataset} {model_type} loaded from torchvision.")

        elif model_type in ["resnet50v2"]:
            import torchvision

            model = torchvision.models.resnet50(weights="IMAGENET1K_V2").to(device)
            print(f"Pretrained {dataset} {model_type} loaded from torchvision.")

        else:
            raise NotImplementedError

    if load_checkpoint is not None:
        ckpt_path = load_checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)

    return model


def load_optimizer(opt, **kwargs):
    import torch.optim as optim

    optimizers = {
        "adam": (optim.Adam, {"weight_decay": 5e-5, **kwargs}),
        "sgd": (optim.SGD, {"weight_decay": 5e-5, **kwargs}),
        "momentum": (optim.SGD, {"momentum": 0.875, "nesterov": True, **kwargs}),
        "rms": (optim.RMSprop, {**kwargs}),
        "adamw": (optim.AdamW, {"weight_decay": 0.05, **kwargs}),
    }
    return optimizers[opt]


def load_pruner(model, pruner, structural, mask_dim, **kwargs):
    if structural:
        from Methods import structural_pruners

        pruner = vars(structural_pruners)[pruner](model, mask_dim, **kwargs)
        print(f"\nStructured pruner loaded successfully, mask-dim = {mask_dim}.\n")
    else:
        from Methods import pruners

        pruner = vars(pruners)[pruner](model, **kwargs)
        print(f"\nUn-structured pruner loaded successfully.\n")

    return pruner


# only apply to masked-model
def load_sparsity(model, sparsity, scope="global"):

    if scope == "global":
        return sparsity

    elif scope == "local":
        return [sparsity for _, _, _ in utils.masked_params(model)]

    else:
        raise NotImplementedError


def load_sparsity_schedule(sparsity, epochs, schedule="exponential", end_sparsity=None):

    def _get_sparse(sparsity, epoch, schedule):
        if schedule == "exponential":
            return sparsity ** ((epoch + 1) / epochs)

        elif schedule == "linear":
            return 1.0 - (1.0 - sparsity) * ((epoch + 1) / epochs)

        elif schedule == "increase":
            if end_sparsity:
                assert sparsity < end_sparsity <= 1.0
                end_s = end_sparsity
            else:
                end_s = 1.0

            return sparsity + (end_s - sparsity) * ((epoch + 1) / epochs)

    if isinstance(sparsity, list):
        return [
            [_get_sparse(s, epoch, schedule) for s in sparsity]
            for epoch in range(epochs)
        ]
    else:
        return [_get_sparse(sparsity, epoch, schedule) for epoch in range(epochs)]
