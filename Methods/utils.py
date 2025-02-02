import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import layers


# judge whether this module is prunable / masked
def is_masked_module(m):
    return hasattr(m, "weight_mask") or hasattr(m, "bias_mask")


# initialize masks and register forward hooks, e.g. model(param * mask).forward(x)
def init_masks_hooks(model, structural=False, mask_dim=0):
    init_masks(model, structural, mask_dim)
    reg_fwd_hooks(model, structural, mask_dim)


# initialize masks for (un)-structural pruning
def init_masks(model, structural=False, mask_dim=1):
    if structural:
        _init_structural_masks(model, mask_dim)
    else:
        _init_unstructural_masks(model)


def _init_structural_masks(model, mask_dim):
    assert mask_dim in [0, 1]  # 0 = out_dim, 1 = in_dim

    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, layers.Linear, layers.Conv2d)):
            weight_mask_structural = torch.ones(m.weight.data.size(mask_dim)).to(
                m.weight.device
            )
            m.register_buffer("weight_mask", weight_mask_structural)
            if m.bias is not None:
                m.register_buffer(
                    "bias_mask", torch.ones_like(m.bias.data).to(m.bias.device)
                )

    print("\nStructral masks initialized successfully ...\n")


def _init_unstructural_masks(model):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, layers.Linear, layers.Conv2d)):
            m.register_buffer(
                "weight_mask", torch.ones_like(m.weight.data).to(m.weight.device)
            )
            if m.bias is not None:
                m.register_buffer(
                    "bias_mask", torch.ones_like(m.bias.data).to(m.bias.device)
                )

    print("\nUnstructral masks initialized successfully ...\n")


# register forward hooks for applying masks, e.g. model(param * mask).forward(x)
def reg_fwd_hooks(model, structural, mask_dim):
    def _apply_mask(module, input, output):
        if hasattr(module, "weight_mask"):
            if structural:
                if module.weight.dim() == 4:
                    view_shape = (-1, 1, 1, 1) if mask_dim == 0 else (1, -1, 1, 1)
                else:
                    view_shape = (-1, 1) if mask_dim == 0 else (1, -1)

                weight_mask_expanded = module.weight_mask.view(view_shape).expand_as(
                    module.weight
                )
                w = module.weight * weight_mask_expanded

            else:
                w = module.weight * module.weight_mask
        else:
            w = module.weight

        if hasattr(module, "bias_mask"):
            b = module.bias * module.bias_mask
        else:
            if structural:
                if mask_dim == 0:  # output channel prune
                    b = (
                        module.bias * module.weight_mask
                        if module.bias is not None
                        else None
                    )
                elif mask_dim == 1:  # input channel prune
                    b = module.bias if module.bias is not None else None
                else:
                    raise ValueError(f"mask_dim = {mask_dim} should be 0 or 1")

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
                        input,
                        module._padding_repeated_twice,
                        mode=module.padding_mode,
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

    for n, m in model.named_modules():
        if isinstance(
            m, (nn.Linear, nn.Conv2d, layers.Linear, layers.Conv2d)
        ) and is_masked_module(m):
            if hasattr(m, "fwd_hook"):
                m.fwd_hook.remove()
            m.fwd_hook = m.register_forward_hook(_apply_mask)
            print(f"Hook registered for {n} successfully ...")

        else:
            print(f"Failed to register hook for {n}")

    hook_type = "Structural" if structural else "Unstructural"
    print(f"\t{hook_type} forward-hooks registered successfully ...")


# enumerate masked parameters
def masked_params(model):
    for name, module in model.named_modules():
        if hasattr(module, "weight_mask"):
            yield f"{name}.weight", module.weight_mask, module.weight

        if hasattr(module, "bias_mask"):
            yield f"{name}.bias", module.bias_mask, module.bias


# delete masks on unmasked modules, e.g. bias, bn, shortcut, last_fc, first_conv1
def free_modules(
    model, free_bn, free_Id, free_bias, free_conv1, free_lastfc=None, verbose=True
):
    def _free_weight_masks(module):
        if hasattr(module, "weight_mask"):
            delattr(module, "weight_mask")

    def _free_bias_masks(module):
        if hasattr(module, "bias_mask"):
            delattr(module, "bias_mask")

    def _free_hooks(module):
        module.fwd_hook.remove()

    is_conv1 = True
    bn_count, Id_count, bias_count = 0, 0, 0

    for module in model.modules():
        if isinstance(module, (layers.Conv2d, nn.Conv2d)):
            if free_conv1 and is_conv1:
                _free_weight_masks(module)
                _free_bias_masks(module)
                is_conv1 = False
                bias_count += 1

        elif isinstance(
            module,
            (layers.BatchNorm1d, layers.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm2d),
        ):
            if free_bn:
                _free_weight_masks(module)
                _free_bias_masks(module)
                bn_count += 1
                bias_count += 1

        elif isinstance(module, (layers.Identity1d, layers.Identity2d)):
            if free_Id:
                _free_weight_masks(module)
                _free_bias_masks(module)
                Id_count += 1
                bias_count += 1

        if free_bias:
            if hasattr(module, "bias_mask") and isinstance(
                module, (layers.Conv2d, nn.Conv2d)
            ):
                _free_bias_masks(module)
                bias_count += 1

        if (
            not hasattr(module, "bias_mask")
            and not hasattr(module, "weight_mask")
            and hasattr(module, "fwd_hook")
        ):
            _free_hooks(module)

    if free_lastfc:
        last_module = [
            module for module in model.modules() if hasattr(module, "weight_mask")
        ][-1]
        if not isinstance(last_module, nn.Linear):
            print(
                f"\nWarning: Check the last module = {last_module}. Expected to be nn.Linear.\n"
            )

        _free_weight_masks(last_module)
        _free_bias_masks(module)
        _free_hooks(module)

    if verbose:
        print(f"\n\tFree-bn = {free_bn}, {bn_count} bn masks are deleted.")
        print(f"\tFree-Id = {free_Id}, {Id_count} Identity masks are deleted.")
        print(
            f"\tFree-bias = {free_bias}, {bias_count} bias masks are deleted. (FC-bias is excluded!)"
        )
        print(f"\tFree-Conv1 = {free_conv1}.")
        print(f"\tFree-last fc = {free_lastfc}.")
