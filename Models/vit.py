from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
)


# def wide_resnet1202(input_shape, num_classes, dense_classifier=False, pretrained=False):
#     plan = _plan(1202, 32)
#     return _resnet('wide_resnet1202', plan, num_classes, dense_classifier, pretrained)


def get_vitb16(input_shape, num_classes, dense_classifier=False, pretrained=False):
    # assert input_shape[-1] == 224, 'only support 224*224 '
    config = AutoConfig.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        finetuning_task="image-classification",
    )
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
    )
    return model
