import timm
import torch
import torch.nn as nn
import core.model.resnet as Resnet
from core.model.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_V2_MASK, IMAGENET_D109_MASK
from copy import deepcopy
from robustbench.utils import load_model
from robustbench.model_zoo.architectures.utils_architectures import ImageNormalizer
from packaging import version
from loguru import logger as log
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
import torchvision
from transformers import BeitFeatureExtractor, Data2VecVisionForImageClassification
import getpass
from typing import Union

def build_model(cfg):
    if cfg.CORRUPTION.DATASET in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
        log.info(f"Wrapping model with mask for dataset {cfg.CORRUPTION.DATASET}")
        base_model = get_torchvision_model(cfg.MODEL.ARCH, 'IMAGENET1K_V1')
        mask = eval(f"{cfg.CORRUPTION.DATASET.upper()}_MASK")
        base_model = ImageNetXWrapper(base_model, mask=mask)
        return base_model.cuda()

    try:
        if cfg.MODEL.ARCH in ['resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']:
            base_model = Resnet.__dict__[cfg.MODEL.ARCH](pretrained=True,num_classes=cfg.CORRUPTION.NUM_CLASS).cuda()
        elif cfg.MODEL.ARCH in ['resnet50_gn','resnetv2_50d_gn']:
            base_model = timm.create_model('resnet50_gn', pretrained=True,num_classes=cfg.CORRUPTION.NUM_CLASS).cuda()
        elif cfg.MODEL.ARCH in ['efficientnet_b4']:
            base_model = timm.create_model('efficientnet_b4', pretrained=True,num_classes=cfg.CORRUPTION.NUM_CLASS).cuda()
        elif cfg.MODEL.ARCH in ['vit_b_16','swin_b','convnext_tiny','mobilenet_v3_small','mobilenet_v3_large','mobilenet_v2','swin_v2_b','swin_v2_s','efficientnet_v2_s','efficientnet_v2_m','convnext_base','densenet161','densenet121','wide_resnet50_2','wide_resnet101_2','resnext50_32x4d']:
            base_model = get_torchvision_model(cfg.MODEL.ARCH,'IMAGENET1K_V1')
        else:
            base_model = load_model(
                model_name=cfg.MODEL.ARCH,
                dataset=cfg.CORRUPTION.DATASET.split('_')[0], 
                threat_model='corruptions'
            )
    except ValueError:
        base_model = load_model(
            model_name=cfg.MODEL.ARCH,
            dataset=cfg.CORRUPTION.DATASET.split('_')[0], 
            threat_model='corruptions'
        )
        
    return base_model.cuda()

        
class TransformerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

class ImageNetXMaskingLayer(nn.Module):
    """ Following: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    """
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]

class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

        self.masking_layer = ImageNetXMaskingLayer(mask)

    def forward(self, x):
        logits = self.model(self.normalize(x))
        return self.masking_layer(logits)

def split_up_model(model, arch_name, dataset_name):

    if dataset_name in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
        mask = eval(f"{dataset_name.upper()}_MASK")
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(mask))
        return encoder, classifier

    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model, model.model.pretrained_cfg["classifier"]):
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)
    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_ResNeXt":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1], nn.Flatten())
        classifier = model.classifier
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
    elif "wide_resnet50_2" in arch_name or "resnext50_32x4d" in arch_name:
        encoder = nn.Sequential(*list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "resnet" in arch_name or arch_name in {"Hendrycks2020AugMix", "Hendrycks2020Many", "Geirhos2018_SIN"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        classifier = model.fc
    elif "Standard_R50" in arch_name:
        encoder = nn.Sequential(*list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute, model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")

    return encoder, classifier


def get_torchvision_model(model_name: str, weight_version: str = "IMAGENET1K_V1"):
    assert version.parse(torchvision.__version__) >= version.parse("0.13"), "Torchvision version has to be >= 0.13"

    # check if the specified model name is available in torchvision
    available_models = torchvision.models.list_models(module=torchvision.models)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in torchvision. Choose from: {available_models}")

    # get the weight object of the specified model and the available weight initialization names
    model_weights = torchvision.models.get_model_weights(model_name)
    available_weights = [init_name for init_name in dir(model_weights) if "IMAGENET1K" in init_name]

    # check if the specified type of weights is available
    if weight_version not in available_weights:
        raise ValueError(f"Weight type '{weight_version}' is not supported for torchvision model '{model_name}'."
                         f" Choose from: {available_weights}")

    # restore the specified weights
    model_weights = getattr(model_weights, weight_version)

    # setup the specified model and initialize it with the specified pre-trained weights
    model = torchvision.models.get_model(model_name, weights=model_weights)

    # get the transformation and add the input normalization to the model
    transform = model_weights.transforms()
    model = normalize_model(model, transform.mean, transform.std)
    log.info(f"Successfully restored '{weight_version}' pre-trained weights"
                f" for model '{model_name}' from torchvision!")

    return model

def get_timm_model(model_name: str):
    """
    Restore a pre-trained model from timm: https://github.com/huggingface/pytorch-image-models/tree/main/timm
    Quickstart: https://huggingface.co/docs/timm/quickstart
    Input:
        model_name: Name of the model to create and initialize with pre-trained weights
    Returns:
        model: The pre-trained model
        preprocess: The corresponding input pre-processing
    """
    # check if the defined model name is supported as pre-trained model
    available_models = timm.list_models(pretrained=True)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in timm. Choose from: {available_models}")

    # setup pre-trained model
    model = timm.create_model(model_name, pretrained=True)
    log.info(f"Successfully restored the weights of '{model_name}' from timm.")

    # add the corresponding input normalization to the model
    if hasattr(model, "pretrained_cfg"):
        log.info(f"General model information: {model.pretrained_cfg}")
        log.info(f"Adding input normalization to the model using: mean={model.pretrained_cfg['mean']} \t std={model.pretrained_cfg['std']}")
        model = normalize_model(model, mean=model.pretrained_cfg["mean"], std=model.pretrained_cfg["std"])
    else:
        raise AttributeError(f"Attribute 'pretrained_cfg' is missing for model '{model_name}' from timm."
                             f" This prevents adding the correct input normalization to the model!")
    return model