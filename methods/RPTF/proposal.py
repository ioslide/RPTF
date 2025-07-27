import math
import numpy as np
from copy import deepcopy
from collections import defaultdict, OrderedDict
from loguru import logger as log
from core.model.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_V2_MASK, IMAGENET_D109_MASK

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from torchvision.models import resnet50

from core.model.build import split_up_model
from methods.RPTF.transformers_cotta import get_tta_transforms
from methods.RPTF.cbp_bn import RobustBN1d, RobustBN2d, MomentumBN

__all__ = ["setup"]


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Calculate entropy of softmax distribution."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def softmax_clamp(logits):
    """Apply softmax and clamp values to prevent numerical instability."""
    logits = F.softmax(logits, dim=1)
    logits = torch.clamp(logits, min=0.0, max=0.99)
    return logits

class SupSoftLikelihoodRatio(nn.Module):
    def __init__(self, gamma=1e-5):
        super(SupSoftLikelihoodRatio, self).__init__()
        self.gamma = gamma
        self.eps = 1e-5

    def __call__(self, logits, target_logits):
        logits = softmax_clamp(logits)
        target_logits = softmax_clamp(target_logits)

        ratio_term = (target_logits * (1 - self.gamma)) / ((1 - target_logits) + self.eps) + self.gamma
        return -1 * (logits * torch.log(ratio_term) / (1 - self.gamma)).sum(1)


class RPTF(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        
        model, bn_layers = configure_model(model, cfg)
        self.bn_layers = bn_layers
        
        params, param_names = collect_params(model)
        log.info(f"==>> param_names: {param_names}")
        
        self.optimizer = self._setup_optimizer(cfg, params)
        self.params = params
        self.param_names = param_names
        self.model = model
        self.model.train()
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS
        
        self.feature_extractor, self.classifier = split_up_model(self.model, cfg.MODEL.ARCH, cfg.CORRUPTION.DATASET)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
        self.src_model = deepcopy(self.model)
        
        if self.cfg.CORRUPTION.DATASET in ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_d109"]:
            mask = eval(f"{cfg.CORRUPTION.DATASET.upper()}_MASK")
            self.classifier_weight = self.classifier[0].weight[mask, :]
        else:
            self.classifier_weight = self.classifier.weight
        
        for param in self.src_model.parameters():
            param.detach_()
            
        assert self.steps > 0, "RPTF requires >= 1 step(s) to forward and update"
        
        self.class_counts = torch.zeros(self.classifier_weight.shape[0]).cuda()
        self.slr = SupSoftLikelihoodRatio(self.cfg.ADAPTER.RPTF.SUP_SLR_GAMMA)
        
        self.transforms = get_tta_transforms(
            cfg=cfg,
            padding_mode="reflect",
            cotta_augs=True
        )
    
    def _setup_optimizer(self, cfg, params):
        if cfg.OPTIM.METHOD == "SGD":
            return torch.optim.SGD(
                params,
                lr=float(cfg.OPTIM.LR),
                momentum=float(cfg.OPTIM.MOMENTUM),
                weight_decay=float(cfg.OPTIM.WD)
            )
        elif cfg.OPTIM.METHOD == "Adam":
            return torch.optim.Adam(
                params,
                lr=float(cfg.OPTIM.LR),
                weight_decay=float(cfg.OPTIM.WD)
            )
        else:
            raise ValueError(f"Invalid optimizer method: {cfg.OPTIM.METHOD}")
    
    def _get_dataset_mask(self):
        dataset = self.cfg.CORRUPTION.DATASET
        if dataset == "imagenet_a":
            return IMAGENET_A_MASK
        elif dataset == "imagenet_r":
            return IMAGENET_R_MASK
        elif dataset == "imagenet_v2":
            return IMAGENET_V2_MASK
        elif dataset == "imagenet_d109":
            return IMAGENET_D109_MASK
        return None

    def forward(self, x, **kwargs):
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        self.optimizer.zero_grad()
        x_aug = self.transforms(x)
        orig_logits = self.model(x)
        aug_logits = self.model(x_aug)
        weights = self._calculate_sample_weights(orig_logits)
        slr_loss_a = self._compute_weighted_loss(orig_logits, orig_logits, weights)
        slr_loss_b = self._compute_weighted_loss(orig_logits, aug_logits, weights)
        norm_modules, cons_losses = self._collect_bn_losses()
        norm_cons_loss = torch.stack(cons_losses).sum() if cons_losses else torch.tensor(0.0).to(x.device)
        bn_weights = [module.weight for module in norm_modules]
        lamada_1, lamada_2 = self._compute_loss_weights(slr_loss_a, slr_loss_b, bn_weights, orig_logits)
        loss = slr_loss_a + slr_loss_b * lamada_1 + norm_cons_loss * lamada_2
        loss.backward()
        self._update_bn_importance(norm_modules)
        self.optimizer.step()
        for module in self.model.modules():
            if isinstance(module, (RobustBN2d, RobustBN1d)):
                module.reinit_low_utility_param()
        return orig_logits
    
    def _update_bn_importance(self, norm_modules):
        for module in norm_modules:
            module.update_importance_based_on_grads()
    
    def _collect_bn_losses(self):
        norm_modules = []
        cons_losses = []
        for module in self.model.modules():
            if isinstance(module, (RobustBN2d, RobustBN1d)):
                norm_modules.append(module)
                cons_losses.append(module.norm_loss)
        
        return norm_modules, cons_losses
    
    def _compute_weighted_loss(self, logits, target_logits, weights):
        loss_per_sample = self.slr(logits, target_logits)
        weighted_loss = (loss_per_sample * weights).sum() / logits.shape[0]
        return weighted_loss
    
    def _compute_loss_weights(self, loss_a, loss_b, bn_weights, orig_logits):
        if not bn_weights:
            return self.cfg.ADAPTER.RPTF.LAMBDA * orig_logits.shape[1], 0.0
        grads_a = torch.autograd.grad(loss_a, bn_weights, retain_graph=True)
        grads_b = torch.autograd.grad(loss_b, bn_weights, retain_graph=True)
        grad_norm_a = sum([torch.norm(g, dim=0).sum().item() for g in grads_a])
        grad_norm_b = sum([torch.norm(g, dim=0).sum().item() for g in grads_b])
        ratio_kd = grad_norm_a / (grad_norm_b + 1e-8)
        lamada_2 = (1 / len(bn_weights)) * ratio_kd
        lamada_1 = self.cfg.ADAPTER.RPTF.LAMBDA * orig_logits.shape[1]
        return lamada_1, lamada_2

    def _calculate_sample_weights(self, logits):
        with torch.no_grad():
            entropy = softmax_entropy(logits)
            weights_cert = -entropy
            weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min() + 1e-6)
            probs = logits.softmax(1)
            pred_classes = probs.argmax(dim=1)
            for c in range(self.classifier_weight.shape[0]):
                self.class_counts[c] += (pred_classes == c).sum().float()
            cls_prior = self.class_counts / (self.class_counts.sum() + 1e-8)
            cls_prior = F.normalize(cls_prior, p=2, dim=0)
            n = cls_prior.sum()
            H = -((cls_prior/n) * torch.log(cls_prior/n + 1e-6)).sum()
            imbalance_ratio = H/torch.log(torch.tensor(cls_prior.shape[0], dtype=torch.float, device=cls_prior.device))
            base_cos_sim = 1 - F.cosine_similarity(
                self.classifier_weight.mean(1).unsqueeze(0), 
                probs, 
                dim=1
            )
            cls_weights = torch.min(cls_prior + 1e-6) / (cls_prior + 1e-6)
            pred_cls_weights = torch.zeros_like(probs)
            for i in range(cls_weights.shape[0]):
                pred_cls_weights[:, i] = cls_weights[i] * probs[:, i]
            sample_imbalance_weights = pred_cls_weights.sum(1)
            weights_div = base_cos_sim * sample_imbalance_weights
            weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min() + 1e-6)
            under_represented = torch.zeros_like(weights_div)
            for i in range(probs.shape[0]):
                primary_class = probs[i].argmax()
                if cls_prior[primary_class] < torch.mean(cls_prior):
                    under_represented[i] = 1.0
            rs_over = torch.min(torch.tensor(1.0), imbalance_ratio)
            rs_under = torch.min(torch.tensor(1.0), imbalance_ratio * 2.0)
            scaling = under_represented * rs_under + (1 - under_represented) * rs_over
            weights_div = weights_div * scaling
            weights = torch.exp(weights_cert * weights_div) ** self.cfg.ADAPTER.RPTF.WEIGHT_EXPONENT
        return weights

def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])
        else:
            setattr(module, names[i], value)


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)
    return module


def copy_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def collect_params(model: nn.Module):
    params = []
    names = []
    
    norm_layer_types = (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm, RobustBN1d, RobustBN2d, MomentumBN)
    
    for name, module in model.named_modules():
        if isinstance(module, norm_layer_types):
            for param_name, param in module.named_parameters():
                if param.requires_grad and param_name in ['weight', 'bias']:
                    params.append(param)
                    names.append(f"{name}.{param_name}")
    
    return params, names


def configure_model_norm(model: nn.Module, cfg):
    model.train()
    model.requires_grad_(False)
    
    norm_layer_types = (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)
    
    for module in model.modules():
        if isinstance(module, norm_layer_types):
            module.train()
            module.requires_grad_(True)
            
    return model, []


def configure_model(model: nn.Module, cfg):
    model.eval()
    model.requires_grad_(False)
    
    bn_layers = []
    layer_dict = {}
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            bn_layers.append(name)
    
    for name in bn_layers:
        bn_layer = get_named_submodule(model, name)
        if isinstance(bn_layer, nn.BatchNorm2d):
            enhanced_bn = RobustBN2d(bn_layer, cfg=cfg)
        elif isinstance(bn_layer, nn.BatchNorm1d):
            enhanced_bn = RobustBN1d(bn_layer, cfg=cfg)
        else:
            raise RuntimeError(f"Unsupported BatchNorm type: {type(bn_layer)}")
        
        enhanced_bn.requires_grad_(True)
        layer_dict[name] = enhanced_bn
        set_named_submodule(model, name, enhanced_bn)

    # Set adjacent layers for each BN layer to help with gradient flow analysis
    _connect_adjacent_layers(model, bn_layers, layer_dict)
    
    return model, bn_layers


def _connect_adjacent_layers(model, bn_layers, layer_dict):
    """Connect each BN layer to its adjacent convolutional/linear layers."""
    for bn_path in bn_layers:
        bn_layer = layer_dict[bn_path]
        parent_path = '.'.join(bn_path.split('.')[:-1])
        
        prev_layer = None
        next_layer = None
        
        if parent_path:
            parent = model
            for part in parent_path.split('.'):
                parent = getattr(parent, part)
            
            layer_name = bn_path.split('.')[-1]
            children = list(parent.named_children())
            
            bn_idx = -1
            for idx, (name, _) in enumerate(children):
                if name == layer_name:
                    bn_idx = idx
                    break
            
            if bn_idx >= 0:
                for idx in range(bn_idx-1, -1, -1):
                    prev_name, prev_module = children[idx]
                    if isinstance(prev_module, (nn.Conv2d, nn.Linear)):
                        prev_layer = prev_module
                        break
                
                for idx in range(bn_idx+1, len(children)):
                    next_name, next_module = children[idx]
                    if isinstance(next_module, nn.Conv2d):
                        if next_module.in_channels == bn_layer.num_features:
                            next_layer = next_module
                            break
                    elif isinstance(next_module, nn.Linear):
                        if next_module.in_features == bn_layer.num_features:
                            next_layer = next_module
                            break
        
        # Connect layers
        bn_layer.set_adjacent_layers(prev_layer, next_layer)


def setup(model: nn.Module, cfg):
    log.info("Setup TTA method: RPTF")
    return RPTF(cfg, model)