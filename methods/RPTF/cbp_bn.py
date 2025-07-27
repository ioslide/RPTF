import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from loguru import logger as log
from collections import defaultdict

class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, cfg):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.eps = bn_layer.eps
        self.momentum = bn_layer.momentum
        self.cfg = cfg
        self.UTILITY_THRESHOLD = self.cfg.ADAPTER.TEST.UTILITY_THRESHOLD
        self.RELATIVE_UTILITY_THRESHOLD = self.cfg.ADAPTER.TEST.RELATIVE_UTILITY_THRESHOLD
        self.PATIENCE = self.cfg.ADAPTER.TEST.LOW_UTILITY_COUNT_PATIENCE
        self.prev_layer = None
        self.next_layer = None
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("global_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("global_var", deepcopy(bn_layer.running_var))
        self.weight = nn.Parameter(deepcopy(bn_layer.weight), requires_grad=True)
        self.bias = nn.Parameter(deepcopy(bn_layer.bias), requires_grad=True)
        self.initial_weight = deepcopy(bn_layer.weight)
        self.initial_bias = deepcopy(bn_layer.bias)
        self.utility = nn.Parameter(torch.ones(self.num_features), requires_grad=False)
        self.register_buffer('low_utility_count', torch.zeros_like(self.weight))
        self.register_buffer('utility_score', torch.ones(self.num_features))
        self.register_buffer('sensitivity_score', torch.zeros_like(self.weight))

    def forward(self, x):
        self.global_mean = self.global_mean.detach()
        self.global_var = self.global_var.detach()

    def set_adjacent_layers(self, prev_layer=None, next_layer=None):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.register_buffer('next_layer_impact', torch.zeros(self.num_features) if next_layer is not None else None)

    def update_importance_based_on_grads(self):
        if self.next_layer is not None and hasattr(self.next_layer, 'weight') and self.training:
            self.sensitivity_score = self.sensitivity_score + self.weight.grad.pow(2)
            utility = self.min_max_normalize(self.sensitivity_score).detach().clone()
            self.utility_score.copy_(utility)
            low_utility_mask = utility <= utility.mean()
            self.low_utility_count[low_utility_mask] += 1
            self.low_utility_count[~low_utility_mask] = torch.maximum(
                self.low_utility_count[~low_utility_mask] - 0.5,
                torch.zeros_like(self.low_utility_count[~low_utility_mask])
            )
        else:
            pass
    
    def min_max_normalize(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-6)

    def reinit_low_utility_param(self):
        persistent_low_idx = torch.where(self.low_utility_count > self.PATIENCE)[0]
        eligible_indices = persistent_low_idx
        if len(persistent_low_idx) > 0:
            _, worst_idx = torch.topk(
                - self.utility_score[persistent_low_idx], 
                min(persistent_low_idx.shape[0], int(0.01 * self.num_features))
            )
            eligible_indices = persistent_low_idx[worst_idx]

        if len(eligible_indices) > 0:
            with torch.no_grad():
                weight_data = self.weight.data.clone()
                weight_data[eligible_indices] = self.initial_weight[eligible_indices]
                self.weight.data.copy_(weight_data)

                bias_data = self.bias.data.clone()
                bias_data[eligible_indices] = self.initial_bias[eligible_indices]
                self.bias.data.copy_(bias_data)

        mask = self.cfg.ADAPTER.TEST.MU
        restored_weight = self.initial_weight.to(self.weight.device) * (1. - mask) + self.weight.data * mask
        self.weight.data.copy_(restored_weight)
        restored_bias = self.initial_bias.to(self.bias.device) * (1. - mask) + self.bias.data * mask
        self.bias.data.copy_(restored_bias)

class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2], unbiased=False, keepdim=False)
            new_mean = self.momentum * self.global_mean + (1 - self.momentum) * b_mean
            new_var = self.momentum * self.global_var + (1 - self.momentum) * b_var

            self.global_mean = (1 - self.momentum) * self.global_mean + self.momentum * deepcopy(new_mean.detach()).view(-1)
            self.global_var = (1 - self.momentum) * self.global_var + self.momentum * deepcopy(new_var.detach()).view(-1)
            new_mean, new_var = new_mean.view(1, -1, 1), new_var.view(1, -1, 1)

            global_mean = self.global_mean.view(1, -1, 1)
            global_var = self.global_var.view(1, -1, 1)
            st_dist = ((b_mean - self.global_mean) ** 2).mean(0)[None] + ((b_var.sqrt() - self.global_var.sqrt()) ** 2).mean(0)[None]
            self.norm_loss = st_dist.mean()
        else:
            new_mean, new_var = self.global_mean.view(1, -1, 1), self.global_var.view(1, -1, 1)

        x = (x - new_mean) / torch.sqrt(new_var + self.eps)
        weight = self.weight.view(1, -1, 1)
        bias = self.bias.view(1, -1, 1)
        return x * weight + bias

class RobustBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)
            new_mean = self.momentum * self.global_mean + (1 - self.momentum) * b_mean
            new_var = self.momentum * self.global_var + (1 - self.momentum) * b_var
            new_mean, new_var = new_mean.view(1, -1, 1, 1), new_var.view(1, -1, 1, 1)

            self.global_mean = (1 - self.momentum) * self.global_mean + self.momentum * deepcopy(new_mean.detach()).view(-1)
            self.global_var = (1 - self.momentum) * self.global_var + self.momentum * deepcopy(new_var.detach()).view(-1)

            global_mean = self.global_mean.view(1, -1, 1, 1)
            global_var = self.global_var.view(1, -1, 1, 1)

            st_dist = ((b_mean.view(1, -1, 1, 1) - global_mean) ** 2).mean(1)[None] + ((b_var.sqrt().view(1, -1, 1, 1) - global_var.sqrt()) ** 2).mean(1)[None]
            self.norm_loss = st_dist.mean()
        else:
            new_mean, new_var = self.global_mean.view(1, -1, 1, 1), self.global_var.view(1, -1, 1, 1)

        x = (x - new_mean) / torch.sqrt(new_var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return x * weight + bias
