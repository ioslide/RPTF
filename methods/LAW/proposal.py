import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import torch.optim as optim
from loguru import logger as log
from methods.LAW.transformers_cotta import get_tta_transforms
from methods.PR.transformers_patch import patch_transforms
from collections import defaultdict
from torch.nn.utils.weight_norm import WeightNorm
from copy import deepcopy
import numpy as np

__all__ = ["setup"]


class LAW(nn.Module):
    def __init__(self, cfg, model,optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.model = model
        self.cfg = cfg
        self.steps = cfg.OPTIM.STEPS
        self.base_lr = self.optimizer.param_groups[0]['lr']
        self.betas = self.optimizer.param_groups[0]['betas']
        self.weight_decay = self.optimizer.param_groups[0]['weight_decay']
        self.transforms = get_tta_transforms(cfg.CORRUPTION.DATASET)
        self.patch_transforms = patch_transforms
        self.eps = 1e-8        
        self.grad_weight = defaultdict(lambda: 0.0)
        self.trainable_dict = {k: v for k, v in self.model.named_parameters() if v.requires_grad}
        self.tau = cfg.ADAPTER.LAW.TAU
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
        self.batch_index = 0

    def forward(self, x, **kwargs):
        outputs = self.forward_and_adapt(x)
        return outputs

    def reset(self):
        self.optimizer.load_state_dict(self.optimizer_state)
        self.model.load_state_dict(self.model_state)
        self.grad_weight = defaultdict(lambda: 0.0)
        self.trainable_dict = {k: v for k, v in self.model.named_parameters() if v.requires_grad}

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        1. Get FIM per each parameter with negative log likelihood loss.
        2. Normalize FIM and apply exponential min-max scaling.
        3. Update learning rate.
        4. Update model parameters with corresponding learning rate.
        """
        logits = self.model(x)
        # logits_aug = self.model(self.transforms(x))
        logits_aug = self.model(self.patch_transforms(x,self.cfg.CORRUPTION.DATASET))
        label = logits.max(1)[1].view(-1)
        loss = F.nll_loss(F.log_softmax(logits, dim=1), label)
        loss.backward(retain_graph=True) 

        min_weight, max_weight =  1e8, -1e8
        for np, param in self.trainable_dict.items():
            self.grad_weight[np] += (param.grad**2)
            min_weight = min(min_weight, self.grad_weight[np].mean().item()**0.5)
            max_weight = max(max_weight, self.grad_weight[np].mean().item()**0.5)

        params = []       
        for k, v in self.grad_weight.items():
            value = v.mean().item()**0.5
            lr_weight = (value-min_weight)/(max_weight-min_weight+self.eps) 
            params.append(
                {
                    "params": self.trainable_dict[k],
                    "lr": self.base_lr*(lr_weight**self.tau),
                    "betas": self.betas,
                    "weight_decay": self.weight_decay
                }
            )

        self.optimizer = torch.optim.Adam(params)
        if self.batch_index == 0:
            self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.optimizer.zero_grad()
        loss = softmax_entropy(logits)+0.01*logits.shape[1]*consistency(logits, logits_aug)
        loss.backward()
        self.optimizer.step()
        self.batch_index += 1
        return logits

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()

@torch.jit.script
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """Consistency loss between two softmax distributions."""
    return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()

def configure_model(model):
    model.eval()
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, nn.BatchNorm1d):
            m.train()
            m.requires_grad_(True)
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.Conv2d)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")           
    return params, names

def setup(model, cfg):
    log.info("Setup TTA method: LAW")
    model = configure_model(model)
    params, param_names = collect_params(model)
    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            params, 
            lr=float(cfg.OPTIM.LR),
            dampening=cfg.OPTIM.DAMPENING,
            momentum=float(cfg.OPTIM.MOMENTUM),
            weight_decay=float(cfg.OPTIM.WD),
            nesterov=cfg.OPTIM.NESTEROV
        )
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = optim.Adam(
            params, 
            lr=float(cfg.OPTIM.LR),
            betas=(cfg.OPTIM.BETA, 0.999),
            weight_decay=float(cfg.OPTIM.WD)
        )
    TTA_model = LAW(
        cfg,
        model,
        optimizer
    )
    return TTA_model

