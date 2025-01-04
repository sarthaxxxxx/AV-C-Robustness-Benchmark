import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast,GradScaler
import math


class Source(nn.Module):
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device


    def forward(self, x):
        for _ in range(self.steps):
            outputs, _ = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode)
            # outputs = self.model.module.forward(a=x[0], v=x[1], mode='multimodal')
            loss = (0, 0)
            outputs = (outputs, outputs)

        return outputs, loss
    

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def collect_params(model):
    """
    Walk the model's modules and collect qkv parameters of the fusion attn module.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    return [], []



def configure_model(model):
    """Configure model for use with Renata."""
    # train mode, but no grad
    model.eval()
    model.requires_grad_(False)

    return model