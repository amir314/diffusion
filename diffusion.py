from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


class Diffusion:
    data: DataLoader
    model: nn.Module
    optimizer: torch.optim.Optimizer
    int_beta: Callable[[torch.Tensor], torch.Tensor]
    T: torch.Tensor

    def __init__(self,
                 data,
                 model,
                 optimizer,
                 int_beta,
                 T) -> None:
        self.data = data
        self.model = model
        self.optim = optimizer
        self.int_beta = int_beta
        self.T = T

    def batch_loss(self,
                   batch) -> torch.Tensor:
        batch_size = batch.shape[0]
        # sample t
        t = self.T * torch.rand((batch_size,))
        # sample y(t) from dy(s) = -1/2 beta(s) * y(s) dt + sqrt(beta(s)) dw(s)
        mean = batch * torch.exp(-0.5 * self.int_beta(t))
        var = torch.max(1 - torch.exp(-self.int_beta(t)), 1e-5) # lower bound the variance
        std = torch.sqrt(var)
        noise = torch.randn(batch.shape)
        y_t = mean + std * noise
        pred = self.model(t, y_t)
        return torch.mean(((pred + noise) / std) ** 2)
