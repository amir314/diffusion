from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from sdelib.sde import SDE


class Diffusion:
    data: DataLoader
    model: nn.Module
    optimizer: torch.optim.Optimizer
    sde: SDE
    weight: Callable[[torch.Tensor], torch.Tensor]
    T: torch.Tensor

    def __init__(self,
                 data,
                 model,
                 optimizer,
                 sde,
                 weight,
                 T) -> None:
        self.data = data
        self.model = model
        self.optim = optimizer
        self.sde = sde
        self.weight = weight
        self.T = T

    def batch_loss(self,
                   batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.shape[0]
        # sample t
        t = self.T * torch.rand((batch_size,))
        # sample from the sde
        mean, std = self.sde.marginal_params(batch, t)
        noise = torch.randn(batch.shape)
        y_t = mean + std * noise
        # compute prediction and weighted loss
        pred = self.model(t, y_t)
        loss_per_batch = (((pred + noise) / std) ** 2).view(batch_size, -1).mean(1) # shape (batch_size,)
        return ( self.weight(t) * loss_per_batch ).mean()

    def train(self, n_epochs: int) -> None:
        for epoch in range(n_epochs):
            for batch in self.data():
                loss = self.batch_loss(batch)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            print(f"Loss after {epoch+1}/{n_epochs} epochs: ", loss)
