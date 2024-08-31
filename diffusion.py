from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


class Diffusion:
    data: DataLoader
    model: nn.Module
    optimizer: torch.optim.Optimizer
    int_beta: Callable[[torch.Tensor], torch.Tensor]
    weight: Callable[[torch.Tensor], torch.Tensor]
    T: torch.Tensor

    def __init__(self,
                 data,
                 model,
                 optimizer,
                 int_beta,
                 weight,
                 T) -> None:
        self.data = data
        self.model = model
        self.optim = optimizer
        self.int_beta = int_beta
        self.weight = weight
        self.T = T

    def batch_loss(self,
                   batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.shape[0]
        # sample t
        t = self.T * torch.rand((batch_size,))
        # sample y(t) from dy(s) = -1/2 beta(s) * y(s) dt + sqrt(beta(s)) dw(s)
        mean = batch * torch.exp(-0.5 * self.int_beta(t))
        var = torch.max(1 - torch.exp(-self.int_beta(t)), 1e-5) # lower bound the variance
        std = torch.sqrt(var)
        noise = torch.randn(batch.shape)
        y_t = mean + std * noise
        # weighted loss across all losses in the batch
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
