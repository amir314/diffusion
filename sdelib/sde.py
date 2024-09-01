from typing import Callable, NamedTuple

import torch


class MarginalParameters(NamedTuple):
    mean: torch.Tensor
    std: torch.Tensor


class SDE:
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    g: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self,
                 f,
                 g) -> None:
        self.f = f
        self.g = g

    def marginal_params(self,
                        x: torch.Tensor,
                        t: torch.Tensor) -> MarginalParameters:
        raise NotImplementedError

    def prob_flow_ode(self,
                      score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> 'SDE':
        def flow_f(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.f(x, t) - 1/2 * self.g(t)**2 * score(x, t)
        return SDE(flow_f, 0)
