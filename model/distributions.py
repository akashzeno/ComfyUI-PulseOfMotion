import torch
import numpy as np


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self, noise=None):
        if noise is None:
            noise = torch.randn(self.mean.shape)
        x = self.mean + self.std * noise.to(device=self.parameters.device)
        return x

    def mode(self):
        return self.mean
