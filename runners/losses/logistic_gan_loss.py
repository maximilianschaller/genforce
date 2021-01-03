# python3.7
"""Defines loss functions."""

import os
import torch
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append(os.getcwd())
from fourier import fourier_dissimilarity

__all__ = ['FourierRegularizedLogisticGANLoss']

apply_loss_scaling = lambda x: x * torch.exp(x * np.log(2.0))
undo_loss_scaling = lambda x: x * torch.exp(-x * np.log(2.0))


class FourierRegularizedLogisticGANLoss(LogisticGANLoss):
    def __init__(self, runner, d_loss_kwargs=None, g_loss_kwargs=None):
        super(FourierRegularizedLogisticGANLoss, self).__init__(runner, d_loss_kwargs, g_loss_kwargs)
        self.lamb = self.g_loss_kwargs['lamb']
        self.metric = self.g_loss_kwargs['metric']
        self.threshold = self.g_loss_kwargs['threshold']

    def g_loss(self, runner, data):
        """Computes loss for generator."""
        # TODO: Use random labels.
        G = runner.models['generator']
        D = runner.models['discriminator']
        labels = data.get('label', None)
        data['image'] = data['image'] / 255. * 2. - 1.
        #latents = runner.inverter.invert(data['image'])
        latents = data['latent']
        G.net.train()
        fakes = G.net.module.synthesis(latents)
        fake_scores = D(fakes, label=labels, **runner.D_kwargs_train)

        g_loss = F.softplus(-fake_scores).mean()
        runner.running_stats.update({'g_loss': g_loss.item()})
        fourier_loss = fourier_dissimilarity(fakes, data['image'], self.metric, self.threshold)
        fourier_loss = torch.mean(fourier_loss)
        runner.running_stats.update({'fourier_loss': fourier_loss.item()})
        total_loss = g_loss + self.lamb * fourier_loss
        runner.running_stats.update({'total_loss': total_loss.item()})
        return g_loss
