# python3.7
"""Collects all loss functions."""

from .logistic_gan_loss import LogisticGANLoss, FourierRegularizedLogisticGANLoss, FourierRegularizedPerceptualLoss

__all__ = ['LogisticGANLoss', 'FourierRegularizedLogisticGANLoss', 'FourierRegularizedPerceptualLoss']
