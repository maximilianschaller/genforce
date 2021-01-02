# python3.7
"""Collects all runners."""

from .stylegan_runner import StyleGANRunner
from .stylegan_runner_fourier_regularized import FourierRegularizedStyleGANRunner

__all__ = ['StyleGANRunner']
