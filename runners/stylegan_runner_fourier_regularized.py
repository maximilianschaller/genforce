# python3.7
"""Contains the runner for StyleGAN."""

from copy import deepcopy

from .stylegan_runner import BaseGANRunner
from utils.inverter import StyleGANInverter

__all__ = ['FourierRegularizedStyleGANRunner']


class FourierRegularizedStyleGANRunner(BaseGANRunner):
    """Defines the runner for StyleGAN."""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.lod = getattr(self, 'lod', None)
        self.inverter = StyleGANInverter(self.models['generator'], "styleganinv_ffhq256", learning_rate=0.01,
                                         iteration=500, reconstruction_loss_weight=1.0, perceptual_loss_weight=5e-5,
                                         regularization_loss_weight=2.0, logger=self.logger)
    def build_models(self):
        super().build_models()
        self.g_smooth_img = self.config.modules['generator'].get(
            'g_smooth_img', 10000)
        #self.models['generator_smooth'] = deepcopy(self.models['generator'])

    def build_loss(self):
        super().build_loss()
        self.running_stats.add(
            f'Gs_beta', log_format='.4f', log_strategy='CURRENT')

    def train_step(self, data, **train_kwargs):
        # Set level-of-details.
        G = self.get_module(self.models['generator'])
        D = self.get_module(self.models['discriminator'])
        #Gs = self.get_module(self.models['generator_smooth'])
        D.lod.data.fill_(self.lod)
        #G.net.lod.data.fill_(self.lod)

        #Gs.synthesis.lod.data.fill_(self.lod)

        """
        # Update discriminator.
        self.set_model_requires_grad('discriminator', True)
        self.set_model_requires_grad('generator', False)

        d_loss = self.loss.d_loss(self, data)
        self.optimizers['discriminator'].zero_grad()
        d_loss.backward()
        self.optimizers['discriminator'].step()
        """

        # Life-long update for generator.
        beta = 0.5 ** (self.batch_size * self.world_size / self.g_smooth_img)
        self.running_stats.update({'Gs_beta': beta})
        """
        self.moving_average_model(model=self.models['generator'],
                                  avg_model=self.models['generator_smooth'],
                                  beta=beta)
        """

        # Update generator.
        if self._iter % self.config.get('D_repeats', 1) == 0:
            self.set_model_requires_grad('discriminator', False)
            self.set_model_requires_grad('generator', True)

            g_loss = self.loss.g_loss(self, data)
            self.optimizers['generator'].zero_grad()
            g_loss.backward()
            self.optimizers['generator'].step()

    def load(self, **kwargs):
        super().load(**kwargs)
        G = self.get_module(self.models['generator'])
        D = self.get_module(self.models['discriminator'])
        #Gs = self.get_module(self.models['generator_smooth'])
        if kwargs['running_metadata']:
            lod = G.net.synthesis.lod.cpu().tolist()
            assert lod == D.lod.cpu().tolist()
            #assert lod == Gs.synthesis.lod.cpu().tolist()
            self.lod = lod
