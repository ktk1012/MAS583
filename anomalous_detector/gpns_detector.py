import torch
import gpytorch
from anomalous_detector import (
    GpModel,
    MODE_CPU,
    MODE_CUDA,
    MODE_AVAILABLE)


class GpnsDetector(object):
    def __init__(
            self,
            mean_module,
            covar_module,
            likelihood,
            train_x,
            train_y,
            mode=MODE_CPU):
        if mode not in MODE_AVAILABLE:
            raise RuntimeError(
                "Unknown mode %s" % mode
            )

        if train_x.size()[0] != train_y.size()[0]:
            raise RuntimeError(
                "Size of target and label is mismatched"
            )

        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        self.mode = mode
        self.model = GpModel(
            mean_module,
            covar_module,
            self.train_x,
            self.train_y,
            self.likelihood)

    # Train hyper parameter
    def train(self, train_iter, verbose_period, mll, optimizer, **kwargs):
        self.model.train()
        self.likelihood.train()

        optimizer_instance = optimizer([
            {'params': self.model.parameters()}
        ], **kwargs)
        mll_instance = mll(self.likelihood, self.model)

        for i in range(train_iter):
            optimizer_instance.zero_grad()
            output = self.model(self.train_x)
            loss = -mll_instance(output, self.train_y)
            loss.backward()
            optimizer_instance.step()

            if (verbose_period is not None and verbose_period > 0) and (i + 1) % verbose_period == 0:
                print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
                    i + 1, train_iter, loss.item(),
                    self.model.covar_module.base_kernel.log_lengthscale.item(),
                    self.model.likelihood.log_noise.item()
                ))

