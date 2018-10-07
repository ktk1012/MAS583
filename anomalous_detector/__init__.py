import gpytorch
from gpytorch.distributions.multivariate_normal import MultivariateNormal


MODE_CPU = "CPU"
MODE_CUDA = "CUDA"
MODE_AVAILABLE = (
    MODE_CPU,
    MODE_CUDA
)


class GpModel(gpytorch.models.ExactGP):
    def __init__(
            self,
            mean_module,
            covar_module,
            train_x,
            train_y,
            likelihood):
        super(GpModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


