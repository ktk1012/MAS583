import torch
import math
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import (
    ScaleKernel,
    RBFKernel)
from gpytorch.likelihoods import GaussianLikelihood
from anomalous_detector.gpns_detector import GpnsDetector


def main():
    x_data = torch.linspace(0, 1, 100)
    y_data = torch.sin(3.7 * x_data * (2 * math.pi)) + torch.randn(x_data.size()) * 0.1

    for i in range(5):
        y_data[i + 35] = y_data[i + 35] + 0.5

    myDetector = GpnsDetector(
        ConstantMean(),
        ScaleKernel(RBFKernel()),
        GaussianLikelihood(),
        x_data,
        y_data)

    optimizer = torch.optim.Adam
    optimizer_kwargs = {'lr': 0.1}
    mll = gpytorch.mlls.ExactMarginalLogLikelihood

    myDetector.train(5000, 10, mll, optimizer, **optimizer_kwargs)


if __name__ == '__main__':
    main()
