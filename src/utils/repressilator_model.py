from typing import Callable

import numpy as np
import torch
from torch import nn

from dynamical_models import DynamicalModel

# TODO: share NeuralDynamics, rbf_activation with lotka_volterra_model.py


class RepressilatorModel(DynamicalModel):
    _NUM_VARIABLES = 3
    _VARIABLE_NAMES = ['Gene 1', 'Gene 2', 'Gene 3']
    _PARAM_NAMES = ['beta', 'n']
    _DEFAULT_PARAM_VALUES = np.array([10.0, 3])
    _DEFAULT_X0 = np.array([1.0, 1.0, 1.2])
    _DEFAULT_T_STEP = 0.2
    _DEFAULT_T = np.arange(0.0, 10.0 + 1e-8, _DEFAULT_T_STEP)

    def __init__(self, param_values: np.ndarray | None = None,
                 x0: np.ndarray | None = None, t: np.ndarray | None = None
                 ) -> None:
        """The repressilator model from Elowitz and Leibler (2000), protein-
        only version.

        See here for more details:
        http://be150.caltech.edu/2020/content/lessons/08_repressilator.html

        Args:
            param_values (np.ndarray | None, optional): values of model
                parameters, namely beta and n.  Defaults parameter values are
                beta = 10.0, n = 3.
        """
        super().__init__(param_values, x0, t)

    @property
    def equations(self) -> Callable:
        def _f(t, x, p=self._param_values):
            dx = np.empty(3)

            dx[0] = p[0] / (1 + x[2] ** p[1]) - x[0]
            dx[1] = p[0] / (1 + x[0] ** p[1]) - x[1]
            dx[2] = p[0] / (1 + x[1] ** p[1]) - x[2]

            return dx

        return _f


class NeuralDynamics(nn.Module):
    def __init__(self, num_hidden_neurons: list[int], activation: Callable):
        """Neural network for the latent derivatives.

        Args:
            num_neurons: number of neurons in each hidden layer.
            activation: activation function to use between layers.
        """
        super().__init__()

        self.activation = activation
        self.module_list = nn.ModuleList()
        # input layer
        self.module_list.append(nn.Linear(3, num_hidden_neurons[0]))
        # hidden layers
        for i in range(len(num_hidden_neurons) - 1):
            self.module_list.append(
                nn.Linear(num_hidden_neurons[i], num_hidden_neurons[i + 1]))
        # output layer
        self.module_list.append(nn.Linear(num_hidden_neurons[-1], 3))

    def forward(self, t, x):
        dx = self.module_list[0](x)

        for module in self.module_list[1:]:
            dx = self.activation(dx)
            dx = module(dx)

        return dx


class HybridDynamics(nn.Module):
    """PyTorch module combining the known and latent derivatives."""
    def __init__(self, latent_dynamics):
        super().__init__()

        self.latent = latent_dynamics

    def forward(self, t, x):
        latent_dx = self.latent(t, x)

        return latent_dx - x


def rbf_activation(x):
    """Radial basis function activation."""
    return torch.exp(-x * x)


def get_hybrid_dynamics(
        num_hidden_neurons: list[int] | None = None, activation: str = 'tanh',
        compile_model: bool = False) -> nn.Module:
    """Return the hybrid dynamics for the repressilator model.

    Args:
        num_hidden_neurons (list[int]): number of neurons in each hidden layer
            of the latent dynamics. If None, then this will be set to [8, 8].
            Default is None.
        activation (str): activation function to use in the latent dynamics.
            Can be either `rbf` for custom radial basis function, or any of the
            activation functions in `torch.nn.functional`. Default is `tanh`.
        compile (bool): whether to compile the model. Default is False.

    Returns:
        nn.Module: hybrid dynamics for the repressilator model.
    """
    if activation == 'rbf':
        activation_func = rbf_activation
    else:
        activation_func = getattr(nn.functional, activation)

    if num_hidden_neurons is None:
        num_hidden_neurons = [8, 8]
    latent_dynamics = NeuralDynamics(num_hidden_neurons, activation_func)
    hybrid_dynamics = HybridDynamics(latent_dynamics)

    if compile_model:
        hybrid_dynamics = torch.compile(hybrid_dynamics)

    return hybrid_dynamics
