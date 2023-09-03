from typing import Callable

import numpy as np
import torch
from torch import nn

from dynamical_models import DynamicalModel


class LotkaVolterraModel(DynamicalModel):
    _NUM_VARIABLES = 2
    _PARAM_NAMES = ['alpha', 'beta', 'gamma', 'delta']
    _DEFAULT_PARAM_VALUES = np.array([1.3, 0.9, 0.8, 1.8])
    _DEFAULT_X0 = np.array([0.44249296, 4.6280594])
    _DEFAULT_T = np.arange(0.0, 4.0 + 1e-8, 0.1)

    def __init__(self, param_values: np.ndarray | None = None,
                 x0: np.ndarray | None = None, t: np.ndarray | None = None
                 ) -> None:
        """The Lotka-Volterra model.

        Args:
            param_values (np.ndarray | None, optional): values of model
                parameters, namely alpha, beta, gamma, delta. Defaults to
                None, in which case it is set to [1.3, 0.9, 0.8, 1.8].
            x0 (np.ndarray | None, optional): initial conditions. Defaults to
                None, in which case it is set to [0.44249296, 4.6280594].
            t (np.ndarray | None, optional): time points. Defaults to None, in
                which case will be set to an array on [0, 4] with step size
                0.1.
        """
        super().__init__(param_values, x0, t)

    @property
    def equations(self) -> Callable:
        def _f(t, x, p=self._param_values):
            dx = np.empty(2)

            dx[0] = p[0] * x[0] - p[1] * x[0] * x[1]
            dx[1] = p[2] * x[0] * x[1] - p[3] * x[1]

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
        self.module_list.append(nn.Linear(2, num_hidden_neurons[0]))
        # hidden layers
        for i in range(len(num_hidden_neurons) - 1):
            self.module_list.append(
                nn.Linear(num_hidden_neurons[i], num_hidden_neurons[i + 1]))
        # output layer
        self.module_list.append(nn.Linear(num_hidden_neurons[-1], 2))

    def forward(self, t, x):
        dx = self.module_list[0](x)

        for module in self.module_list[1:]:
            dx = self.activation(dx)
            dx = module(dx)

        return dx


class HybridDynamics(nn.Module):
    """PyTorch module combining the known and latent derivatives."""
    def __init__(self, growth_rates, latent_dynamics, dtype=torch.float32):
        super().__init__()

        self.growth_rates = torch.tensor(growth_rates, dtype=dtype)
        self.latent = latent_dynamics

    def forward(self, t, x):
        latent_dx = self.latent(t, x)
        known_dx = self.growth_rates * x

        return known_dx + latent_dx


def rbf_activation(x):
    """Radial basis function activation."""
    return torch.exp(-x * x)


def get_hybrid_dynamics(
        growth_rates: np.ndarray,
        num_hidden_neurons: list[int] | None = None,
        activation: str = 'tanh'
        ) -> nn.Module:
    """Return the hybrid dynamics for the Lotka-Volterra model.

    Args:
        growth_rates (np.ndarray): growth rates of the Lotka-Volterra model.
            Expect 2 elements.
        num_hidden_neurons (list[int]): number of neurons in each hidden layer
            of the latent dynamics. If None, then this will be set to [5, 5].
            Default is None.
        activation (str): activation function to use in the latent dynamics.
            Can be either `rbf` for custom radial basis function, or any of the
            activation functions in `torch.nn.functional`. Default is `tanh`.

    Returns:
        nn.Module: hybrid dynamics for the Lotka-Volterra model.
    """
    if activation == 'rbf':
        activation_func = rbf_activation
    else:
        activation_func = getattr(nn.functional, activation)

    if num_hidden_neurons is None:
        num_hidden_neurons = [5, 5]
    latent_dynamics = NeuralDynamics(num_hidden_neurons, activation_func)
    hybrid_dynamics = HybridDynamics(growth_rates, latent_dynamics)

    return hybrid_dynamics
