from typing import Callable

import numpy as np
import torch
from torch import nn

from dynamical_models import DynamicalModel
from dynamical_models import NeuralDynamics, rbf_activation


class LotkaVolterraModel(DynamicalModel):
    _NUM_VARIABLES = 2
    _VARIABLE_NAMES = ['Prey', 'Predator']
    _PARAM_NAMES = ['alpha', 'beta', 'gamma', 'delta']
    _DEFAULT_PARAM_VALUES = np.array([1.3, 0.9, 0.8, 1.8])
    _DEFAULT_X0 = np.array([0.44249296, 4.6280594])
    _DEFAULT_T_STEP = 0.1
    _DEFAULT_T = np.arange(0.0, 4.0 + 1e-8, _DEFAULT_T_STEP)

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


def get_neural_dynamics(
        num_hidden_neurons: list[int] | None = None, activation: str = 'tanh',
        compile_model: bool = False) -> nn.Module:
    """Return the neural dynamics for the Lotka-Volterra model.

    Args:
        num_hidden_neurons (list[int]): number of neurons in each hidden layer
            of the latent dynamics. If None, then this will be set to [5, 5].
            Default is None.
        activation (str): activation function to use in the latent dynamics.
            Can be either `rbf` for custom radial basis function, or any of the
            activation functions in `torch.nn.functional`. Default is `tanh`.
        compile (bool): whether to compile the model. Default is False.

    Returns:
        nn.Module: hybrid dynamics for the Lotka-Volterra model.
    """
    if activation == 'rbf':
        activation_func = rbf_activation
    else:
        activation_func = getattr(nn.functional, activation)

    if num_hidden_neurons is None:
        num_hidden_neurons = [5, 5]

    nerual_dynamics = NeuralDynamics(2, num_hidden_neurons, activation_func)

    if compile_model:
        nerual_dynamics = torch.compile(nerual_dynamics)

    return nerual_dynamics


def get_hybrid_dynamics(
        growth_rates: np.ndarray,
        num_hidden_neurons: list[int] | None = None, activation: str = 'tanh',
        compile_model: bool = False, dtype=torch.float32) -> nn.Module:
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
        compile (bool): whether to compile the model. Default is False.
        dtype (torch.dtype): data type of the model. Default is torch.float32.

    Returns:
        nn.Module: hybrid dynamics for the Lotka-Volterra model.
    """
    if activation == 'rbf':
        activation_func = rbf_activation
    else:
        activation_func = getattr(nn.functional, activation)

    if num_hidden_neurons is None:
        num_hidden_neurons = [5, 5]
    latent_dynamics = NeuralDynamics(2, num_hidden_neurons, activation_func)
    hybrid_dynamics = HybridDynamics(growth_rates, latent_dynamics,
                                     dtype=dtype)

    if compile_model:
        hybrid_dynamics = torch.compile(hybrid_dynamics)

    return hybrid_dynamics
