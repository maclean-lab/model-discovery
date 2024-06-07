from typing import Callable

import numpy as np
import torch
from torch import nn

from dynamical_models import DynamicalModel
from dynamical_models import NeuralDynamics, rbf_activation, \
    identity_activation


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
                parameters, namely beta and n. Defaults parameter values are
                beta = 10.0, n = 3.
            x0 (np.ndarray | None, optional): initial conditions. Defaults to
                None, in which case it is set to [1.0, 1.0, 1.2].
            t (np.ndarray | None, optional): time points. Defaults to None, in
                which case will be set to an array on [0, 10] with step size
                0.2.
        """
        super().__init__(param_values=param_values, x0=x0, t=t)
        self._has_equations = True

    @property
    def equations(self) -> Callable:
        def _f(t, x, p=self._param_values):
            dx = np.empty(3)

            dx[0] = p[0] / (1 + x[2] ** p[1]) - x[0]
            dx[1] = p[0] / (1 + x[0] ** p[1]) - x[1]
            dx[2] = p[0] / (1 + x[1] ** p[1]) - x[2]

            return dx

        return _f


class HybridDynamics(nn.Module):
    """PyTorch module combining the known and latent derivatives."""
    def __init__(self, latent_dynamics):
        super().__init__()

        self.latent = latent_dynamics

    def forward(self, t, x):
        latent_dx = self.latent(t, x)

        return latent_dx - x


def get_neural_dynamics(
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
    elif activation == 'identity':
        activation_func = identity_activation
    else:
        activation_func = getattr(nn.functional, activation)

    if num_hidden_neurons is None:
        num_hidden_neurons = [8, 8]
    neural_dynamics = NeuralDynamics(3, num_hidden_neurons, activation_func)

    if compile_model:
        neural_dynamics = torch.compile(neural_dynamics)

    return neural_dynamics


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
    elif activation == 'identity':
        activation_func = identity_activation
    else:
        activation_func = getattr(nn.functional, activation)

    if num_hidden_neurons is None:
        num_hidden_neurons = [8, 8]
    latent_dynamics = NeuralDynamics(3, num_hidden_neurons, activation_func)
    hybrid_dynamics = HybridDynamics(latent_dynamics)

    if compile_model:
        hybrid_dynamics = torch.compile(hybrid_dynamics)

    return hybrid_dynamics
