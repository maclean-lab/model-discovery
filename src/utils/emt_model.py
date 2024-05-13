import os.path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from time_series_data import TimeSeries
from dynamical_models import DynamicalModel, NeuralDynamics, rbf_activation


class EmtModel(DynamicalModel):
    _NUM_VARIABLES = 3
    _VARIABLE_NAMES = ['Epithelial', 'Intermediate', 'Mesenchymal']
    _DEFAULT_X0 = np.array([1.0, 0.0, 0.0])
    _DEFAULT_T_STEP = 1.0
    _DEFAULT_T = np.arange(0.0, 10.0 + 1e-8, _DEFAULT_T_STEP)

    def __init__(self, t: np.ndarray | None = None, **kwargs) -> None:
        """The EMT model.

        Note that no governing equations are defined for this model. Noisy
        samples are generated from mean and standard deviation.

        Args:
            t (np.ndarray | None, optional): time points. Defaults to None, in
                which case will be set to an array on [0, 10] with step size
                1.0.
        """
        # load mean and standard deviation from file
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        self._mean = pd.read_csv(os.path.join(data_dir, 'emt_mean.csv'),
                                 index_col=0).to_numpy()
        self._std = pd.read_csv(os.path.join(data_dir, 'emt_std.csv'),
                                index_col=0).to_numpy()

        super().__init__(t=t)

    def __check_t(self, t: np.ndarray) -> None:
        """Check if the input time points are valid.

        Overrides the parent method.

        Args:
            t (np.ndarray): time points.

        Raises:
            ValueError: if t is not a 1D array.
            ValueError: if t is out of bounds
            ValueError: if t is not monotonically increasing.
            ValueError: if t does not have a step size of 1.0.
        """
        if t.ndim != 1:
            raise ValueError('t must be a 1D array')

        # check if t is out of bounds
        num_time_points = self._mean.shape[0]
        if t[0] < 0 or t[-1] >= num_time_points:
            raise ValueError(
                f't must be in the range [0, {num_time_points - 1}]')

        for i in range(t.size - 1):
            if t[i] >= t[i + 1]:
                raise ValueError('t must bg monotonically increasing')

            if np.abs(t[i + 1] - t[i] - self._DEFAULT_T_STEP) > 1e-8:
                raise ValueError(
                    f't must have a step size of {self._DEFAULT_T_STEP}')

    @property
    def equations(self) -> Callable:
        raise NotImplementedError(
            'EMT model does not have governing equations')

    def get_samples(self, num_samples: int,
                    rng: np.random.Generator | None = None, **kwargs
                    ) -> list[TimeSeries]:
        """Generate time series samples from the EMT model.

        Overrides the parent method since no governing equations are defined.

        Args:
            num_samples (int): the number of samples (i.e. number of
                multivariate time series) to generate.
            rng (np.random.Generator | None, optional): random number
                generator. Defaults to None.

        Returns:
            list[TimeSeries]: list of time series samples.
        """
        if rng is None:
            rng = np.random.default_rng()

        ts_samples = []
        for _ in range(num_samples):
            x = np.zeros((self.t.size, self._NUM_VARIABLES))  # sample

            for t in self.t:
                t = int(t)  # current time point
                i = int(t - self.t[0])  # index of current time point in x

                match t:
                    case 0:
                        x[i, 0] = 1.0
                    case 1 | 2:
                        # small I, use mean+-std for bounds
                        bounds = (self._mean[t, 1] - self._std[t, 1],
                                  self._mean[t, 1] + self._std[t, 1])
                        x[i, 1] = self._draw_truncated_normal(
                            rng, self._mean[t, 1], self._std[t, 1], bounds)
                        x[i, 0] = 1 - x[i, 1]
                    case 3:
                        # E and I are close, use mean+-std to bound E
                        bounds = (self._mean[t, 0] - self._std[t, 0],
                                  self._mean[t, 0] + self._std[t, 0])
                        x[i, 0] = self._draw_truncated_normal(
                            rng, self._mean[t, 0], self._std[t, 0], bounds)
                        # I = 1 - E
                        x[i, 1] = 1 - x[i, 0]
                    case 4 | 5:
                        # small E, use mean+-std for bounds
                        bounds = (self._mean[t, 0] - self._std[t, 0],
                                  self._mean[t, 0] + self._std[t, 0])
                        x[i, 0] = self._draw_truncated_normal(
                            rng, self._mean[t, 0], self._std[t, 0], bounds)
                        # small M, use mean+-std for bounds
                        bounds = (self._mean[t, 2] - self._std[t, 2],
                                  self._mean[t, 2] + self._std[t, 2])
                        x[i, 2] = self._draw_truncated_normal(
                            rng, self._mean[t, 2], self._std[t, 2], bounds)
                        x[i, 1] = 1 - x[i, 0] - x[i, 2]
                    case 6:
                        # small E, use mean+-std for bounds
                        bounds = (self._mean[t, 0] - self._std[t, 0],
                                  self._mean[t, 0] + self._std[t, 0])
                        x[i, 0] = self._draw_truncated_normal(
                            rng, self._mean[t, 0], self._std[t, 0], bounds)
                        # small I, use mean+-std for bounds
                        bounds = (self._mean[t, 1] - self._std[t, 1],
                                  self._mean[t, 1] + self._std[t, 1])
                        x[i, 2] = self._draw_truncated_normal(
                            rng, self._mean[t, 1], self._std[t, 1], bounds)
                        x[i, 2] = 1 - x[i, 0] - x[i, 1]
                    case 7:
                        # small E, use mean+-std for bounds
                        bounds = (self._mean[t, 0] - self._std[t, 0],
                                  self._mean[t, 0] + self._std[t, 0])
                        x[i, 0] = self._draw_truncated_normal(
                            rng, self._mean[t, 0], self._std[t, 0], bounds)
                        # small I, use [0, mean+std] for bounds
                        bounds = (0.0, self._mean[t, 1] + self._std[t, 1])
                        x[i, 1] = self._draw_truncated_normal(
                            rng, self._mean[t, 1], self._std[t, 1], bounds)
                        x[i, 2] = 1 - x[i, 0] - x[i, 1]
                    case 8:
                        # large M, use [mean-std, 1.0] for bounds
                        bounds = (self._mean[t, 2] - self._std[t, 2], 1.0)
                        x[i, 2] = self._draw_truncated_normal(
                            rng, self._mean[t, 2], self._std[t, 2], bounds)
                        x[i, 0] = x[i, 1] = (1 - x[i, 2]) / 2
                    case _ if t >= 9:
                        x[i, 2] = 1.0

            ts_samples.append(TimeSeries(self.t, x))

        return ts_samples

    @staticmethod
    def _draw_truncated_normal(rng: np.random.Generator, mean: float,
                               std: float, bounds: tuple[float, float]
                               ) -> float:
        """Draw a random number from a normal distribution within bounds.

        Args:
            rng (np.random.Generator): random number generator.
            mean (float): mean of the normal distribution.
            std (float): standard deviation of the normal distribution.
            bounds (tuple[float, float]): bounds of the random number.

        Returns:
            float: generated random number.
        """
        x = rng.normal(loc=mean, scale=std)

        while x < bounds[0] or x > bounds[1]:
            x = rng.normal(loc=mean, scale=std)

        return x


def get_neural_dynamics(
        num_hidden_neurons: list[int] | None = None, activation: str = 'tanh',
        compile_model: bool = False) -> nn.Module:
    """Return the neural dynamics for the EMT model.

    Args:
        num_hidden_neurons (list[int]): number of neurons in each hidden layer
            of the latent dynamics. If None, then this will be set to [8].
            Default is None.
        activation (str): activation function to use in the latent dynamics.
            Can be either `rbf` for custom radial basis function, or any of the
            activation functions in `torch.nn.functional`. Default is `tanh`.
        compile (bool): whether to compile the model. Default is False.

    Returns:
        nn.Module: neural dynamics for the EMT model.
    """
    if activation == 'rbf':
        activation_func = rbf_activation
    else:
        activation_func = getattr(nn.functional, activation)

    if num_hidden_neurons is None:
        num_hidden_neurons = [8]

    neural_dynamics = NeuralDynamics(3, num_hidden_neurons, activation_func)

    if compile_model:
        neural_dynamics = torch.compile(neural_dynamics)

    return neural_dynamics


def get_hybrid_dynamics(*args, **kwargs):
    raise NotImplementedError(
        'Hybrid dynamics for the EMT model is not implemented')
