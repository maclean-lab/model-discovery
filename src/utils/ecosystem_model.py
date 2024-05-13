from typing import Callable

import numpy as np

from dynamical_models import DynamicalModel


class EcosystemModel(DynamicalModel):
    _NUM_VARIABLES = 3
    _PARAM_NAMES = ['alpha_1', 'alpha_2', 'delta_1', 'delta_2', 'delta_3']
    _DEFAULT_PARAM_VALUES = np.array([0.2, 0.3, 0.05, 0.09, 0.1])
    _DEFAULT_X0 = np.array([20.0, 20.0, 10.0])
    _DEFAULT_T = np.arange(0.0, 20.0 + 1e-8, 0.5)

    def __init__(self, param_values: np.ndarray | None = None,
                 x0: np.ndarray | None = None, t: np.ndarray | None = None
                 ) -> None:
        """A three-species ecosystem model.

        Model parameters are: alpha_1, alpha_2, delta_1, delta_2, delta_3.

        Args:
            param_values (np.ndarray | None, optional): values of model
                parameters of the ecosystem model. Defaults to None, in which
                case is set to [0.2, 0.3, 0.05, 0.09, 0.1].
            x0 (np.ndarray | None, optional): initial conditions. Defaults to
                None, in which it is set to [20.0, 20.0, 10.0].
            t (np.ndarray | None, optional): time points. Defaults to None, in
                which case it is set to an array on [0, 20] with step size 0.5.
        """
        super().__init__(param_values, x0, t)

    @property
    def equations(self) -> Callable:
        def _f(t, x, p=self._param_values):
            dx = np.empty(3)

            dx[0] = p[0] * x[0] * (1 - 0.01 * x[0] - 0.05 * x[1]) - p[2] * x[0]
            dx[1] = p[1] * x[0] * (1 - 0.08 * x[2]) - p[3] * x[1]
            dx[2] = p[3] * x[1] - p[4] * x[2]

            return dx

        return _f


class EcosystemModelAlt(DynamicalModel):
    # An alternative ecosystem model
    _NUM_VARIABLES = 3
    _PARAM_NAMES = ['alpha_1', 'alpha_2', 'delta_1', 'delta_2', 'delta_3']
    _DEFAULT_PARAM_VALUES = np.array([0.2, 0.3, 0.2, 0.3, 0.1])
    _DEFAULT_X0 = np.array([20.0, 20.0, 10.0])
    _DEFAULT_T = np.arange(0.0, 20.0 + 1e-8, 0.5)

    def __init__(self, param_values: np.ndarray | None = None,
                 x0: np.ndarray | None = None, t: np.ndarray | None = None
                 ) -> None:
        """An alternative three-species ecosystem model.

        Model parameters are: alpha_1, alpha_2, delta_1, delta_2, delta_3.
        Note: for now, some species will have negative population values using
        the default parameter values.

        Args:
            param_values (np.ndarray | None, optional): values of model
                parameters, namely alpha_1, alpha_2, delta_1, delta_2, delta_3.
                Defaults to None, in which case it is set to
                [0.2, 0.3, 0.2, 0.3, 0.1].
            x0 (np.ndarray | None, optional): initial conditions. Defaults to
                None, in which it is set to [20.0, 20.0, 10.0].
            t (np.ndarray | None, optional): time points. Defaults to None, in
                which case it is set to an array on [0, 20] with step size 0.5.
        """
        super().__init__(param_values, x0, t)

    @property
    def equations(self) -> Callable:
        def _f(t, x, p=self._param_values):
            dx = np.empty(3)

            dx[0] = p[0] * x[0] * (1 - x[0] - x[1]) - p[2] * x[0]
            dx[1] = p[1] * x[0] * (1 - x[2]) - p[3] * x[1]
            dx[2] = p[3] * x[1] - p[4] * x[2]

            return dx

        return _f
