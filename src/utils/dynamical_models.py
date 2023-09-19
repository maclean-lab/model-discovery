from abc import ABCMeta, abstractmethod
from typing import Callable, Optional
import numpy as np
from scipy.integrate import solve_ivp

from time_series_data import TimeSeries

# TODO: move all model classes to dedicated modules


class DynamicalModel(metaclass=ABCMeta):
    _NUM_VARIABLES = 0
    _PARAM_NAMES = []
    _DEFAULT_PARAM_VALUES = np.empty(0)
    _DEFAULT_X0 = np.empty(0)
    _DEFAULT_T = np.empty(0)

    def __init__(self, param_values: np.ndarray | None = None,
                 x0: np.ndarray | None = None, t: np.ndarray | None = None
                 ) -> None:
        """Abstract base class for dynamical models with an underlying system
        of ordinary differential equations.

        To implement an actual class of dynamical models, the following
        constants should be defined:
            - _NUM_VARIABLES: number of variables in the model
            - _PARAM_NAMES: names of the parameters
            - _DEFAULT_PARAM_VALUES: default values of the parameters
            - _DEFAULT_X0: default initial conditions
            - _DEFAULT_T: default time points
        In addition, the following method should be implemented:
            - equations: right-hand side of the ODE system, which has the form
              `f(t, x, p=self._param_values)` and can be used by
              scipy.integrate.solve_ivp to simulate the model

        Args:
            param_values (np.ndarray | None, optional): model parameters.
                Defaults to None, in which case the default parameter values
                are used.
            x0 (np.ndarray | None, optional): initial conditions. Defaults to
                None, in which case the default initial conditions are used.
            t (np.ndarray | None, optional): time points. Defaults to None, in
                which case the default time points are used.
        """
        if param_values is None:
            self._param_values = self._DEFAULT_PARAM_VALUES.copy()
        else:
            self.__check_params(param_values)
            self._param_values = param_values.copy()

        if x0 is None:
            self._x0 = self._DEFAULT_X0.copy()
        else:
            self.__check_x0(x0)
            self._x0 = x0.copy()

        if t is None:
            self._t = self._DEFAULT_T.copy()
        else:
            self.__check_t(t)
            self._t = t.copy()

    def __check_params(self, param_values: np.ndarray):
        """Check if the input parameters are valid.

        Args:
            param_values (np.ndarray): values of model parameters.

        Raises:
            ValueError: if the number of input parameters does not match the
                number of model parameters.
        """
        num_params = self._DEFAULT_PARAM_VALUES.size
        num_input_params = param_values.size

        if num_input_params != num_params:
            msg = f'expected {num_params} model parameters, '
            msg += f'got {num_input_params} instead'

            raise ValueError(msg)

    def __check_x0(self, x0: np.ndarray):
        """Check if the input initial conditions are valid.

        Args:
            x0 (np.ndarray): initial conditions.

        Raises:
            ValueError: if the number of input initial conditions does not
                match the number of model variables.
        """
        num_input_x0 = x0.size

        if num_input_x0 != self._NUM_VARIABLES:
            msg = f'expected x0 of length {self._NUM_VARIABLES}, '
            msg += f'got {num_input_x0} instead'

            raise ValueError(msg)

    def __check_t(self, t: np.ndarray):
        """Check if the input time points are valid.

        Args:
            t (np.ndarray): time points.

        Raises:
            ValueError: if the input time points are not a 1-dimensional array
                or are not monotonically increasing.
        """
        if t.ndim != 1:
            raise ValueError('t must be a 1-dimensional array')

        for i in range(t.size - 1):
            if t[i] >= t[i + 1]:
                raise ValueError('t must be a monotonically increasing array')

    @property
    def num_variables(self) -> int:
        return self._NUM_VARIABLES

    @property
    def param_names(self) -> list[str]:
        return self._PARAM_NAMES.copy()

    @property
    def param_values(self) -> np.ndarray:
        return self._param_values.copy()

    @param_values.setter
    def param_values(self, param_values: np.ndarray) -> None:
        self.__check_params(param_values)
        self._param_values = param_values.copy()

    @property
    @abstractmethod
    def equations(self) -> Callable:
        """Right-hand side of the governing ordinary differential equations.

        This method should implement and return an inner method of the form
        `f(t, x, p=self._param_values)` where `t` is the time, `x` is the state
        variable, and `p` is the parameter vector. For example,
        ```
        def _f(t, x, p=self._param_values):
            dx = np.zeros(self._NUM_VARIABLES)

            # compute dx here

            return dx
        ```
        """
        pass

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @x0.setter
    def x0(self, x0: np.ndarray) -> None:
        self.__check_x0(x0)
        self._x0 = x0.copy()

    @property
    def t(self) -> np.ndarray:
        return self._t.copy()

    @t.setter
    def t(self, t: np.ndarray) -> None:
        self.__check_t(t)
        self._t = t.copy()

    @property
    def t_span(self) -> tuple[float, float]:
        return (self._t[0], self._t[-1])

    @classmethod
    def get_num_variables(cls) -> int:
        return cls._NUM_VARIABLES

    @classmethod
    def get_param_names(cls) -> list[str]:
        return cls._PARAM_NAMES.copy()

    @classmethod
    def get_default_param_values(cls) -> np.ndarray:
        return cls._DEFAULT_PARAM_VALUES.copy()

    @classmethod
    def get_default_x0(cls) -> np.ndarray:
        return cls._DEFAULT_X0.copy()

    @classmethod
    def get_default_t(cls) -> np.ndarray:
        return cls._DEFAULT_T.copy()

    @classmethod
    def get_default_t_span(cls) -> tuple[float, float]:
        return (cls._DEFAULT_T[0], cls._DEFAULT_T[-1])

    def simulate(self, t_span: tuple[float, float], x0: np.ndarray, **kwargs
                 ) -> TimeSeries:
        """Simulate the model given time span and initial conditions.

        Args:
            t_span (tuple[float, float]): time span.
            x0 (np.ndarray): initial conditions.
            **kwargs: additional arguments to be passed to
                scipy.integrate.solve_ivp.

        Returns:
            TimeSeries: simulated time series.

        Raises:
            RuntimeError: if scipy.integrate.solve_ivp fails to simulate the
                model with the given time span and initial conditions.
        """
        if self._param_values.size > 0:
            kwargs['args'] = (self._param_values, )
        solution = solve_ivp(self.equations, t_span, x0, **kwargs)

        # return solved x with shape t.size * num_vars
        if not solution.success:
            raise RuntimeError('unable to simulate the model')

        return TimeSeries(solution.t, solution.y.T)

    def get_sample(self, sample_size: int, noise_level: float = 0.0,
                   rng: Optional[np.random.Generator] = None,
                   ) -> list[TimeSeries]:
        """Generate a sample of noisy time series from the model.

        Can generate noiseless time series if noise_level is set to 0.0.

        Args:
            sample_size (int): the number of time series in the sample.
            noise_level (float, optional): the level of noise. Defaults to 0.0.
            rng (Optional[np.random.Generator], optional): a random number
                generator. Defaults to None.

        Returns:
            list[TimeSeries]: a list of noisy time series generated from the
                model (clean if noise_level is set 0.0).
        """
        # get clean data
        t_span = (self._t[0], self._t[-1])
        x = self.simulate(t_span, self._x0, t_eval=self._t).x

        # TODO: add support of other types of random noise
        ts_sample = []
        noise_scale = noise_level * np.mean(x, axis=0)

        if rng is None:
            rng = np.random.default_rng()

        # generate sample
        for _ in range(sample_size):
            if noise_level > 0:
                # generate a noisy observation
                x_noise = rng.normal(
                    size=(self._t.size - 1, x.shape[1])) * noise_scale
                x_obs = x.copy()
                x_obs[1:, :] += x_noise
            else:
                # use the clean data as observation
                x_obs = x

            ts_sample.append(TimeSeries(self._t, x_obs))

        return ts_sample


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


class RepressilatorModel(DynamicalModel):
    _NUM_VARIABLES = 3
    _PARAM_NAMES = ['beta', 'n']
    _DEFAULT_PARAM_VALUES = np.array([10.0, 3])
    _DEFAULT_X0 = np.array([1.0, 1.0, 1.2])
    _DEFAULT_T = np.arange(0.0, 10.0 + 1e-8, 0.2)

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
