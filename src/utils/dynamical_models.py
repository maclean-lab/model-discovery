from abc import ABCMeta, abstractmethod
from typing import Callable, Literal
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn

from time_series_data import TimeSeries


class DynamicalModel(metaclass=ABCMeta):
    _NUM_VARIABLES = 0
    _VARIABLE_NAMES = []
    _PARAM_NAMES = []
    _DEFAULT_PARAM_VALUES = np.empty(0)
    _DEFAULT_X0 = np.empty(0)
    _DEFAULT_T_STEP = None
    _DEFAULT_T = np.empty(0)

    def __init__(self, param_values: np.ndarray | None = None,
                 x0: np.ndarray | None = None,
                 t: np.ndarray | None = None) -> None:
        """Abstract base class for dynamical models with an underlying system
        of ordinary differential equations.

        To implement an actual class of dynamical models, the following
        constants should be defined:
            - `_NUM_VARIABLES`: number of variables in the model
            - `_VARIABLE_NAMES`: names of the variables
            - `_PARAM_NAMES`: names of the parameters
            - `_DEFAULT_PARAM_VALUES`: default values of the parameters
            - `_DEFAULT_X0`: default initial conditions
            - `_DEFAULT_T_STEP`: default step size of time points
            - `_DEFAULT_T`: default time points
        In addition, if the governing ODE system is known:
            - Set has_equations to True
            - Implement the equations property, which is right-hand side of the
              ODE system and returns a function of the form
              `f(t, x, p=self._param_values)`. The function is to be used by
              scipy.integrate.solve_ivp for simulating the model
        The following constants can be optionally defined:
            - _DEFAULT_T_STEP: default step size of time points, which is
                needed only when _DEFAULT_T has uniform step size


        Args:
            param_values (np.ndarray | None, optional): model parameters.
                Defaults to None, in which case the default parameter values
                are used.
            x0 (np.ndarray | None, optional): initial conditions. Defaults to
                None, in which case the default initial conditions are used.
            t (np.ndarray | None, optional): time points. Defaults to None, in
                which case the default time points are used.
        """
        # set parameter values
        if param_values is None:
            self._param_values = self._DEFAULT_PARAM_VALUES.copy()
        else:
            self.__check_params(param_values)
            self._param_values = param_values.copy()

        # set default parameter names
        if not self._PARAM_NAMES:
            self._PARAM_NAMES = [
                f'param_{i}' for i in range(len(self._param_values))]

        # set initial conditions
        if x0 is None:
            self._x0 = self._DEFAULT_X0.copy()
        else:
            self.__check_x0(x0)
            self._x0 = x0.copy()

        # set default variable names
        if not self._VARIABLE_NAMES:
            self._VARIABLE_NAMES = [
                f'x_{i}' for i in range(self._NUM_VARIABLES)]

        # set time points
        if t is None:
            self._t = self._DEFAULT_T.copy()
            self._t_step = self._DEFAULT_T_STEP
        else:
            self.__check_t(t)
            self._t = t.copy()

            # set self._t_step if t has uniform step size
            if self._t.size == 1:
                self._t_step = None
            elif self._t.size == 2:
                self._t_step = self._t[1] - self._t[0]
            elif self._t.size > 2:
                t_diff = self._t[1:] - self._t[:-1]
                if np.all(t_diff == t_diff[0]):
                    self._t_step = t_diff[0]

        self._has_equations = False

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

    def __check_t(self, t: np.ndarray) -> None:
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
                raise ValueError('t must be monotonically increasing')

    @property
    def num_variables(self) -> int:
        return self._NUM_VARIABLES

    @property
    def variable_names(self) -> list[str]:
        return self._VARIABLE_NAMES.copy()

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
    def has_equations(self) -> bool:
        return self._has_equations

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

        For a model with no known ODE system, raise an error as follows:
        ```
        def equations(self):
            raise NotImplementedError(
                'model does not have governing equations')
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

    @property
    def t_step(self) -> float | None:
        return self._t_step

    @classmethod
    def get_num_variables(cls) -> int:
        return cls._NUM_VARIABLES

    @classmethod
    def get_variable_names(cls) -> list[str]:
        return cls._VARIABLE_NAMES.copy()

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

    @classmethod
    def get_default_t_step(cls) -> float | None:
        return cls._DEFAULT_T_STEP

    def simulate(self, t_span: tuple[float, float], x0: np.ndarray,
                 **kwargs) -> TimeSeries:
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

    def get_samples(
        self,
        num_samples: int,
        bounds: list[tuple[float, float]] | None = None,
        noise_type: Literal['additive', 'multiplicative'] = 'additive',
        noise_level: float = 0.0,
        clean_x0: bool = False,
        integrator_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[TimeSeries]:
        """Generate time series samples from the model.

        Can generate noiseless time series if noise_level is set to 0.0.

        Args:
            num_samples (int): the number of samples (i.e. number of
                multivariate time series) to generate.
            bounds (list[tuple[float, float]] | None, optional): lower and
                upper bounds of generated values for each variable. Defaults to
                None, in which case no bounds are imposed.
            noise_type (Literal['additive', 'multiplicative'], optional): the
                type of noise to add to the data. Defaults to 'additive'.
            noise_level (float, optional): the level of noise. For additive
                noise, the standard deviation of the noise for a variable is
                the same for all time points. For multiplicative noise, the
                standard deviation of the noise is proportional to the value of
                the variable at each time point. Defaults to 0.0, in which case
                no noise is added.
            clean_x0 (bool, optional): whether to use noise-free values for
                initial conditions. Defaults to False.
            rng (Optional[np.random.Generator], optional): random number
                generator. Defaults to None.

        Returns:
            list[TimeSeries]: a list of noisy time series generated from the
                model (clean if noise_level is set 0.0).

        Raises:
            NotImplementedError: if the governing equations of the model are
                not implemented.
            ValueError: if bounds has a different length than the number of
                variables.
            ValueError: if lower bound of a variable is greater than or equal
                to its upper bound.
        """
        if not self._has_equations:
            raise NotImplementedError(
                'No governing equations defined for the model')

        # check lower and upper bounds
        if bounds is None:
            bounds = [(None, None)] * self._NUM_VARIABLES
        elif len(bounds) != self._NUM_VARIABLES:
            raise ValueError('bounds must have the same length as the number '
                             'of variables')
        else:
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None and ub is not None and lb >= ub:
                    msg = f'lower bound of variable {i} must be smaller than '
                    msg += 'its upper bound'
                    raise ValueError(msg)

                processed_bounds = [None, None]
                if lb is not None and np.isfinite(lb):
                    processed_bounds[0] = lb
                if ub is not None and np.isfinite(ub):
                    processed_bounds[1] = ub
                bounds[i] = tuple(processed_bounds)

        # simulate clean data from known ODEs
        if integrator_kwargs is None:
            integrator_kwargs = {}
        integrator_kwargs['t_eval'] = self._t
        if 'method' not in integrator_kwargs:
            integrator_kwargs['method'] = 'LSODA'
        t_span = (self._t[0], self._t[-1])
        x = self.simulate(t_span, self._x0, **integrator_kwargs).x

        # get noise scale from noise type and noise level
        match noise_type:
            case 'additive':
                # noise_scale.shape = (num_vars, )
                noise_scale = noise_level * np.mean(x, axis=0)
            case 'multiplicative':
                # noise_scale.shape = x.shape = (t.size, num_vars)
                noise_scale = noise_level * x

        # initialize random number generator
        if rng is None:
            rng = np.random.default_rng()

        # generate samples
        ts_samples = []
        for _ in range(num_samples):
            if noise_level > 0:
                # generate a noisy observation
                x_obs = x + rng.normal(size=x.shape) * noise_scale

                if clean_x0:
                    # recover x0
                    x_obs[0, :] = x[0, :]
            else:
                # use the clean data as observation
                x_obs = x

            # clip values to bounds
            for i, (lb, ub) in enumerate(bounds):
                if lb is not None or ub is not None:
                    x_obs[:, i] = np.clip(x_obs[:, i], lb, ub)

            # add observation to samples
            ts_samples.append(TimeSeries(self._t, x_obs))

        return ts_samples


class NeuralDynamics(nn.Module):
    def __init__(self, num_vars: int, num_hidden_neurons: list[int],
                 activation: Callable):
        """Neural network for the latent derivatives.

        Args:
            num_vars: number of variables in the dynamical model.
            num_neurons: number of neurons in each hidden layer.
            activation: activation function to use between layers.
        """
        super().__init__()

        self.activation = activation
        self.module_list = nn.ModuleList()
        # input layer
        self.module_list.append(nn.Linear(num_vars, num_hidden_neurons[0]))
        # hidden layers
        for i in range(len(num_hidden_neurons) - 1):
            self.module_list.append(
                nn.Linear(num_hidden_neurons[i], num_hidden_neurons[i + 1]))
        # output layer
        self.module_list.append(nn.Linear(num_hidden_neurons[-1], num_vars))

    def forward(self, t, x):
        dx = self.module_list[0](x)

        for module in self.module_list[1:]:
            dx = self.activation(dx)
            dx = module(dx)

        return dx


def rbf_activation(x):
    """Radial basis function activation."""
    return torch.exp(-x * x)
