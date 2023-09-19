import os
import os.path
import sys
from typing import Callable, Iterable, Literal, Any
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader as TorchDataLoader
from torchdiffeq import odeint_adjoint as tdf_odeint
import torchode
import pysindy as ps
from pysindy.feature_library import CustomLibrary
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tqdm

from time_series_data import TimeSeries, align_time_series, get_dataloader

# TODO: log events, e.g. training, evaluating, plotting
# TODO: save and load trained models from logs
# TODO: allow for different extensions for output files (e.g. PDF, PNG)

# alternative colors for plotting time series
_ALT_COLORS = ['#6929c4', '#1192e8', '#005d5d', '#9f1853', '#fa4d56']


class BaseTimeSeriesLearner(metaclass=ABCMeta):
    def __init__(self, train_data: list[TimeSeries],
                 output_dir: str | bytes | os.PathLike, output_prefix: str,
                 *args, **kwargs) -> None:
        """Abstract base class for learning time series data.

        The training data is immutable after initialization. A subclass should
        implement the `train()` method to learn a model (e.g. a neural network)
        from the training data, and the `eval()` method to evaluate the learned
        model on some data.  Each time the `train()` method is called, it may
        store a new model as class attribute `model`, potentially overwriting
        an existing one. Similarly, each time the 'eval()' method is called, it
        may store new evaluation data and prediction data as class attributes
        (`eval_data` and `pred_data` respectively), potentially overwriting
        existing ones.

        Args:
            train_data (list[TimeSeries]): time series data to be learned.
            output_dir (str | bytes | os.PathLike): directory to save outputs.
            output_prefix (str): prefix for output files.
            *args: additional positional arguments that a subclass may use.
            **kwargs: additional keyword arguments that a subclass may use.
        """
        self._train_data = train_data
        self._num_vars = train_data[0].x.shape[1]
        self._output_dir = output_dir
        self._output_prefix = output_prefix
        self._is_trained = False
        self._model: Any | None = None
        self._valid_data: list[TimeSeries] | None = None
        self._valid_metrics: list[dict[str, Any]] | None = None
        self._eval_data: list[TimeSeries] | None = None
        self._pred_data: list[TimeSeries] | None = None
        self._is_evaluated = False
        self._eval_metrics: dict[str, Any] | None = None

    @property
    def train_data(self) -> list[TimeSeries]:
        """list[TimeSeries]: time series data used for training."""
        return [ts.copy() for ts in self._train_data]

    @property
    def num_vars(self) -> int:
        """int: number of variables in time series data."""
        return self._num_vars

    @property
    def output_dir(self) -> str | bytes | os.PathLike:
        """str | bytes | os.PathLike: directory to save outputs."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, new_dir) -> None:
        self._output_dir = new_dir

    @property
    def output_prefix(self) -> str:
        """str: prefix for output files."""
        return self._output_prefix

    @output_prefix.setter
    def output_prefix(self, new_prefix) -> None:
        self._output_prefix = new_prefix

    @property
    def is_trained(self) -> bool:
        """bool: whether a model has been trained from training data."""
        return self._is_trained

    @property
    def model(self) -> Any | None:
        """Any | None: model learned from training data."""
        return self._model

    @property
    def valid_data(self) -> list[TimeSeries] | None:
        """list[TimeSeries] | None: time series data used for model validation
        after training."""
        if self._valid_data is None:
            return None

        return [ts.copy() for ts in self._valid_data]

    @property
    def valid_metrics(self) -> list[dict[str, Any]] | None:
        """list[dict[str, Any]] | None: validation metrics."""
        return self._valid_metrics

    @property
    def eval_data(self) -> list[TimeSeries] | None:
        """list[TimeSeries] | None: time series data used for evaluation."""
        if self._eval_data is None:
            return None

        return [ts.copy() for ts in self._eval_data]

    @property
    def pred_data(self) -> list[TimeSeries] | None:
        """list[TimeSeries] | None: time series data predicted from evaluation
        data."""
        if self._pred_data is None:
            return None

        return [ts.copy() for ts in self._pred_data]

    @property
    def is_evaluated(self) -> bool:
        """bool: whether the model has been evaluated on some data."""
        return self._is_evaluated

    @property
    def eval_metrics(self) -> dict[str, Any] | None:
        """dict[str, Any] | None: evaluation metrics."""
        return self._eval_metrics

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def eval(self, *args, **kwargs) -> None:
        pass

    def _x0_to_eval_data(self, eval_data: list[TimeSeries] | None,
                         t_eval: np.ndarray | list[np.ndarray] | None,
                         x0_eval: np.ndarray | list[np.ndarray] | None
                         ) -> list[TimeSeries] | None:
        """Convert initial conditions and time points, if both provided, to
        list of time series as evaluation data.

        Args:
            eval_data (list[TimeSeries]): list of time series to evaluate.
            t_eval (np.ndarray | list[np.ndarray] | None): time points of
                evaluation data.
            x0_eval (np.ndarray | list[np.ndarray] | None): initial conditions
                of evaluation data.

        Returns:
            list of time series as evaluation data.
        """
        if [t_eval, x0_eval].count(None) == 1:
            msg = 'only one of t_eval and x0_eval provided; both ' + \
                'required if one of them is given'
            raise ValueError(msg)

        if t_eval is not None:
            if isinstance(x0_eval, np.ndarray):
                x0_eval = [x0_eval]

            if isinstance(t_eval, np.ndarray):
                t_eval = [t_eval] * len(x0_eval)

            eval_data = []
            for t, x0 in zip(t_eval, x0_eval):
                x = np.full((t.size, self._num_vars), np.nan)
                if x0.ndim == 1:
                    x0 = x0[np.newaxis, :]
                x[:x0.shape[0], :] = x0
                eval_data.append(TimeSeries(t, x))

        return eval_data

    def _get_mse(self, pred_data: list[TimeSeries], ref_data: list[TimeSeries]
                 ) -> dict[str, Any]:
        """Compute mean squared error MSE between prediction data and
        reference data.

        Two types of mean squared error are computed: one for each variable
        individually (stored in `indiv_mse`) and one for all variables (stored
        in `mse`).

        Args:
            pred_data (list[TimeSeries]): prediction data.
            ref_data (list[TimeSeries]): reference data.

        Returns:
            dict[str, Any]: mean squared errors.
        """
        mse = np.zeros(self._num_vars)
        num_valid_predictions = 0
        metrics = {}

        for ts_ref, ts_pred in zip(ref_data, pred_data):
            if ts_pred is None or len(ts_pred) == 0:
                continue

            # prediction data contains values only at t[0] but reference data
            # has more
            if (len(ts_pred) == 1 and len(ts_ref) > 1
                    and ts_pred.t[0] in ts_ref.t):
                continue

            ts_ref, ts_pred = align_time_series(ts_ref, ts_pred)
            if ts_ref is None or ts_pred is None:
                continue

            mse += np.mean(np.square(ts_ref.x - ts_pred.x), axis=0)
            num_valid_predictions += 1

        if num_valid_predictions > 0:
            mse /= num_valid_predictions
        else:
            mse = np.full(self._num_vars, np.nan)

        metrics['indiv_mse'] = mse
        metrics['mse'] = np.mean(mse)

        return metrics

    def print_eval_metrics(self) -> None:
        """Print evaluation metrics.

        Raises:
            RuntimeError: if eval() method has not been called.
        """
        if not self._is_evaluated:
            raise RuntimeError('eval() method must be called first')

        if not self._eval_metrics:
            print('No evaluation metrics available', flush=True)

        print('Evaluation metrics:', flush=True)
        for metric_name, metric_val in self._eval_metrics.items():
            if isinstance(metric_val, Iterable):
                metric_val = ', '.join([str(val) for val in metric_val])
                metric_val = '[' + metric_val + ']'
            else:
                metric_val = str(metric_val)

            print(f'{metric_name}: {metric_val}', flush=True)

    def plot_pred_data(self, output_suffix: str = 'pred_data',
                       plot_ref_data: bool = True,
                       ref_data: list[TimeSeries] | None = None,
                       ref_data_labels: list[str] | None = None,
                       pred_data_labels: list[str] | None = None,
                       color_scheme: Literal['mpl', 'alt'] = 'mpl') -> None:
        """Plot prediction data.

        Only valid after calling the `eval()` method. If reference data is
        provided, it is plotted with lighter colors.

        Args:
            output_suffix (str): suffix for output file (without extension).
                Default is 'pred_data'.
            plot_ref_data (bool): whether to plot reference data.
            ref_data (list[TimeSeries] | None): reference data to plot. If
                `plot_ref_data` is set to `True` but no reference data is
                given, then `eval_data` is used.
            ref_data_labels (list[str] | None): labels for reference data.
            pred_data_labels (list[str] | None): labels for prediction data.
            color_scheme (Literal['mpl', 'alt']): color scheme to use. Default
                is 'mpl' for Matplotlib colors (C0, C1, ...), 'alt' for
                alternative colors.
        """
        figure_path = os.path.join(
            self._output_dir, f'{self._output_prefix}_{output_suffix}.pdf')

        if plot_ref_data and ref_data is None:
            # use eval_data for reference if ref_data is not provided
            ref_data = self._eval_data

        # define default labels for ref_data and pred_data
        if not ref_data_labels:
            ref_data_labels = [f'eval[{i}]' for i in range(self._num_vars)]

        if not pred_data_labels:
            pred_data_labels = [f'pred[{i}]' for i in range(self._num_vars)]

        # define colors for each state variable
        if color_scheme == 'mpl':
            data_colors = [f'C{i}' for i in range(self._num_vars)]
        else:
            data_colors = [_ALT_COLORS[i] for i in range(self._num_vars)]

        # plot all pairs of ref_data and pred_data
        with PdfPages(figure_path) as pdf:
            for ts_pred, ts_ref in zip(self._pred_data, ref_data):
                plt.figure(figsize=(6, 4))

                # plot ref_data
                if plot_ref_data and ts_ref is not None:
                    for i in range(self._num_vars):
                        plt.plot(ts_ref.t, ts_ref.x[:, i], marker='o',
                                 linestyle='', color=data_colors[i], alpha=0.3,
                                 label=ref_data_labels[i])

                # plot pred_data
                if ts_pred is not None:
                    for i in range(self._num_vars):
                        plt.plot(ts_pred.t, ts_pred.x[:, i],
                                 color=data_colors[i],
                                 label=pred_data_labels[i])

                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()

                pdf.savefig()
                plt.close()


class NeuralTimeSeriesLearner(BaseTimeSeriesLearner):
    def __init__(self, train_data: list[TimeSeries],
                 output_dir: str | bytes | os.PathLike, output_prefix: str,
                 *args, torch_dtype: torch.dtype = torch.float32, **kwargs
                 ) -> None:
        """Class for learning time series data using PyTorch.

        To train a model, call the `train()` method with a PyTorch module that
        can learn multi-dimensional time series data on a windows of fixed
        size. The module should take in a tensor of shape (batch_size,
        window_size - 1, num_vars) and output a tensor of shape (batch_size,
        num_vars).

        Args:
            train_data (list[TimeSeries]): time series data to be learned.
            output_dir (str | bytes | os.PathLike): directory to save outputs.
            output_prefix (str): prefix for output files.
            *args: additional positional arguments that a subclass may use.
            torch_dtype (torch.dtype): data type for PyTorch tensors. Default
                is torch.float32.
            **kwargs: additional keyword arguments that a subclass may use.
        """
        super().__init__(train_data, output_dir, output_prefix, *args,
                         **kwargs)

        self._torch_dtype = torch_dtype
        self._training_losses = []  # training losses for all iterations

    def train(self, model: nn.Module, loss_func: Callable,
              optimizer_type: type[torch.optim.Optimizer],
              learning_rate: float, window_size: int, batch_size: int,
              num_epochs: int, input_mask: np.ndarray | Tensor, *args,
              seed: int | None = None, optimizer_kwargs: dict | None = None,
              training_params: Iterable | None = None,
              valid_data: list[TimeSeries] | None = None,
              valid_kwargs: dict | None = None, save_epoch_model: bool = False,
              epoch_model_output_suffix: str = 'model_state',
              reset_training_losses: bool = True, verbose: bool = False,
              show_progress: bool = False, **kwargs) -> None:
        """Train a model on training data.

        The training data is split into smaller windows of fixed size. The
        model should predict value of one time point in the window from the
        values of other time points in the window.

        Validation is performed after each epoch if validation data is
        provided. The validation data is evaluated as in the `eval()` method,
        except that the validation data must be full time series data, not
        just initial conditions.

        Args:
            model (nn.Module): PyTorch module to train. It should take in a
                tensor of shape (batch_size, window_size - 1, num_vars) and
                output a tensor of shape (batch_size, num_vars).
            loss_func (Callable): a function for loss between training data and
                model output.
            optimizer_type (type[torch.optim.Optimizer]): type of PyTorch
                optimizer.
            learning_rate (float): learning rate for optimizer.
            window_size (int): size of window for splitting training data.
            batch_size (int): batch size for training.
            num_epochs (int): number of training epochs.
            input_mask (np.ndarray | Tensor): boolean array as mask for input
                data to the model. Its length should be equal to window_size.
                Exactly one element should be False to indicate the time point
                to be predicted. The other elements should be True to indicate
                the time points to be used as model input. For example, if the
                mask is [True, True, False], the model will predict the value
                at the last time point from the values at the first two.
            *args: additional positional arguments. Not used here.
            seed (int | None): random seed for PyTorch. Default is None.
            optimizer_kwargs (dict | None): additional keyword arguments for
                the optimizer. Default is None.
            training_params (Iterable | None): model parameters to be trained.
                If set to None, all model parameters will be trained. Default
                is None.
            valid_data (list[TimeSeries] | None): validation data. If set to
                None, no validation will be performed. Default is None.
            valid_kwargs (dict | None): additional keyword arguments for
                validation. Refer to `eval()` for possible keyword arguments.
                Default is None.
            save_epoch_model (bool): whether to save the model after each
                epoch. Default is False.
            epoch_model_output_suffix (str): suffix for output file name of
                epoch model. Note that the epoch number will be appended to
                this suffix, e.g., '{epoch_model_output_suffix}_epoch_1'.
                Default is 'model_state'.
            reset_training_losses (bool): whether to clear any training losses
                from previous training. If set to False, training losses of
                this training run will be appended to previous ones. Default is
                True.
            verbose (bool): whether to print additional information. Default is
                `False`.
            show_progress (bool): whether to show a progress bar for each
                epoch. Default is `False`.
            **kwargs: additional keyword arguments. Not used here.
        """
        if verbose:
            print('Training model...', flush=True)

        # initialize training
        self._model = model
        self._loss_func = loss_func
        if not optimizer_kwargs:
            optimizer_kwargs = {}
        self._window_size = window_size
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._input_mask = torch.as_tensor(input_mask, dtype=torch.bool)

        if seed is not None:
            torch.manual_seed(seed)
        if training_params is None:
            training_params = self._model.parameters()
        optimizer = optimizer_type(training_params, lr=learning_rate,
                                   **optimizer_kwargs)
        if reset_training_losses:
            self._training_losses = []

        # load training data
        dataloader = get_dataloader(
            self._train_data, self._batch_size, window_size=self._window_size,
            dtype=self._torch_dtype, shuffle=True)

        # set up validation data
        if valid_data is None:
            self._valid_data = None
            self._valid_metrics = None
        else:
            self._valid_data = valid_data

            if valid_kwargs is None:
                valid_kwargs = {}

            valid_kwargs.setdefault('verbose', verbose)
            valid_kwargs.setdefault('show_progress', show_progress)

            self._valid_metrics = []

        # train the model
        for epoch in range(self._num_epochs):
            if verbose:
                print(f'Epoch {epoch}', flush=True)

            self._model.train()
            self._train(dataloader, optimizer, verbose=verbose,
                        show_progress=show_progress, **kwargs)

            # save current model
            if save_epoch_model:
                state_dict_path = self._output_prefix + \
                    f'_{epoch_model_output_suffix}_epoch_{epoch:03d}.pt'
                state_dict_path = os.path.join(
                    self._output_dir, state_dict_path)
                torch.save(self._model.state_dict(), state_dict_path)

                if verbose:
                    print('Model state for current epoch saved', flush=True)

            # validate the model
            if self._valid_data is not None:
                if verbose:
                    print('Validating model for current epoch...', flush=True)

                self._model.eval()
                num_valid_epochs = valid_kwargs.get('num_eval_epochs', 1)

                with torch.no_grad():
                    for _ in range(num_valid_epochs):
                        valid_dataloader = get_dataloader(
                            self._valid_data, 1, dtype=self._torch_dtype,
                            shuffle=False)
                        valid_pred_data = []
                        self._eval(valid_dataloader, valid_pred_data,
                                   **valid_kwargs)

                self._valid_metrics.append(
                    self._get_mse(valid_pred_data, valid_data))

                if verbose:
                    print('Validation metrics:', flush=True)
                    for metric_name, metric_values in \
                            self._valid_metrics[-1].items():
                        print(f'{metric_name}: {metric_values}', flush=True)
                    print('', flush=True)

        self._is_trained = True

        if verbose:
            print('Training finished', flush=True)

    def _train(self, dataloader: TorchDataLoader,
               optimizer: torch.optim.Optimizer, verbose: bool = False,
               show_progress: bool = False, **kwargs) -> None:
        """Train the model for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): PyTorch dataloader for
                training data.
            optimizer (torch.optim.Optimizer): PyTorch optimizer.
            verbose (bool): whether to print additional information. Default is
                `False`.
            show_progress (bool): whether to show a progress bar for training.
                Default is `False`.
        """
        for t, x in tqdm.tqdm(dataloader, disable=not show_progress):
            optimizer.zero_grad()

            t_in = t[:, self._input_mask]
            x_in = x[:, self._input_mask, :]
            x_out = x[:, ~self._input_mask, :].squeeze()

            x_pred = self._model(t_in, x_in)
            loss = self._loss_func(x_out, x_pred)
            loss.backward()
            optimizer.step()

            self._training_losses.append(loss.item())

    def save_model(self, output_suffix: str = 'model_state',
                   verbose: bool = False) -> None:
        """
        Save the model to a file.

        In particular, only the state dict of the model is saved.

        Args:
            output_suffix (str): suffix for the state dict file.
            verbose (bool): whether to print additional information. Default is
                `False`.
        """
        # TODO: save to a log file along with hyperparameters like batch_size
        state_dict_path = os.path.join(
            self.output_dir, f'{self._output_prefix}_{output_suffix}.pt')
        torch.save(self._model.state_dict(), state_dict_path)

        if verbose:
            print('Model saved', flush=True)

    def load_model(self, model: nn.Module, output_suffix: str = 'model_state',
                   verbose: bool = False) -> None:
        """
        Load state dict from a file into the model.

        Args:
            model (nn.Module): PyTorch module to load state dict into.
            output_suffix (str): suffix for the state dict file.
            verbose (bool): whether to print additional information. Default is
                `False`.
        """
        # TODO: load from a log file along with hyperparameters like batch_size
        self._model = model
        state_dict_path = os.path.join(
            self.output_dir, f'{self._output_prefix}_{output_suffix}.pt')
        self._model.load_state_dict(torch.load(state_dict_path))
        self._is_trained = True

        if verbose:
            print('Model loaded', flush=True)

    def eval(self, eval_data: list[TimeSeries] | None = None,
             t_eval: np.ndarray | list[np.ndarray] | None = None,
             x0_eval: np.ndarray | list[np.ndarray] | None = None,
             ref_data: list[TimeSeries] | None = None,
             method: Literal['rolling', 'autoregressive'] = 'autoregressive',
             num_eval_epochs: int = 1, verbose: bool = False,
             show_progress=False, **kwargs):
        """Evaluate the learned model on some data.

        Time series predicted by the learned model from the evaluation data
        will be saved in the `pred_data` attribute. Performance metrics will be
        computed and saved in the `eval_metrics` attribute.

        Two methods are available for prediction: `rolling` and
        `autoregressive`.

        `rolling` performs prediction on a rolling window.

        `autoregressive` begins prediction from the first window and uses the
        predicted values as input for the next window; only valid if the last
        element of input mask if `False`. Prefers (`t_eval`, `x0_eval`) over
        `eval_data` if both are provided.

        Note that, if (`t_eval`, `x0_eval`) is used for evaluation but no valid
        reference data is provided, the performance metrics will be correctly
        computed.

        Args:
            eval_data (list[TimeSeries] | None): list of time series to
                evaluate. If set to `None`, the training data will be used.
                Default is `None`.
            t_eval (np.ndarray | list[np.ndarray] | None): time points of
                evaluation data. Only used if method is `autoregressive`.  If
                given, `x0_eval` must also be given. Default is `None`.
            x0_eval (np.ndarray | list[np.ndarray] | None): initial conditions
                of evaluation data. Only used if method is `autoregressive`. If
                given, `t_eval` must also be given.  Default is `None`.
            ref_data (list[TimeSeries] | None): list of reference time series
                to be used for evaluating performance metrics. Default is
                `None`.
            method (Literal['rolling', 'autoregressive']): method for
                evaluation. Default is `autoregressive`.
            num_eval_epochs (int): number of evaluation epochs. Only used if
                method is `rolling`. Default is 1.
            verbose (bool): whether to print additional information. Default is
                `False`.
            show_progress (bool): whether to show a progress bar for
                evaluation. Default is `False`.
            **kwargs: additional keyword arguments. Not used here.
        """
        if not self._is_trained:
            raise RuntimeError('no model has been trained yet')

        if verbose:
            print('Evaluating with learned model...', flush=True)

        if method == 'autoregressive':
            if self._input_mask[-1]:
                msg = 'autoregressive prediction only works for models' + \
                    'that predict on the last time point in a window'
                raise RuntimeError(msg)

            eval_data = self._x0_to_eval_data(eval_data, t_eval, x0_eval)

        if eval_data is None:
            eval_data = self._train_data

        self._eval_data = eval_data

        if num_eval_epochs < 1 or method != 'rolling':
            num_eval_epochs = 1

        if ref_data is None:
            ref_data = eval_data
        self._eval_metrics = {}

        self._model.eval()
        with torch.no_grad():
            for _ in range(num_eval_epochs):
                self._pred_data = []
                dataloader = get_dataloader(
                    eval_data, 1, dtype=self._torch_dtype, shuffle=False)
                self._eval(dataloader, self._pred_data, method=method,
                           verbose=verbose, show_progress=show_progress)

                # prepare eval_data for next epoch
                eval_data = [ts.copy() for ts in self._pred_data]

        self._is_evaluated = True
        self._eval_metrics.update(self._get_mse(self._pred_data, ref_data))

        if verbose:
            print('Evaluation finished', flush=True)
            self.print_eval_metrics()

    def _eval(self, dataloader: TorchDataLoader, pred_data: list[TimeSeries],
              method: Literal['rolling', 'autoregressive'] = 'autoregressive',
              verbose: bool = False, show_progress: bool = False, **kwargs
              ) -> None:
        """Core method for evaluating the learned model.

        Args:
            dataloader (torch.utils.data.DataLoader): PyTorch dataloader for
                evaluation data.
            pred_data (list[TimeSeries]): list to store predicted time series.
            method (Literal['rolling', 'autoregressive']): method for
                evaluation. Default is `autoregressive`.
            verbose (bool): whether to print additional information. Default is
                `False`.
            show_progress (bool): whether to show a progress bar for
                evaluation. Default is `False`.
        """
        method_func = getattr(self, f'_eval_{method}')

        for t, x in tqdm.tqdm(dataloader, disable=not show_progress):
            pred_data.append(method_func(t, x))

    def _eval_rolling(self, t_in: Tensor, x_in: Tensor) -> TimeSeries:
        """Evaluate the learned model on a rolling window.

        Args:
            t_in (Tensor): time points of input data.
            x_in (Tensor): value of input data at all time points.

        Returns:
            TimeSeries: predicted data.
        """
        output_size = t_in.shape[1] - self._window_size + 1
        t_out = np.empty(output_size)
        x_out = np.empty((output_size, x_in.shape[2]))
        left, right = 0, self._window_size

        while left < output_size:
            t_window = t_in[:, left:right]
            t_window_in = t_window[:, self._input_mask]
            x_window_in = x_in[:, left:right, :][:, self._input_mask, :]

            t_out[left] = t_window[:, ~self._input_mask].item()
            x_out[left, :] = self._model(t_window_in, x_window_in).squeeze()

            left += 1
            right += 1

        return TimeSeries(t_out, x_out)

    def _eval_autoregressive(self, t_in: Tensor, x_in: Tensor) -> TimeSeries:
        """Evaluate the learned model using autoregressive prediction.

        Args:
            t_in (Tensor): time points of input data.
            x_in (Tensor): value of input data at all time points.

        Returns:
            TimeSeries: predicted data.
        """
        output_size = t_in.shape[1] - self._window_size + 1
        x_out = torch.clone(x_in)
        left, right = 0, self._window_size - 1

        while left < output_size:
            t_window_in = t_in[:, left:right]
            x_window_in = x_out[:, left:right, :]
            x_out[:, right, :] = self._model(t_window_in, x_window_in)

            left += 1
            right += 1

        # remove the input part of first window
        t_out = t_in[:, self._window_size - 1:].squeeze().numpy()
        x_out = x_out[:, self._window_size - 1:, :].squeeze().numpy()

        return TimeSeries(t_out, x_out)

    def plot_training_losses(self, output_suffix: str = 'training_losses'
                             ) -> None:
        """Plot training losses.

        Only valid after calling the `train()` method. Losses from all calls to
        the `train()` method since last "reset" are plotted. A "reset" occurs
        when the `train()` method is called with `reset_training_losses` set to
        `True`.

        Args:
            output_suffix (str): suffix for output file (without extension).
                Default is `training_losses`.

        Raises:
            RuntimeError: if no model has been trained yet.
        """
        if not self._is_trained:
            raise RuntimeError('no model trained yet')

        if not self._training_losses:
            raise RuntimeError('no training losses to plot')

        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(self._training_losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        figure_path = os.path.join(
            self._output_dir, f'{self._output_prefix}_{output_suffix}.pdf')
        plt.savefig(figure_path)
        plt.close()


class NeuralDynamicsLearner(NeuralTimeSeriesLearner):
    def __init__(self, train_data: list[TimeSeries],
                 output_dir: str | bytes | os.PathLike, output_prefix: str,
                 *args, torch_dtype: torch.dtype = torch.float32, **kwargs
                 ) -> None:
        """Class for learning time series data as ODE dynamics using PyTorch.

        To train a model, call the `train()` method with a PyTorch module that
        takes in a tensor of shape (batch_size, num_vars) and output a tensor
        of shape (batch_size, num_vars). The input tensor is the value of
        variables at a single time point. The output tensor is the derivative
        of the variables at that time point. Note that the module should not
        include batch operations such as batch normalization, as the ODE solver
        from `torchdiffeq` does not support batch input with different time
        points.

        In addition to `pred_data`, this class also stores `sub_pred_data` for
        outputs of selected sub-modules if specified when calling the `eval()`
        method.

        Args:
            train_data (list[TimeSeries]): time series data to be learned.
            output_dir (str | bytes | os.PathLike): directory to save outputs.
            output_prefix (str): prefix for output files.
            *args: additional positional arguments. Not used here.
            torch_dtype (torch.dtype): data type for PyTorch tensors. Default
                is torch.float32.
            **kwargs: additional keyword arguments. Not used here.
        """
        super().__init__(train_data, output_dir, output_prefix, *args,
                         torch_dtype=torch_dtype, **kwargs)

        self._ode_integrator = None
        self._sub_pred_data: dict[str, list[np.ndarray]] | None = None

    @property
    def sub_pred_data(self) -> dict[str, list[np.ndarray]] | None:
        """dict[str, list[np.ndarray]] | None: outputs of sub-modules.

        If sub-modules are specified when calling the `eval()` method, this
        property stores the outputs of the sub-modules as a dict mapping
        sub-module name to data. Each data is a list of NumPy arrays. Each
        array is the value predicted by the sub-module at all time points,
        namely `pred_data.t`.
        """
        # TODO: change to dict[str, list[TimeSeries]]?
        if self._sub_pred_data is None:
            return None

        return self._sub_pred_data

    def train(self, model: nn.Module, loss_func: Callable,
              optimizer_type: type[torch.optim.Optimizer],
              learning_rate: float, window_size: int, batch_size: int,
              num_epochs: int, *args, seed: int | None = None,
              optimizer_kwargs: dict | None = None,
              training_params: Iterable | None = None,
              integrator_backend:
              Literal['torchdiffeq', 'torchode'] = 'torchdiffeq',
              torchode_step_method: Literal['Dopri5', 'Tsit5'] = 'Dopri5',
              valid_data: list[TimeSeries] | None = None,
              valid_kwargs: dict | None = None, save_epoch_model: bool = False,
              epoch_model_output_suffix: str = 'model_state',
              reset_training_losses: bool = True, verbose: bool = False,
              show_progress: bool = False, **kwargs) -> None:
        """Train a model on training data.

        The training data is split into smaller windows of fixed size. The
        model should predict value of one time point in the window from the
        values of other time points in the window.

        Validation is performed after each epoch if validation data is
        provided. The validation data is evaluated as in the `eval()` method,
        except that the validation data must be full time series data, not
        just initial conditions.

        Args:
            model (nn.Module): PyTorch module to train. It should take in a
                tensor of shape (batch_size, window_size - 1, num_vars) and
                output a tensor of shape (batch_size, num_vars).
            loss_func (Callable): a function for loss between training data and
                model output.
            optimizer_type (type[torch.optim.Optimizer]): type of PyTorch
                optimizer.
            learning_rate (float): learning rate for optimizer.
            window_size (int): size of window for splitting training data.
            batch_size (int): batch size for training.
            num_epochs (int): number of training epochs.
            *args: additional positional arguments. Not used here.
            seed (int | None): random seed for PyTorch. Default is None.
            optimizer_kwargs (dict | None): additional keyword arguments for
                the optimizer. Default is None.
            training_params (Iterable | None): model parameters to be trained.
                If set to None, all model parameters will be trained. Default
                is None.
            integrator_backend (Literal['torchdiffeq', 'torchode']): backend
                for ODE integrator. Default is `torchdiffeq`.
            torchode_step_method (Literal['Dopri5', 'Tsit5']): type of ODE
                integration method from `torchode`. Only used if
                `integrator_backend` is `torchode`. Default is `Dopri5`.
            valid_data (list[TimeSeries] | None): validation data. If set to
                None, no validation will be performed. Default is None.
            valid_kwargs (dict | None): additional keyword arguments for
                validation. Refer to `eval()` for possible keyword arguments.
                Default is None.
            save_epoch_model (bool): whether to save the model after each
                epoch. Default is False.
            epoch_model_output_suffix (str): suffix for output file name of
                epoch model. Note that the epoch number will be appended to
                this suffix, e.g., '{epoch_model_output_suffix}_epoch_1'.
                Default is 'model_state'.
            reset_training_losses (bool): whether to clear any training losses
                from previous training. Default is True.
            verbose (bool): whether to print additional information. Default is
                `False`.
            show_progress (bool): whether to show a progress bar for each
                epoch. Default is `False`.
            **kwargs: additional keyword arguments. Not used here.
        """
        dummy_input_mask = torch.zeros(window_size, dtype=bool)
        self._integrator_backend = integrator_backend

        # set up torchode integrator
        if self._integrator_backend == 'torchode':
            self._get_torch_ode_integrator(model, torchode_step_method)

        super().train(model, loss_func, optimizer_type, learning_rate,
                      window_size, batch_size, num_epochs, dummy_input_mask,
                      *args, seed=seed, optimizer_kwargs=optimizer_kwargs,
                      training_params=training_params, valid_data=valid_data,
                      valid_kwargs=valid_kwargs,
                      save_epoch_model=save_epoch_model,
                      epoch_model_output_suffix=epoch_model_output_suffix,
                      reset_training_losses=reset_training_losses,
                      verbose=verbose, show_progress=show_progress, **kwargs)

    def _train(self, dataloader: TorchDataLoader,
               optimizer: torch.optim.Optimizer, verbose: bool = False,
               show_progress: bool = False, **kwargs) -> None:
        """Train the model for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): PyTorch dataloader for
                training data.
            optimizer (torch.optim.Optimizer): PyTorch optimizer.
            verbose (bool): whether to print additional information. Default is
                `False`.
            show_progress (bool): whether to show a progress bar for training.
                Default is `False`.
        """
        for t, x in tqdm.tqdm(dataloader, disable=not show_progress):
            loss = 0
            optimizer.zero_grad()

            if self._integrator_backend == 'torchode':
                ivp_problem = torchode.InitialValueProblem(
                    y0=x[:, 0, :], t_eval=t)
                ivp_solution = self._ode_integrator.solve(ivp_problem)
                loss = self._loss_func(x, ivp_solution.ys)
            else:  # self._integrator_backend == 'torchdiffeq'
                for i in range(self._batch_size):
                    x_out = tdf_odeint(self._model, x[i, 0, :], t[i, :])
                    loss += self._loss_func(x[i, ...], x_out)
                loss /= self._batch_size

            loss.backward()
            optimizer.step()
            self._training_losses.append(loss.item())

    def _get_torch_ode_integrator(
            self, model: nn.Module,
            step_method: Literal['Dopri5', 'Tsit5'] = 'Dopri5') -> None:
        """Get ODE integrator from `torchode`.

        The ODE integrator will be stored in the `_ode_integrator` attribute.

        Args:
            model (nn.Module): PyTorch module to be integrated.
            torchode_step_method (Literal['Dopri5', 'Tsit5']): type of ODE
                integration method from `torchode`.  Default is `Dopri5`.
        """
        ode_term = torchode.ODETerm(model)
        ode_step_method = getattr(torchode, step_method)(term=ode_term)
        ode_step_controller = torchode.IntegralController(
            atol=1e-9, rtol=1e-7, term=ode_term)
        ode_integrator = torchode.AutoDiffAdjoint(
            ode_step_method, ode_step_controller)
        self._ode_integrator = torch.compile(ode_integrator)

    def eval(self, eval_data: list[TimeSeries] | None = None,
             t_eval: np.ndarray | list[np.ndarray] | None = None,
             x0_eval: np.ndarray | list[np.ndarray] | None = None,
             ref_data: list[TimeSeries] | None = None,
             integrator_backend:
             Literal['torchdiffeq', 'scipy'] = 'torchdiffeq',
             integrator_kwargs: dict | None = None,
             torchode_step_method: Literal['Dopri5', 'Tsit5'] = 'Dopri5',
             sub_modules: str | list[str] | None = None,
             verbose: bool = False, show_progress=False, **kwargs) -> None:
        """Evaluate the learned model on some data.

        Time series predicted by the learned model from the evaluation data
        will be saved in the `pred_data` attribute. Performance metrics will be
        computed and saved in the `eval_metrics` attribute.

        Prefers (`t_eval`, `x0_eval`) over `eval_data` if both are provided.

        Note that, if (`t_eval`, `x0_eval`) is used for evaluation but no valid
        reference data is provided, the performance metrics will be correctly
        computed.

        Args:
            eval_data (list[TimeSeries] | None): list of time series to
                evaluate. If set to `None`, the training data will be used.
                Default is `None`.
            t_eval (np.ndarray | list[np.ndarray] | None): time points of
                evaluation data. Only used if method is `autoregressive`.  If
                given, `x0_eval` must also be given. Default is `None`.
            x0_eval (np.ndarray | list[np.ndarray] | None): initial conditions
                of evaluation data. Only used if method is `autoregressive`. If
                given, `t_eval` must also be given.  Default is `None`.
            ref_data (list[TimeSeries] | None): list of reference time series
                to be used for evaluating performance metrics. Default is
                `None`.
            integrator_backend (Literal['scipy', 'torchdiffeq', 'torchode']):
                backend of ODE integrator. Default is `scipy`.
            integrator_kwargs (dict | None): additional keyword arguments for
                ODE integrator. `t_eval` is ignored. Default is `None`.
            torchode_step_method (Literal['Dopri5', 'Tsit5']): type of ODE
                integration method from `torchode`. Only used if
                `integrator_backend` is `torchode`. Default is `Dopri5`.
            sub_modules (str | list[str] | None): name(s) of sub-modules to
                be evaluated in addition to the main module. Outputs from sub-
                modules are stored in `sub_pred_data`, a dict mapping sub-
                module name to data. Default is `None`.
            verbose (bool): whether to print additional information. Default is
                `False`.
            show_progress (bool): whether to show a progress bar for
                evaluation. Default is `False`.
            **kwargs: additional keyword arguments. Not used here.
        """
        if not self._is_trained:
            raise RuntimeError('no model has been trained yet')

        if verbose:
            print('Evaluating with learned model...', flush=True)

        eval_data = self._x0_to_eval_data(eval_data, t_eval, x0_eval)
        if eval_data is None:
            eval_data = self._train_data

        self._eval_data = eval_data
        self._pred_data = []
        self._sub_pred_data = {}
        if sub_modules is not None:
            if isinstance(sub_modules, str):
                sub_modules = list(sub_modules)

            self._sub_pred_data = {m: [] for m in sub_modules}

        if ref_data is None:
            ref_data = eval_data
        self._eval_metrics = {}

        if integrator_backend == 'torchode':
            self._get_torch_ode_integrator(
                self._model, step_method=torchode_step_method)

        self._model.eval()
        with torch.no_grad():
            dataloader = get_dataloader(eval_data, 1, dtype=self._torch_dtype,
                                        shuffle=False)
            self._eval(dataloader, self._pred_data,
                       integrator_backend=integrator_backend,
                       integrator_kwargs=integrator_kwargs,
                       sub_modules=sub_modules,
                       sub_pred_data=self._sub_pred_data, verbose=verbose,
                       show_progress=show_progress)

        self._is_evaluated = True
        self._eval_metrics.update(self._get_mse(self._pred_data, ref_data))

        if verbose:
            print('Evaluation finished', flush=True)
            self.print_eval_metrics()

    def _eval(self, dataloader: TorchDataLoader, pred_data: list[TimeSeries],
              integrator_backend:
              Literal['scipy', 'torchdiffeq', 'torchode'] = 'torchdiffeq',
              integrator_kwargs: dict | None = None,
              sub_modules: str | list[str] | None = None,
              sub_pred_data: dict[str, list[np.ndarray]] | None = None,
              verbose: bool = False, show_progress: bool = False, **kwargs
              ) -> None:
        """ Core method for evaluating the learned dynamics.

        Args:
            dataloader (torch.utils.data.DataLoader): PyTorch dataloader for
                evaluation data.
            pred_data (list[TimeSeries]): list to store predicted time series.
            integrator_backend (Literal['scipy', 'torchdiffeq', 'torchode']):
                backend of ODE integrator. Default is `scipy`.
            integrator_kwargs (dict | None): additional keyword arguments for
                ODE integrator. `t_eval` is ignored. Default is `None`.
            sub_modules (str | list[str] | None): name(s) of sub-modules to
                be evaluated in addition to the main module. Outputs from sub-
                modules are stored in `sub_pred_data`, a dict mapping sub-
                module name to data. Default is `None`.
            sub_pred_data (dict[str, list[np.ndarray]] | None): dict to store
                outputs of sub-modules. Default is `None`.
            verbose (bool): whether to print additional information. Default is
                `False`.
            show_progress (bool): whether to show a progress bar for
                evaluation. Default is `False`.
            **kwargs: additional keyword arguments. Not used here.
        """
        if sub_modules is not None and sub_pred_data is None:
            raise ValueError('sub_pred_data must be provided if sub_modules'
                             ' is provided')

        if integrator_kwargs is None:
            integrator_kwargs = {}
        else:
            integrator_kwargs = integrator_kwargs.copy()
            # ignore existing t_eval in integrator_kwargs
            integrator_kwargs.pop('t_eval', None)

        def _scipy_ode_func(t_, x_):
            t_ = torch.as_tensor(t_, dtype=self._torch_dtype)
            x_ = torch.as_tensor(x_, dtype=self._torch_dtype)

            return self._model(t_, x_).numpy()

        for t, x in tqdm.tqdm(dataloader, disable=not show_progress):
            t = t[0, ...]
            x = x[0, ...]
            t_np = t.numpy()
            x_np = x.numpy()

            is_integrator_successful = True

            # TODO: implement with torchode
            if integrator_backend == 'torchdiffeq':
                x_pred = tdf_odeint(
                    self._model, x[0, :], t, **integrator_kwargs)
                pred_data.append(TimeSeries(t.numpy(), x_pred.numpy()))
            elif integrator_backend == 'torchode':
                ivp_problem = torchode.InitialValueProblem(
                    y0=x[None, 0, :], t_eval=t[None, :])
                ivp_solution = self._ode_integrator.solve(
                    ivp_problem, **integrator_kwargs)
                x_pred = ivp_solution.ys[0, ...]
                pred_data.append(TimeSeries(t.numpy(), x_pred.numpy()))
            else:  # integrator_backend == 'scipy'
                ivp_solution = solve_ivp(
                    _scipy_ode_func, (t_np[0], t_np[-1]), x_np[0, :],
                    t_eval=t_np, **integrator_kwargs)

                is_integrator_successful = ivp_solution.success
                if is_integrator_successful:
                    x_pred = ivp_solution.y.T
                    pred_data.append(TimeSeries(t_np, x_pred))
                    x_pred = torch.tensor(x_pred, dtype=self._torch_dtype)
                else:
                    pred_data.append(None)

            # compute results for sub-modules
            if is_integrator_successful and sub_modules is not None:
                for module_name in sub_modules:
                    module = getattr(self._model, module_name)
                    module_pred = module(t, x_pred).numpy()
                    sub_pred_data[module_name].append(module_pred)


def mask_array_to_bin(mask: Iterable) -> int:
    """Convert a boolean array to an equivalent integer under its binary
    representation

    Args:
        mask (Iterable): an iterable with boolean elements

    Returns:
        int: binary representation of the input mask
    """
    num = 0

    for m in mask:
        num = num << 1 | m

    return num


class OdeSystemLearner(BaseTimeSeriesLearner):
    def __init__(self, train_data: list[TimeSeries],
                 output_dir: str | bytes | os.PathLike, output_prefix: str,
                 *args, **kwargs) -> None:
        """Class for learning symbolic form of ordinary differential equation
        system from time series data using SINDy.

        Args:
            train_data (list[TimeSeries]): time series data to be learned.
            output_dir (str | bytes | os.PathLike): directory to save outputs.
            output_prefix (str): prefix for output files.
            *args: additional positional arguments. Not used here.
            **kwargs: additional keyword arguments. Not used here.
        """
        super().__init__(train_data, output_dir, output_prefix, *args,
                         **kwargs)

    def train(self, *args, backend: Literal['pysindy', 'custom'] = 'pysindy',
              optimizer_type: Any | None = None, threshold: float = 0.1,
              learn_dx: bool = False, normalize_columns: bool = False,
              basis_funcs: list[Callable] | None = None,
              basis_names: list[Callable] | None = None,
              optimizer_kwargs: dict[str, Any] | None = None,
              valid_data: list[TimeSeries] | None = None,
              valid_kwargs: dict[str, Any] | None = None,
              verbose: bool = False, **kwargs) -> None:
        """Reconstruct symbolic form of ordinary differential equations from
        training data using SINDy.

        Args:
            *args: additional positional arguments. Not used here.
            backend (Literal['pysindy', 'custom']): backend for equation
                learning. Currently only 'pysindy' is implemented, which uses
                the PySINDy package. Default is 'pysindy'.
            optimizer_type (Any | None): type of optimizer for equation
                learning. Default is None. For PySINDy backend, default is
                STLSQ.
            threshold (float): threshold for sparsity of equation learning.
                Default is 0.1.
            learn_dx (bool): whether to learn from the derivative of the data.
                Default is False.
            normalize_columns (bool): whether to normalize each column of
                Theta(X), where Theta is the library of basis functions and X
                is the time series data. Default is False.
            basis_funcs (list[Callable] | None): list of basis functions to
                use. Default is None, which uses the default basis functions
                determined by the backend.
            basis_names (list[Callable] | None): list of names for basis
                functions. Default is None, which uses the default basis names
                determined by the backend.
            optimizer_kwargs (dict[str, Any] | None): additional keyword
                arguments for the optimizer. Default is None.
            valid_data (list[TimeSeries] | None): validation data. If set to
                None, no validation will be performed. Default is None.
            valid_kwargs (dict[str, Any] | None): additional keyword arguments
                for validation. Refer to `eval()` for possible keyword
                arguments. Default is None.
            verbose (bool): whether to print additional information. Default is
                `False`.
            **kwargs: additional keyword arguments. Not used here.
        """
        if backend == 'custom':
            raise NotImplementedError('custom SINDy not implemented yet')

        # initialize SINDy learning
        self._backend = backend
        self._normalize_columns = normalize_columns

        if optimizer_type is None and backend == 'pysindy':
            optimizer_type = ps.STLSQ
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer = optimizer_type(
            threshold=threshold, normalize_columns=self._normalize_columns,
            verbose=verbose, **optimizer_kwargs)
        if basis_funcs is None:
            basis_lib = None
        else:
            basis_lib = CustomLibrary(basis_funcs, basis_names)
        self._model = ps.SINDy(optimizer=optimizer, feature_library=basis_lib)

        # transform training data
        x = [ts.x for ts in self._train_data]
        t = [ts.t for ts in self._train_data]

        if valid_data is None:
            self._valid_data = None
            self._valid_metrics = None
        else:
            self._valid_data = valid_data
            self._valid_metrics = {}

        if verbose:
            print('Learning ODEs...', flush=True)

        if learn_dx:
            dx = [ts.dx for ts in self._train_data]

            if all(np.array_equal(t[0], t_i) for t_i in t) \
                    and all(np.array_equal(dx[0], dx_i) for dx_i in dx):
                # fit one sample only if all time series share the same time
                # points and dx's are equal
                self._model.fit(x[0], t=t[0], x_dot=dx[0])
            else:
                self._model.fit(x, t=t, x_dot=dx, multiple_trajectories=True)
        else:
            self._model.fit(x, t=t, multiple_trajectories=True)

        self._is_trained = True

        # validate learned ODEs
        if valid_data is not None:
            if verbose:
                print('Validating learned ODEs...', flush=True)

            valid_pred_data = []
            self._eval(self._valid_data, valid_pred_data, **valid_kwargs)
            self._valid_metrics.update(
                self._get_mse(valid_pred_data, valid_data))
            self._valid_metrics.update(
                self._get_aicc(valid_pred_data, valid_data))

        if verbose:
            print('Learned ODEs:', flush=True)
            self._model.print()
            sys.stdout.flush()

    def eval(self, eval_data: list[TimeSeries] | None = None,
             t_eval: np.ndarray | list[np.ndarray] | None = None,
             x0_eval: np.ndarray | list[np.ndarray] | None = None,
             ref_data: list[TimeSeries] | None = None,
             eval_func: Callable | None = None,
             integrator_kwargs: dict | None = None, verbose: bool = False,
             **kwargs) -> None:
        """Evaluate the learned ODEs on data.

        Time series predicted by the learned model from the evaluation data
        will be saved in the `pred_data` attribute. Performance metrics will be
        computed and saved in the `eval_metrics` attribute.

        Note that, if (`t_eval`, `x0_eval`) is used for evaluation but no valid
        reference data is provided, the performance metrics will be correctly
        computed.

        Args:
            eval_data (list[TimeSeries] | None): list of time series to
                evaluate. If set to `None`, the training data will be used.
                Default is `None`.
            t_eval (np.ndarray | list[np.ndarray] | None): time points of
                evaluation data. Only used if method is `autoregressive`.  If
                given, `x0_eval` must also be given. Default is `None`.
            x0_eval (np.ndarray | list[np.ndarray] | None): initial conditions
                of evaluation data. Only used if method is `autoregressive`. If
                given, `t_eval` must also be given.  Default is `None`.
            ref_data (list[TimeSeries] | None): list of reference time series
                to be used for evaluating performance metrics. Default is
                `None`.
            eval_func (Callable | None): function for evaluation. Useful if
                `eval_func` is a larger system in which the learned system is
                part of. Must be of the form `def eval_func(t, x, model)` where
                `model` is the learned model. If set to `None`, the learned
                system will be used. Default is `None`.
            integrator_kwargs (dict | None): additional keyword arguments for
                the integrator. Default is `None`.
            verbose (bool): whether to print additional information. Default is
                `False`.
            **kwargs: additional keyword arguments. Not used here.
        """
        if not self._is_trained:
            raise RuntimeError('No ODEs learned yet')

        if verbose:
            print('Evaluating with learned ODEs...', flush=True)

        # set up evaluation data and prediction data
        eval_data = self._x0_to_eval_data(eval_data, t_eval, x0_eval)
        if eval_data is None:
            eval_data = self._train_data
        self._eval_data = eval_data
        eval_sample_size = len(eval_data)
        self._pred_data = []

        # set up integrator
        if integrator_kwargs is None:
            integrator_kwargs = {}

        if ref_data is None:
            ref_data = eval_data
        self._eval_metrics = {}

        num_successes = 0

        self._eval(self._eval_data, self._pred_data, eval_func,
                   integrator_kwargs)

        self._is_evaluated = True
        self._eval_metrics.update(self._get_mse(self._pred_data, ref_data))
        self._eval_metrics.update(self._get_aicc(self._pred_data, ref_data))

        for ts_eval, ts_pred in zip(self._eval_data, self._pred_data):
            if ts_pred is not None and np.array_equal(ts_eval.t, ts_pred.t):
                num_successes += 1
        self._eval_metrics['success_rate'] = num_successes / eval_sample_size

        if verbose:
            print('Evaluation finished', flush=True)
            print('Number of evaluation samples:', eval_sample_size,
                  flush=True)
            print('Number of successful evaluations:', num_successes,
                  flush=True)
            self.print_eval_metrics()

    def _eval(self, eval_data: list[TimeSeries], pred_data: list[TimeSeries],
              eval_func: Callable | None = None,
              integrator_kwargs: dict | None = None) -> None:
        """Core method for evaluating the learned ODEs.

        Args:
            eval_data (list[TimeSeries]): list of time series to evaluate.
            pred_data (list[TimeSeries]): list to store predicted time series.
            eval_func (Callable | None): function for evaluation. Must be of
                the form `def eval_func(t, x, model)` where `model` is the
                learned model. If set to `None`, the learned system will be
                used. Default is `None`.
            integrator_kwargs (dict | None): additional keyword arguments for
                the integrator. Default is `None`.
        """
        # set up evaluation function
        if eval_func is None:
            def default_eval_func(t, x):
                return self._model.predict(x[np.newaxis, :])[0]

            eval_func = default_eval_func

        # convert 'model' in args (used by solve_ivp()) to the actual learned
        # model. this is useful when the learned model is needed for validation
        # inside train(), as self._model is yet to be created when train() is
        # first called
        if 'args' in integrator_kwargs and integrator_kwargs['args']:
            args = list(integrator_kwargs['args'])
            try:
                idx = args.index('model')
                args[idx] = self._model
                integrator_kwargs['args'] = tuple(args)
            except ValueError:
                pass

        for ts in eval_data:
            ivp_solution = solve_ivp(eval_func, (ts.t[0], ts.t[-1]),
                                     ts.x[0, :], t_eval=ts.t,
                                     **integrator_kwargs)

            if isinstance(ivp_solution.t, np.ndarray):
                ts_pred = TimeSeries(ivp_solution.t, ivp_solution.y.T)
                pred_data.append(ts_pred)
            else:
                pred_data.append(None)

    def _get_aicc(self, pred_data: list[TimeSeries],
                  ref_data: list[TimeSeries]) -> dict[str, Any]:
        """Compute the Akaike information criterion with correction (AICc)
        for the learned ODEs from evaluation data and prediction data.

        See Mangan et al., 2017 (doi: 10.1098/rspa.2017.0009).

        Args:
            pred_data (list[TimeSeries]): prediction data.
            ref_data (list[TimeSeries]): reference data.

        Returns:
            dict[str, Any]: AICc for each variable and the overall AICc.
        """
        k = self._model.coefficients().size  # number of free params to fit
        rss = np.zeros(self._num_vars)  # residual sum of squares
        m = 0  # number of prediction samples actually used for computing AICc
        metrics = {}

        for ts_ref, ts_pred in zip(ref_data, pred_data):
            if ts_pred is None or len(ts_pred) == 0:
                continue

            # prediction data contains values only at t[0] but reference data
            # has more
            if (len(ts_pred) == 1 and len(ts_ref) > 1
                    and ts_pred.t[0] in ts_ref.t):
                continue

            ts_ref, ts_pred = align_time_series(ts_ref, ts_pred)
            if ts_ref is None or ts_pred is None:
                continue

            rss += np.mean(np.square(ts_ref.x - ts_pred.x), axis=0)
            m += 1

        if m > 0:
            indiv_aic = m * np.log(rss / m) + 2 * k
            aic = m * np.log(np.mean(rss) / m) + 2 * k
            correction = 2 * (k + 1) * (k + 2) / (m - k - 2)
            metrics['indiv_aicc'] = indiv_aic + correction
            metrics['aicc'] = np.mean(aic) + correction
        else:
            metrics['indiv_aicc'] = np.full(self._num_vars, np.nan)
            metrics['aicc'] = np.nan

        return metrics


def get_nn_derivatives(model: nn.Module, x: Tensor) -> Tensor:
    """Compute the derivatives of a neural network model at a given point.

    Args:
        model (nn.Module): neural network model.
        x (Tensor): point at which to compute the derivative.

    Returns:
        Tensor: derivatives of the model at the given point.
    """
    x = x.clone().requires_grad_(True)
    y = model(x)
    dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]

    return dy_dx
