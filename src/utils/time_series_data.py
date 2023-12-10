from __future__ import annotations
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
import h5py


class TimeSeries:
    def __init__(self, t: np.ndarray, x: np.ndarray,
                 dx: np.ndarray | None = None) -> None:
        """Utility class for discretized multivariate time series data.

        Args:
            t (np.ndarray): time points of shape (T, ) where T is the number of
                time points.
            x (np.ndarray): value of the time series of shape (T, d) where T is
                the number of time points and d is the number of variables.
            dx (np.ndarray | None, optional): value of derivatives of shape
                (T, d). Defaults to None.

        Raises:
            RuntimeError: if t is not a 1-d array.
            RuntimeError: if x is not a 1-d or 2-d array.
            RuntimeError: if t and x have different number of time points.
            RuntimeError: if dx is not a 1-d or 2-d array when dx is not None.
            RuntimeError: if t and dx have different number of time points
                when dx is not None.
            RuntimeError: if x and dx have different number of variables
                when dx is not None.
        """
        # check input t
        if t.ndim != 1:
            raise RuntimeError('t is expected to be a 1-d array')

        # check input x
        match x.ndim:
            case 1:
                x = x[:, np.newaxis].copy()
            case 2:
                x = x.copy()
            case _:
                raise RuntimeError('x is expected to be a 1-d or 2-d array')

        if t.shape[0] != x.shape[0]:
            msg = f't has {t.shape[0]} time point(s), but x has {x.shape[0]}'
            raise RuntimeError(msg)

        # check input dx
        if dx is not None:
            match dx.ndim:
                case 1:
                    dx = dx[:, np.newaxis].copy()
                case 2:
                    dx = dx.copy()
                case _:
                    msg = 'dx is expected to be a 1-d or 2-d array'
                    raise RuntimeError(msg)

            if t.shape[0] != dx.shape[0]:
                msg = f't has {t.shape[0]} time point(s), but dx has ' \
                      f'{dx.shape[0]}'
                raise RuntimeError(msg)

            if x.shape[1] != dx.shape[1]:
                msg = f'x has {x.shape[1]} variable(s), but dx has ' \
                      f'{dx.shape[1]}'
                raise RuntimeError(msg)

        # store input data
        self._t = t.copy()
        self._x = x
        self._dx = dx

    def __getitem__(self, idx):
        dx = None

        if isinstance(idx, int):
            t = self._t[idx, np.newaxis]
            x = self._x[idx, :, np.newaxis]

            if self._dx is not None:
                dx = self._dx[idx, :, np.newaxis]
        elif isinstance(idx, slice) or isinstance(idx, np.ndarray):
            t = self._t[idx]
            x = self._x[idx, :]

            if self._dx is not None:
                dx = self._dx[idx, :]
        else:
            raise RuntimeError('index must be an integer or a slice')

        return TimeSeries(t, x, dx=dx)

    def __len__(self):
        return self._t.size

    # TODO: implement __repr__

    @property
    def t(self) -> np.ndarray:
        """np.ndarray: time points of the time series."""
        return self._t.copy()

    @t.setter
    def t(self, t_new: np.ndarray) -> None:
        self._t = t_new.copy()

    @property
    def x(self) -> np.ndarray:
        """np.ndarray: values of the time series."""
        return self._x.copy()

    @x.setter
    def x(self, x_new: np.ndarray) -> None:
        self._x = x_new.copy()

    @property
    def dx(self) -> np.ndarray | None:
        """np.ndarray | None: values of the derivatives of the time series."""
        return None if self._dx is None else self._dx.copy()

    @dx.setter
    def dx(self, dx_new: np.ndarray) -> None:
        self._dx = dx_new.copy()

    @property
    def num_vars(self) -> int:
        """int: number of variables in the time series."""
        return self._x.shape[1]

    def copy(self) -> TimeSeries:
        """Return a copy of the time series.

        Returns:
            TimeSeries: copy of the time series.
        """
        if self._dx is None:
            return TimeSeries(self._t.copy(), self._x.copy())

        return TimeSeries(self._t.copy(), self._x.copy(), self._dx.copy())


def align_time_series(a: TimeSeries | None, b: TimeSeries | None
                      ) -> tuple[TimeSeries | None, TimeSeries | None]:
    """Given two time series, find their overlapping parts (i.e. data on
    overlapping time points).

    Args:
        a (TimeSeries): first time series.
        b (TimeSeries): second time series.

    Returns:
        tuple[TimeSeries | None, TimeSeries | None]: aligned time series. If
            the two time series do not have any overlapping time points, (None,
            None) is returned.
    """
    if a is None or b is None:
        return None, None

    t, a_idx, b_idx = np.intersect1d(a.t, b.t, return_indices=True)

    if t.size == 0:
        return None, None

    return a[a_idx], b[b_idx]


class TimeSeriesDataset(TorchDataset):
    def __init__(self, data: list[TimeSeries], window_size: int | None = None,
                 window_order: Literal['sample', 'time'] = 'sample',
                 dtype: type = torch.float32) -> None:
        """PyTorch dataset for time series data.

        Loads a list of time series into a PyTorch dataset. If desired, the
        input time series can be split into windows of a fixed size. Each
        time series in the dataset is converted into a PyTorch tensor.

        Args:
            data (list[TimeSeries]): list of time series in the dataset
            window_size (int | None, optional): window size to split time
                series. Defaults to None.
            window_order (Literal['sample', 'time'], optional): order of the
                windows. If set to 'sample', windows from the same time series
                are grouped together. If set to 'time', all time series are
                assumed to have the same time points and windows of the same
                time interval from different time series are grouped together.
                Defaults to 'sample'.
            dtype (type, optional): data type of the time series. Defaults to
                torch.float32.
        """
        self._t = []
        self._x = []

        if window_size is None:
            for d in data:
                self._t.append(torch.tensor(d.t, dtype=dtype))
                self._x.append(torch.tensor(d.x, dtype=dtype))
        else:
            window_size = max(window_size, 1)

            if window_order == 'time':
                self._order_windows_by_time(data, window_size, dtype=dtype)
            else:  # sequence_order == 'sample'
                self._order_windows_by_sample(data, window_size, dtype=dtype)

    def _order_windows_by_sample(self, data, window_size, dtype):
        """
        Split input time series into windows of a fixed size such that windows
        from the same time series are grouped together.

        Args:
            data (list[TimeSeries]): list of time series in the dataset.
            window_size (int): window size to split time series.
            dtype (type): data type of the time series.
        """
        for d in data:
            t, x = d.t, d.x

            for j in range(t.size - window_size + 1):
                self._t.append(
                    torch.tensor(t[j:j + window_size], dtype=dtype))
                self._x.append(
                    torch.tensor(x[j:j + window_size], dtype=dtype))

    def _order_windows_by_time(self, data, window_size, dtype):
        """
        Split input time series into windows of a fixed size such that windows
        from the same time interval are grouped together.

        All time series are assumed to share the same time points.

        Args:
            data (list[TimeSeries]): list of time series in the dataset.
            window_size (int): window size to split time series.
            dtype (type): data type of the time series.
        """
        t = data[0].t

        for j in range(t.size - window_size + 1):
            for d in data:
                self._t.append(
                    torch.tensor(d.t[j:j + window_size], dtype=dtype))
                self._x.append(
                    torch.tensor(d.x[j:j + window_size], dtype=dtype))

    def __len__(self):
        return len(self._t)

    def __getitem__(self, index):
        return self._t[index], self._x[index]


def get_dataloader(
    data: list[TimeSeries], batch_size: int, window_size: int | None = None,
    window_order: Literal['sample', 'time'] = 'sample',
    dtype: type = torch.float32, shuffle: bool = True
) -> TorchDataLoader:
    """Return a PyTorch dataloader for time series data.

    Args:
        data (list[TimeSeries]): list of time series in the dataset.
        batch_size (int): batch size for training.
        window_size (int | None, optional): window size to split time series.
            Defaults to None.
        window_order (Literal['sample', 'time'], optional): order of the
            time series windows. See :py:class:`TimeSeriesDataset` for more
            information. Defaults to 'sample'.
        dtype (type, optional): data type of the time series. Defaults to
            torch.float32.
        shuffle (bool, optional): whether to shuffle samples in the dataset.
            Defaults to True.
    """
    dataset = TimeSeriesDataset(
        data, window_size=window_size, window_order=window_order, dtype=dtype)
    dataloader = TorchDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def load_dataset_from_h5(h5_fd: h5py.File, dataset_name: str
                         ) -> list[TimeSeries]:
    '''
    Load a dataset from a HDF5 file.

    Note dataset here means a collection of time series, not a HDF5 dataset.
    Each time series dataset is stored as a HDF5 group in the HDF5. Each time
    series sample is stored as a HDF5 data group. Within each data group, the
    time points are stored as a HDF5 dataset named 't' and the values are
    stored as a HDF5 dataset named 'x'.

    Args:
        h5_fd (h5py.File): HDF5 file descriptor.
        dataset_name (str): name of the dataset.

    Returns:
        list[TimeSeries]: list of time series in the dataset.
    '''
    samples = []
    data_group = h5_fd[dataset_name]

    for i in range(data_group.attrs['num_samples']):
        data = data_group[f'sample_{i:04d}']
        samples.append(TimeSeries(data['t'][...], data['x'][...]))

    return samples
