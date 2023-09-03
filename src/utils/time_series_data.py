from __future__ import annotations
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader


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
            RuntimeError: if x is not a 2-d array.
            RuntimeError: if t and x have different number of time points.
        """
        # check inputs
        if t.ndim != 1:
            raise RuntimeError('t is expected to be a 1-d array')

        if x.ndim != 2:
            raise RuntimeError('x is expected to be a 2-d array')

        if t.shape[0] != x.shape[0]:
            msg = f't has {t.shape[0]} time point(s), but x has {x.shape[0]}'
            raise RuntimeError(msg)

        self._t = t.copy()
        self._x = x.copy()
        self._dx = None if dx is None else dx.copy()

    def __getitem__(self, idx):
        dx = None

        if isinstance(idx, int):
            t = self._t[idx, np.newaxis]
            x = self._x[idx, :, np.newaxis]

            if self._dx is not None:
                dx = self._dx[idx, :, np.newaxis]
        else:
            t = self._t[idx]
            x = self._x[idx, :]

            if self._dx is not None:
                dx = self._dx[idx, :]

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
    """
    dataset = TimeSeriesDataset(
        data, window_size=window_size, window_order=window_order, dtype=dtype)
    dataloader = TorchDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
