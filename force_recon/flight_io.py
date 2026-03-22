"""Load flight CSV: time_s, ch0, ch1, ... (acceleration in g)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_flight_csv(path: str | Path, *, delimiter: str = ",") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    First row: time_s,col0,col1,... (header names optional but recommended).

    Returns time_s (n,), acc_g (n, n_ch), column_names (excluding time).
    """
    path = Path(path)
    with path.open() as f:
        header = f.readline().strip().split(delimiter)
    data = np.loadtxt(path, delimiter=delimiter, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t = data[:, 0]
    acc = data[:, 1:]
    names = header[1:] if len(header) > acc.shape[1] else [f"ch{i}" for i in range(acc.shape[1])]
    if len(names) != acc.shape[1]:
        names = [f"ch{i}" for i in range(acc.shape[1])]
    return t, acc, names
