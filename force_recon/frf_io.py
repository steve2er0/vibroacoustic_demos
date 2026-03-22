"""Load mobility / accelerance from CSV (three files: Fx, Fy, Fz columns)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_mobility_csv_triplet(
    path_x: str | Path,
    path_y: str | Path,
    path_z: str | Path,
    *,
    delimiter: str = ",",
    skiprows: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Each file: first column frequency (Hz), then pairs (re, im) per sensor in order.

    Returns
    -------
    freqs_hz : (n_f,)
    H : (n_f, n_s, 3) complex — columns are unit Fx, Fy, Fz load cases.
    """
    paths = [path_x, path_y, path_z]
    freqs = None
    stacks = []
    for p in paths:
        data = np.loadtxt(p, delimiter=delimiter, skiprows=skiprows)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        f = data[:, 0]
        rest = data[:, 1:]
        if rest.shape[1] % 2 != 0:
            raise ValueError(f"Expected even number of re/im columns in {p}")
        n_s = rest.shape[1] // 2
        c = rest[:, 0::2] + 1j * rest[:, 1::2]
        if freqs is None:
            freqs = f
        elif not np.allclose(freqs, f):
            raise ValueError(f"Frequency column mismatch for {p}")
        stacks.append(c)
    H = np.stack(stacks, axis=2)
    return freqs, H


def load_accelerance_csv_triplet(
    path_x: str | Path,
    path_y: str | Path,
    path_z: str | Path,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Same CSV layout but values are accelerance a/F (skip jω step)."""
    return load_mobility_csv_triplet(path_x, path_y, path_z, **kwargs)
