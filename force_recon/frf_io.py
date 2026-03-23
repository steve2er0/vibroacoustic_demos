"""Load mobility / accelerance from CSV or create dummy mobility arrays."""

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


def build_ones_mobility(
    time_s: np.ndarray,
    n_sensors: int,
    *,
    n_loads: int = 3,
    value: complex = 1.0 + 0.0j,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a constant dummy mobility tensor H = value for workflow plumbing.

    The frequency grid is derived from the supplied flight time vector so it spans
    0..Nyquist and can be interpolated onto any shorter analysis window FFT grid.
    """
    t = np.asarray(time_s, dtype=np.float64).ravel()
    if t.size < 2:
        raise ValueError("Need at least two time samples to build dummy mobility")
    dt = np.diff(t)
    med = float(np.median(dt))
    if med <= 0:
        raise ValueError("Time vector must be strictly increasing")
    fs_hz = 1.0 / med
    freqs_hz = np.fft.rfftfreq(t.size, d=1.0 / fs_hz)
    H = np.full((freqs_hz.size, int(n_sensors), int(n_loads)), complex(value), dtype=np.complex128)
    return freqs_hz, H
