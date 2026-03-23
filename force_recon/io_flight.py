"""Flight acceleration time series: load, slice, convert g → m/s², FFT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from force_recon.flight_io import load_flight_data as _load_flight_data
from force_recon.flight_io import load_flight_mat_files as _load_flight_mat_files
from force_recon.units import G0_STANDARD, g_to_m_s2


def load_flight_csv(
    path: str | Path,
    time_column: str = "time_s",
    channel_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    CSV with time in seconds and acceleration columns in g.

    Returns time_s, acc_g (n_samples, n_ch), channel_names
    """
    df = pd.read_csv(path)
    if time_column not in df.columns:
        raise KeyError(f"Missing {time_column}; columns={list(df.columns)}")
    t = df[time_column].to_numpy(dtype=float)
    if channel_columns is None:
        channel_columns = [c for c in df.columns if c != time_column]
    acc = df[channel_columns].to_numpy(dtype=float)
    return t, acc, channel_columns


def load_flight_mat_files(paths):
    """Compatibility wrapper for per-channel MAT flight inputs."""
    return _load_flight_mat_files(paths)


def load_flight_data(flight_input):
    """Compatibility wrapper for CSV or per-channel MAT flight inputs."""
    return _load_flight_data(flight_input)


def slice_time(
    t: np.ndarray,
    acc: np.ndarray,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return samples with t_start <= t <= t_end."""
    m = (t >= t_start) & (t <= t_end)
    return t[m], acc[m]


def complex_spectrum_rfft(
    acc_m_s2: np.ndarray,
    fs: float,
    window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided rFFT per column.

    Returns
    -------
    freqs_hz : (n_bins,) positive frequencies including 0
    spec : (n_bins, n_ch) complex spectrum (same scaling as numpy.fft.rfft)
    """
    x = np.asarray(acc_m_s2, dtype=float)
    n = x.shape[0]
    if window is not None:
        w = window.reshape(-1, 1) if x.ndim > 1 else window
        if x.ndim == 1:
            x = x * np.asarray(window)
        else:
            x = x * w
    if x.ndim == 1:
        spec = np.fft.rfft(x, axis=0)
    else:
        spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, np.asarray(spec, dtype=np.complex128)


def interpolate_complex_1d(
    f_src: np.ndarray,
    z_src: np.ndarray,
    f_tgt: np.ndarray,
) -> np.ndarray:
    """Linear interpolate complex samples along f_src onto f_tgt (real/imag)."""
    re = np.interp(f_tgt, f_src, z_src.real)
    im = np.interp(f_tgt, f_src, z_src.imag)
    return re + 1j * im


def interpolate_transfer_to_freq(
    freq_src_hz: np.ndarray,
    A_src: np.ndarray,
    freq_tgt_hz: np.ndarray,
) -> np.ndarray:
    """
    A_src: (n_src, n_s, 3). Return A at freq_tgt_hz: (n_tgt, n_s, 3).
    """
    n_tgt = len(freq_tgt_hz)
    n_s, nc = A_src.shape[1], A_src.shape[2]
    out = np.zeros((n_tgt, n_s, nc), dtype=np.complex128)
    for i in range(n_s):
        for j in range(nc):
            out[:, i, j] = interpolate_complex_1d(
                freq_src_hz, A_src[:, i, j], freq_tgt_hz
            )
    return out
