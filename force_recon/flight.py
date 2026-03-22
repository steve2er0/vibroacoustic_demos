"""Flight time series: slicing, g→SI, windowed rFFT for complex spectra."""

from __future__ import annotations

import numpy as np
from scipy import signal


def slice_time(
    time_s: np.ndarray,
    data_g: np.ndarray,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return time and data (n_ch,) or (n, n_ch) within [t_start, t_end]."""
    t = np.asarray(time_s, dtype=np.float64).ravel()
    mask = (t >= t_start) & (t <= t_end)
    t_sel = t[mask]
    d = np.asarray(data_g, dtype=np.float64)
    if d.ndim == 1:
        d_sel = d[mask]
    else:
        d_sel = d[mask, :]
    return t_sel, d_sel


def complex_rfft_matrix(
    acc_m_s2: np.ndarray,
    fs_hz: float,
    *,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Windowed rFFT per column. Same convention used for forward check.

    acc_m_s2: (n_samples, n_ch)
    Returns freqs_hz (n_bins,), spectrum (n_bins, n_ch) complex
    """
    x = np.asarray(acc_m_s2, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, n_ch = x.shape
    win = signal.windows.get_window(window, n)
    wsum = np.sum(win**2)
    scale = np.sqrt(n / wsum) if wsum > 0 else 1.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs_hz)
    out = np.zeros((len(freqs), n_ch), dtype=np.complex128)
    for j in range(n_ch):
        u = x[:, j] * win * scale
        out[:, j] = np.fft.rfft(u)
    return freqs, out


def welch_psd_1d(
    x: np.ndarray,
    fs_hz: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD (V^2/Hz if x is in V; here acceleration^2/Hz)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if nperseg is None:
        nperseg = min(256, max(32, n // 4))
    nperseg = min(nperseg, n)
    f, pxx = signal.welch(
        x,
        fs_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        return_onesided=True,
    )
    return f, pxx


def infer_fs(time_s: np.ndarray) -> float:
    t = np.asarray(time_s, dtype=np.float64).ravel()
    if len(t) < 2:
        raise ValueError("Need at least two time samples")
    dt = np.diff(t)
    med = float(np.median(dt))
    if med <= 0:
        raise ValueError("Non-increasing time")
    return 1.0 / med
