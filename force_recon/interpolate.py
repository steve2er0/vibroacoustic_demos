"""Interpolate complex FRF / accelerance onto FFT frequency grid."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def interp_complex_frequency(
    f_target_hz: np.ndarray,
    f_source_hz: np.ndarray,
    values: np.ndarray,
    *,
    fill_value: complex = np.nan + 0j,
) -> np.ndarray:
    """
    values: (n_f_src, ...) complex; linear interp on real/imag per trailing element.

    Returns (len(f_target), ...) with NaN outside source range if fill_value is nan.
    """
    v = np.asarray(values, dtype=np.complex128)
    f_src = np.asarray(f_source_hz, dtype=np.float64)
    f_tgt = np.asarray(f_target_hz, dtype=np.float64)
    orig_shape = v.shape
    v2 = v.reshape(orig_shape[0], -1)
    n_out = f_tgt.shape[0]
    n_col = v2.shape[1]
    out = np.empty((n_out, n_col), dtype=np.complex128)
    for j in range(n_col):
        ir = interp1d(
            f_src,
            v2[:, j].real,
            bounds_error=False,
            fill_value=np.nan,
        )
        ii = interp1d(
            f_src,
            v2[:, j].imag,
            bounds_error=False,
            fill_value=np.nan,
        )
        out[:, j] = ir(f_tgt) + 1j * ii(f_tgt)
    out = out.reshape((n_out,) + orig_shape[1:])
    if not np.isnan(fill_value):
        out = np.where(np.isnan(out), fill_value, out)
    return out
