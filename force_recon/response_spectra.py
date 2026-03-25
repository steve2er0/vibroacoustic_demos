"""Utilities for PSD / CPSD estimates from complex harmonic response spectra."""

from __future__ import annotations

from pathlib import Path
import csv
import re

import numpy as np


def load_complex_response_csv(
    path: str | Path,
    *,
    delimiter: str = ",",
    skiprows: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a complex response CSV with columns:
    ``freq_hz,re0,im0,re1,im1,...`` or named ``<name>_re,<name>_im`` pairs.

    Returns
    -------
    freqs_hz
        ``(n_f,)`` frequency vector in Hz.
    response_complex
        ``(n_f, n_ch)`` complex response matrix.
    channel_names
        Parsed channel names for the re/im pairs.
    """
    path = Path(path)
    header = _read_header(path, delimiter=delimiter)
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError("Expected frequency column followed by at least one re/im pair.")

    freqs_hz = np.asarray(data[:, 0], dtype=np.float64)
    pair_cols = data[:, 1:]
    if pair_cols.shape[1] % 2 != 0:
        raise ValueError("Expected an even number of re/im columns after frequency.")

    response_complex = pair_cols[:, 0::2] + 1j * pair_cols[:, 1::2]
    channel_names = _derive_channel_names(header, response_complex.shape[1])
    return freqs_hz, response_complex, channel_names


def infer_uniform_df_hz(freqs_hz: np.ndarray) -> float:
    """Infer the constant frequency spacing in Hz."""
    f = np.asarray(freqs_hz, dtype=np.float64).ravel()
    if f.size < 2:
        raise ValueError("Need at least two frequency samples to infer df.")
    df = np.diff(f)
    df0 = float(df[0])
    atol = max(1e-12, 1e-9 * max(np.max(np.abs(f)), 1.0))
    if not np.allclose(df, df0, rtol=1e-6, atol=atol):
        raise ValueError("Frequency vector is not uniformly spaced; supply df_hz explicitly.")
    if df0 <= 0:
        raise ValueError("Frequency vector must be strictly increasing.")
    return df0


def spectral_matrix_from_complex_response(
    response_complex: np.ndarray,
    *,
    df_hz: float,
    phasor_convention: str = "peak",
) -> np.ndarray:
    """
    Build a one-sided spectral matrix from complex harmonic response lines.

    Parameters
    ----------
    response_complex
        ``(n_f, n_ch)`` complex response lines.
    df_hz
        Frequency spacing of the response lines.
    phasor_convention
        ``"peak"`` if the complex response magnitude is peak amplitude,
        ``"rms"`` if it is already RMS amplitude.

    Returns
    -------
    cpsd
        ``(n_f, n_ch, n_ch)`` complex spectral matrix with units
        ``(response units)^2 / Hz``.
    """
    X = np.asarray(response_complex, dtype=np.complex128)
    if X.ndim != 2:
        raise ValueError("response_complex must have shape (n_freq, n_ch).")
    if not np.isfinite(df_hz) or df_hz <= 0:
        raise ValueError("df_hz must be a positive finite scalar.")

    convention = phasor_convention.strip().lower()
    if convention == "peak":
        power_scale = 0.5
    elif convention == "rms":
        power_scale = 1.0
    else:
        raise ValueError("phasor_convention must be 'peak' or 'rms'.")

    return power_scale * X[:, :, None] * np.conjugate(X[:, None, :]) / float(df_hz)


def auto_psd_from_complex_response(
    response_complex: np.ndarray,
    *,
    df_hz: float,
    phasor_convention: str = "peak",
) -> np.ndarray:
    """Return the diagonal auto-PSD channels from the complex response lines."""
    cpsd = spectral_matrix_from_complex_response(
        response_complex, df_hz=df_hz, phasor_convention=phasor_convention
    )
    return np.real(np.diagonal(cpsd, axis1=1, axis2=2))


def write_auto_psd_csv(
    path: str | Path,
    freqs_hz: np.ndarray,
    auto_psd: np.ndarray,
    channel_names: list[str],
) -> None:
    """Write auto-PSD channels to CSV."""
    path = Path(path)
    P = np.asarray(auto_psd, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError("auto_psd must have shape (n_freq, n_ch).")
    if P.shape[1] != len(channel_names):
        raise ValueError("channel_names must match the PSD channel count.")

    header = ["freq_hz"] + [f"{_sanitize_name(name)}_psd" for name in channel_names]
    rows = np.column_stack([np.asarray(freqs_hz, dtype=np.float64), P])
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="")


def write_cpsd_csv(
    path: str | Path,
    freqs_hz: np.ndarray,
    cpsd: np.ndarray,
    channel_names: list[str],
) -> None:
    """Write upper-triangle CPSD terms to CSV as re/im pairs."""
    path = Path(path)
    S = np.asarray(cpsd, dtype=np.complex128)
    if S.ndim != 3 or S.shape[1] != S.shape[2]:
        raise ValueError("cpsd must have shape (n_freq, n_ch, n_ch).")
    if S.shape[1] != len(channel_names):
        raise ValueError("channel_names must match the CPSD channel count.")

    header = ["freq_hz"]
    cols = [np.asarray(freqs_hz, dtype=np.float64)]
    n_ch = S.shape[1]
    for i in range(n_ch):
        for j in range(i, n_ch):
            pair_name = f"{_sanitize_name(channel_names[i])}__{_sanitize_name(channel_names[j])}"
            header.extend([f"{pair_name}_re", f"{pair_name}_im"])
            cols.extend([S[:, i, j].real, S[:, i, j].imag])
    rows = np.column_stack(cols)
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="")


def _read_header(path: Path, *, delimiter: str) -> list[str]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        row = next(reader)
    return [cell.strip().lstrip("#").strip() for cell in row]


def _derive_channel_names(header: list[str], n_ch: int) -> list[str]:
    if len(header) < 1 + 2 * n_ch:
        return [f"ch{i}" for i in range(n_ch)]

    names: list[str] = []
    for idx in range(n_ch):
        re_col = header[1 + 2 * idx]
        im_col = header[2 + 2 * idx]
        names.append(_pair_name_from_columns(re_col, im_col, idx))
    return names


def _pair_name_from_columns(re_col: str, im_col: str, idx: int) -> str:
    re_match = re.match(r"^(.*)_re$", re_col, flags=re.IGNORECASE)
    im_match = re.match(r"^(.*)_im$", im_col, flags=re.IGNORECASE)
    if re_match and im_match and re_match.group(1) == im_match.group(1) and re_match.group(1):
        return re_match.group(1)

    if re.fullmatch(r"re\d+", re_col, flags=re.IGNORECASE) and re.fullmatch(r"im\d+", im_col, flags=re.IGNORECASE):
        return f"ch{idx}"

    return f"ch{idx}"


def _sanitize_name(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "ch"
