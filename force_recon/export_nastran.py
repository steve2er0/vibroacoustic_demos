"""Export estimated spectra to NASTRAN-oriented text (TABRND1, TABLED1 snippets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _fmt_real(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("NASTRAN export only supports finite real values.")
    return f"{value:.16E}"


def write_force_spectrum_csv(
    freqs_hz: np.ndarray,
    F_hat: np.ndarray,
    path: str | Path,
    *,
    valid_mask: np.ndarray | None = None,
) -> None:
    """
    CSV columns: freq_hz, Fx_re, Fx_im, Fy_re, Fy_im, Fz_re, Fz_im [, valid]
    F in Newtons (complex phasor consistent with reconstruction).
    """
    path = Path(path)
    m = valid_mask if valid_mask is not None else np.ones(len(freqs_hz), dtype=bool)
    rows = []
    for k in range(len(freqs_hz)):
        fx, fy, fz = F_hat[k]
        rows.append(
            [
                freqs_hz[k],
                fx.real,
                fx.imag,
                fy.real,
                fy.imag,
                fz.real,
                fz.imag,
                int(m[k]),
            ]
        )
    arr = np.array(rows)
    header = "freq_hz,Fx_re,Fx_im,Fy_re,Fy_im,Fz_re,Fz_im,valid"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def write_reconstruction_diagnostics_csv(
    freqs_hz: np.ndarray,
    F_hat: np.ndarray,
    cond_number: np.ndarray,
    singular_values: np.ndarray,
    mac_xy: np.ndarray,
    mac_xz: np.ndarray,
    mac_yz: np.ndarray,
    response_mac: np.ndarray,
    relative_residual: np.ndarray,
    valid_mask: np.ndarray,
    path: str | Path,
) -> None:
    """Write a per-frequency diagnostics table for GUI / post-processing."""
    f = np.asarray(freqs_hz, dtype=np.float64).ravel()
    F = np.asarray(F_hat, dtype=np.complex128)
    s = np.asarray(singular_values, dtype=np.float64)
    cols = np.column_stack(
        [
            f,
            F[:, 0].real,
            F[:, 0].imag,
            F[:, 1].real,
            F[:, 1].imag,
            F[:, 2].real,
            F[:, 2].imag,
            np.asarray(cond_number, dtype=np.float64).ravel(),
            s[:, 0],
            s[:, 1],
            s[:, 2],
            np.asarray(mac_xy, dtype=np.float64).ravel(),
            np.asarray(mac_xz, dtype=np.float64).ravel(),
            np.asarray(mac_yz, dtype=np.float64).ravel(),
            np.asarray(response_mac, dtype=np.float64).ravel(),
            np.asarray(relative_residual, dtype=np.float64).ravel(),
            np.asarray(valid_mask, dtype=np.int64).ravel(),
        ]
    )
    header = ",".join(
        [
            "freq_hz",
            "Fx_re",
            "Fx_im",
            "Fy_re",
            "Fy_im",
            "Fz_re",
            "Fz_im",
            "cond_number",
            "sigma1",
            "sigma2",
            "sigma3",
            "mac_xy",
            "mac_xz",
            "mac_yz",
            "response_mac",
            "relative_residual",
            "valid",
        ]
    )
    np.savetxt(Path(path), cols, delimiter=",", header=header, comments="")


def tabrnd1_snippet(
    tid: int,
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    *,
    xaxis: int = 1,
    yaxis: int = 1,
) -> str:
    """
    Minimal TABRND1 block (check field widths against your NASTRAN QRG).

    psd: one-sided force PSD (N^2/Hz or lbf^2/Hz) same length as freqs.
    """
    lines = [f"TABRND1,{tid},{xaxis},{yaxis}"]
    valid_rows: list[str] = []
    for f, g in zip(freqs_hz, psd):
        if np.isfinite(f) and np.isfinite(g) and g >= 0:
            valid_rows.append(f"+,{_fmt_real(float(f))},{_fmt_real(float(g))}")
    if valid_rows:
        valid_rows[-1] += ",ENDT"
        lines.extend(valid_rows)
    else:
        lines.append("+,SKIP,SKIP,ENDT")
    return "\n".join(lines)


def tabled1_re_im_snippets(
    tid_re: int,
    tid_im: int,
    freqs_hz: np.ndarray,
    F_complex: np.ndarray,
    *,
    scale: float = 1.0,
) -> str:
    """Two TABLED1 tables for real and imaginary part of scalar F(f)."""
    F = np.asarray(F_complex, dtype=np.complex128).ravel() * scale
    f = np.asarray(freqs_hz, dtype=np.float64).ravel()
    lines = [f"TABLED1,{tid_re},LINEAR,LINEAR"]
    real_rows: list[str] = []
    for fi, z in zip(f, F.real):
        if np.isfinite(fi) and np.isfinite(z):
            real_rows.append(f"+,{_fmt_real(float(fi))},{_fmt_real(float(z))}")
    if real_rows:
        real_rows[-1] += ",ENDT"
        lines.extend(real_rows)
    else:
        lines.append("+,SKIP,SKIP,ENDT")
    lines.append(f"TABLED1,{tid_im},LINEAR,LINEAR")
    imag_rows: list[str] = []
    for fi, z in zip(f, F.imag):
        if np.isfinite(fi) and np.isfinite(z):
            imag_rows.append(f"+,{_fmt_real(float(fi))},{_fmt_real(float(z))}")
    if imag_rows:
        imag_rows[-1] += ",ENDT"
        lines.extend(imag_rows)
    else:
        lines.append("+,SKIP,SKIP,ENDT")
    return "\n".join(lines)


def psd_from_force_fft(
    F_hat: np.ndarray,
    freqs_hz: np.ndarray,
    fs_hz: float,
    n_time: int,
) -> np.ndarray:
    """
    Rough one-sided force PSD from |F|^2 / equivalent noise bandwidth.

    Uses effective resolution df = fs/n_time for scaling (same order as periodogram).
    """
    df = fs_hz / max(n_time, 1)
    F = np.asarray(F_hat, dtype=np.complex128)
    return (np.abs(F) ** 2) / df
