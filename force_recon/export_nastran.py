"""Export estimated spectra to NASTRAN-oriented text (TABRND1, TABLED1 snippets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
    lines = [f"TABRND1 {tid} {xaxis} {yaxis}"]
    for f, g in zip(freqs_hz, psd, strict=False):
        if np.isfinite(f) and np.isfinite(g) and g >= 0:
            lines.append(f"        {f:.6e} {g:.6e}")
    lines.append("ENDT")
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
    lines = [f"TABLED1 {tid_re}"]
    for fi, z in zip(f, F.real, strict=False):
        if np.isfinite(fi) and np.isfinite(z):
            lines.append(f"        {fi:.6e} {z:.6e}")
    lines.append("ENDT")
    lines.append(f"TABLED1 {tid_im}")
    for fi, z in zip(f, F.imag, strict=False):
        if np.isfinite(fi) and np.isfinite(z):
            lines.append(f"        {fi:.6e} {z:.6e}")
    lines.append("ENDT")
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
