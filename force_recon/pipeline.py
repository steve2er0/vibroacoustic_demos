"""End-to-end force reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from force_recon import conditioning, inverse, units
from force_recon.interpolate import interp_complex_frequency


@dataclass
class ReconstructionResult:
    freqs_hz: np.ndarray
    """Analysis frequency bins (from flight rFFT)."""

    F_hat: np.ndarray
    """(n_f, 3) complex estimated forces Fx, Fy, Fz."""

    a_meas_fft: np.ndarray
    """(n_f, n_s) complex measured acceleration spectrum."""

    a_pred_fft: np.ndarray
    """(n_f, n_s) complex A @ F_hat."""

    cond_number: np.ndarray
    singular_values: np.ndarray
    """(n_f, 3) singular values."""

    mac_xy: np.ndarray
    mac_xz: np.ndarray
    mac_yz: np.ndarray

    response_mac: np.ndarray
    """MAC between full a_meas and a_pred per frequency."""

    relative_residual: np.ndarray
    """Relative residual ||a_pred - a_meas|| / ||a_meas|| per frequency."""

    valid_mask: np.ndarray
    """True where inversion was performed (finite A, omega > 0)."""


def reconstruct_forces(
    *,
    time_s: np.ndarray,
    acc_g: np.ndarray,
    t_start: float,
    t_end: float,
    frf_freqs_hz: np.ndarray,
    H_imperial: np.ndarray,
    g0: float = units.G0_STANDARD,
    tikhonov_lambda: float = 0.0,
    mobility_is_si: bool = False,
    skip_zero_hz: bool = True,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    fft_window: str = "hann",
) -> ReconstructionResult:
    """
    Parameters
    ----------
    time_s, acc_g
        Full flight record; acc_g shape (n,) or (n, n_s) in g.
    t_start, t_end
        Analysis window (seconds).
    frf_freqs_hz, H_imperial
        FRF grid and mobility H from NASTRAN: (n_f_frp, n_s, 3) in (in/s)/lbf
        unless mobility_is_si (then H is (m/s)/N).
    f_min_hz, f_max_hz
        Optional reconstruction band limits on FFT bins.
    fft_window
        Window passed to ``force_recon.flight.complex_rfft_matrix`` (e.g., ``"hann"``,
        ``"boxcar"``).
    """
    from force_recon import flight as fl

    t_full = np.asarray(time_s, dtype=np.float64).ravel()
    acc = np.asarray(acc_g, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)
    t_sel, acc_sel = fl.slice_time(t_full, acc, t_start, t_end)
    if len(t_sel) < 8:
        raise ValueError("Time slice too short for FFT")
    fs = fl.infer_fs(t_sel)
    acc_si = units.g_to_m_s2(acc_sel, g0)
    if f_min_hz is not None and f_max_hz is not None and f_max_hz < f_min_hz:
        raise ValueError("f_max_hz must be >= f_min_hz")

    freqs_fft, a_fft = fl.complex_rfft_matrix(acc_si, fs, window=fft_window)
    n_f, n_s = a_fft.shape

    if H_imperial.shape[1] != n_s:
        raise ValueError(
            f"H has {H_imperial.shape[1]} sensors, flight has {n_s} channels"
        )

    H = np.asarray(H_imperial, dtype=np.complex128)
    if not mobility_is_si:
        H = units.mobility_imp_to_si(H)

    omega_frp = 2 * np.pi * frf_freqs_hz
    A_frp = units.accelerance_from_mobility(H, omega_frp)

    A_fft = interp_complex_frequency(freqs_fft, frf_freqs_hz, A_frp)

    F_hat = np.zeros((n_f, 3), dtype=np.complex128)
    a_pred = np.zeros((n_f, n_s), dtype=np.complex128)
    cond = np.full(n_f, np.nan)
    svs = np.full((n_f, 3), np.nan)
    mac_xy = np.full(n_f, np.nan)
    mac_xz = np.full(n_f, np.nan)
    mac_yz = np.full(n_f, np.nan)
    rmac = np.full(n_f, np.nan)
    rel_resid = np.full(n_f, np.nan)
    valid = np.zeros(n_f, dtype=bool)

    for k in range(n_f):
        fk = freqs_fft[k]
        if skip_zero_hz and fk <= 0:
            continue
        if f_min_hz is not None and fk < f_min_hz:
            continue
        if f_max_hz is not None and fk > f_max_hz:
            continue
        Ak = A_fft[k]
        if np.any(~np.isfinite(Ak)):
            continue
        ak = a_fft[k]
        Mmac = conditioning.mac_column_matrix(Ak)
        pairs = conditioning.off_diagonal_mac_pairs(Mmac)
        mac_xy[k] = pairs.get("MAC_xy", np.nan)
        mac_xz[k] = pairs.get("MAC_xz", np.nan)
        mac_yz[k] = pairs.get("MAC_yz", np.nan)
        s = conditioning.singular_values(Ak)
        svs[k, : len(s)] = s
        cond[k] = conditioning.condition_number(Ak)
        F_hat[k] = inverse.solve_force_complex(Ak, ak, lam=tikhonov_lambda)
        a_pred[k] = inverse.forward_acceleration(Ak, F_hat[k])
        denom = np.linalg.norm(ak)
        rel_resid[k] = float(np.linalg.norm(a_pred[k] - ak) / denom) if denom > 0 else 0.0
        rmac[k] = inverse.response_mac(ak, a_pred[k])
        valid[k] = True

    return ReconstructionResult(
        freqs_hz=freqs_fft,
        F_hat=F_hat,
        a_meas_fft=a_fft,
        a_pred_fft=a_pred,
        cond_number=cond,
        singular_values=svs,
        mac_xy=mac_xy,
        mac_xz=mac_xz,
        mac_yz=mac_yz,
        response_mac=rmac,
        relative_residual=rel_resid,
        valid_mask=valid,
    )
