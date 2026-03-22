"""Shared plot logic for Tk and macOS Matplotlib GUIs."""

from __future__ import annotations

import numpy as np

from force_recon import units
from force_recon.export_nastran import psd_from_force_fft
from force_recon.flight import welch_psd_1d


def draw_spectra(
    ax_fx,
    ax_ax,
    res,
    time_s: np.ndarray,
    acc_g: np.ndarray,
    ch_i: int,
    ch_names: list[str],
    t0: float,
    t1: float,
    g0: float,
) -> None:
    mask = (time_s >= t0) & (time_s <= t1)
    seg = acc_g[mask, ch_i]
    fs_seg = 1.0 / np.median(np.diff(time_s[mask]))
    n_seg = int(np.sum(mask))

    m = res.valid_mask
    f_hz = res.freqs_hz[m]

    Sff = np.zeros((len(res.freqs_hz), 3))
    for j in range(3):
        Sff[:, j] = psd_from_force_fft(res.F_hat[:, j], res.freqs_hz, fs_seg, n_seg)

    ax_fx.clear()
    for j, name in enumerate(["Fx", "Fy", "Fz"]):
        ax_fx.semilogy(f_hz, np.maximum(Sff[m, j], 1e-30), label=name)
    ax_fx.set_xlabel("Hz")
    ax_fx.set_ylabel("N²/Hz (approx)")
    ax_fx.set_title("Reconstructed force PSD")
    ax_fx.legend()
    ax_fx.grid(True, alpha=0.3)

    f_w, p_meas = welch_psd_1d(units.g_to_m_s2(seg, g0), fs_seg)
    a_pred_t = np.fft.irfft(res.a_pred_fft[:, ch_i], n=n_seg)
    f_w2, p_pred = welch_psd_1d(
        np.real(a_pred_t),
        fs_seg,
        nperseg=min(256, max(32, n_seg // 4)),
    )
    ax_ax.clear()
    ax_ax.loglog(f_w, np.maximum(p_meas, 1e-30), label="meas")
    ax_ax.loglog(f_w2, np.maximum(p_pred, 1e-30), label="pred")
    ax_ax.set_xlabel("Hz")
    ax_ax.set_ylabel("(m/s²)²/Hz")
    ax_ax.set_title(f"Acceleration PSD — {ch_names[ch_i]}")
    ax_ax.legend()
    ax_ax.grid(True, which="both", alpha=0.3)


def draw_conditioning(ax_list, res) -> None:
    m = res.valid_mask
    f_hz = res.freqs_hz[m]
    ax_list[0].clear()
    ax_list[0].semilogy(f_hz, np.maximum(res.cond_number[m], 1e-30))
    ax_list[0].set_title("κ(A)")
    ax_list[0].set_xlabel("Hz")
    ax_list[0].grid(True, alpha=0.3)

    ax_list[1].clear()
    ax_list[1].plot(f_hz, res.singular_values[m, 2])
    ax_list[1].set_title("σ₃")
    ax_list[1].set_xlabel("Hz")
    ax_list[1].grid(True, alpha=0.3)

    ax_list[2].clear()
    ax_list[2].plot(f_hz, res.mac_xy[m], label="MAC_xy")
    ax_list[2].plot(f_hz, res.mac_xz[m], label="MAC_xz")
    ax_list[2].plot(f_hz, res.mac_yz[m], label="MAC_yz")
    ax_list[2].set_ylim(-0.05, 1.05)
    ax_list[2].set_title("Column MAC")
    ax_list[2].set_xlabel("Hz")
    ax_list[2].legend(fontsize=8)
    ax_list[2].grid(True, alpha=0.3)

    ax_list[3].clear()
    ax_list[3].plot(f_hz, res.response_mac[m])
    ax_list[3].set_ylim(-0.05, 1.05)
    ax_list[3].set_title("Response MAC")
    ax_list[3].set_xlabel("Hz")
    ax_list[3].grid(True, alpha=0.3)
