"""Integration: synthetic CSV (accelerance) → mobility → pipeline."""

from pathlib import Path

import numpy as np
import pytest

from force_recon.io_flight import load_flight_csv, slice_time
from force_recon.io_transfer import load_transfer_csv_x_y_z
from force_recon.pipeline import reconstruct_forces
from force_recon.units import G0_STANDARD


def test_synthetic_example_files():
    d = Path(__file__).resolve().parent.parent / "examples" / "synthetic_data"
    if not (d / "flight.csv").exists():
        pytest.skip("Run: python3 examples/generate_synthetic_csv.py")

    t, acc_g, _ch = load_flight_csv(d / "flight.csv")
    freq_t, A_stack = load_transfer_csv_x_y_z(
        d / "transfer_fx.csv",
        d / "transfer_fy.csv",
        d / "transfer_fz.csv",
    )
    omega = 2 * np.pi * freq_t
    w_safe = np.where(np.abs(omega) < 1e-12, 1.0, omega)
    H_si = A_stack / (1j * w_safe[:, np.newaxis, np.newaxis])

    t_s, acc_s = slice_time(t, acc_g, float(t[0]), float(t[-1]))
    res = reconstruct_forces(
        time_s=t_s,
        acc_g=acc_s,
        t_start=float(t_s[0]),
        t_end=float(t_s[-1]),
        frf_freqs_hz=freq_t,
        H_imperial=H_si,
        g0=G0_STANDARD,
        tikhonov_lambda=0.0,
        mobility_is_si=True,
    )
    assert res.F_hat.shape[1] == 3
    m = res.valid_mask
    assert np.sum(m) > 10
    assert np.all(np.isfinite(res.cond_number[m]))
