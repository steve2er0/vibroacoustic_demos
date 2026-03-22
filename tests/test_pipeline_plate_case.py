"""End-to-end check on a simply-supported plate synthetic dataset."""

import numpy as np

from force_recon import pipeline
from force_recon.simply_supported_plate import generate_simply_supported_plate_case


def test_simply_supported_plate_force_recovery():
    case = generate_simply_supported_plate_case(
        damping_ratio=0.012,
        m_max=6,
        n_max=6,
        fs_hz=1024.0,
        duration_s=4.0,
        frf_df_hz=1.0,
        noise_std_g=0.0,
    )

    res = pipeline.reconstruct_forces(
        time_s=case.time_s,
        acc_g=case.acc_g,
        t_start=case.t_start,
        t_end=case.t_end,
        frf_freqs_hz=case.frf_freqs_hz,
        H_imperial=case.H_imperial,
        tikhonov_lambda=1e-7,
        mobility_is_si=False,
        f_min_hz=5.0,
        f_max_hz=220.0,
        fft_window=case.fft_window,
    )

    active = case.force_active_mask
    m = res.valid_mask & active
    assert np.sum(m) >= 4

    f_true = case.true_force_fft[m]
    f_hat = res.F_hat[m]
    rel_force_err = np.linalg.norm(f_hat - f_true) / (np.linalg.norm(f_true) + 1e-14)
    assert rel_force_err < 0.05

    med_mac = float(np.nanmedian(res.response_mac[m]))
    med_rr = float(np.nanmedian(res.relative_residual[m]))
    assert med_mac > 0.99
    assert med_rr < 0.03
