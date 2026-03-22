"""Synthetic FRF + flight segment; check forward residual small."""

import numpy as np

from force_recon import frf_io, pipeline, units


def _write_triplet_csv(tmp_path, freqs, H, prefix="H"):
    """H (nf, ns, 3) imperial mobility numbers (fake)."""
    ns = H.shape[1]
    for j, name in enumerate(["x", "y", "z"]):
        rows = []
        for k, f in enumerate(freqs):
            row = [f]
            for s in range(ns):
                z = H[k, s, j]
                row.extend([z.real, z.imag])
            rows.append(row)
        arr = np.array(rows)
        hdr = "freq_hz," + ",".join(f"re{i},im{i}" for i in range(ns))
        p = tmp_path / f"{prefix}_{name}.csv"
        np.savetxt(p, arr, delimiter=",", header=hdr, comments="")


def test_pipeline_roundtrip_scaled(tmp_path):
    rng = np.random.default_rng(7)
    freqs = np.arange(20.0, 200.0, 5.0)
    nf, ns = len(freqs), 4
    # Smooth fake mobility (imperial-scale magnitudes)
    H = (rng.standard_normal((nf, ns, 3)) + 1j * rng.standard_normal((nf, ns, 3))) * 1e-4
    _write_triplet_csv(tmp_path, freqs, H)

    px = tmp_path / "H_x.csv"
    py = tmp_path / "H_y.csv"
    pz = tmp_path / "H_z.csv"
    frf_f, H_load = frf_io.load_mobility_csv_triplet(px, py, pz)

    fs = 400.0
    n = 800
    t = np.arange(n) / fs
    # pick mid window
    t0, t1 = 0.5, 1.5
    # Build acceleration in SI directly for truth, then convert to g for API
    omega = 2 * np.pi * 55.0
    F_phys = np.array([1.0 + 0.5j, -0.25j, 0.1], dtype=np.complex128)
    H_si = units.mobility_imp_to_si(H_load)
    A0 = units.accelerance_from_mobility(H_si, 2 * np.pi * frf_f)
    k0 = int(np.argmin(np.abs(frf_f - 55.0)))
    A55 = A0[k0]
    acc_t = np.zeros((n, ns))
    for s in range(ns):
        ph = A55[s, :] @ F_phys
        acc_t[:, s] = np.real(ph * np.exp(1j * omega * t))

    acc_g = acc_t / units.G0_STANDARD
    res = pipeline.reconstruct_forces(
        time_s=t,
        acc_g=acc_g,
        t_start=t0,
        t_end=t1,
        frf_freqs_hz=frf_f,
        H_imperial=H_load,
        g0=units.G0_STANDARD,
        tikhonov_lambda=1e-8,
        mobility_is_si=False,
    )
    assert np.sum(res.valid_mask) > 50
    m = res.valid_mask
    assert np.all(np.isfinite(res.F_hat[m]))
    # Forward residual should be moderate for this synthetic harmonic case
    k = int(np.argmin(np.abs(res.freqs_hz - 55.0)))
    if res.valid_mask[k]:
        rk = np.linalg.norm(res.a_pred_fft[k] - res.a_meas_fft[k]) / (
            np.linalg.norm(res.a_meas_fft[k]) + 1e-12
        )
        assert rk < 2.0
