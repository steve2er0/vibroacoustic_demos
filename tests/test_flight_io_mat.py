"""Tests for per-channel MAT flight-data import."""

from pathlib import Path

import numpy as np
from scipy.io import savemat

from force_recon import frf_io, pipeline, units
from force_recon.flight_io import load_flight_data, load_flight_mat_files


def _write_channel_mat(path: Path, struct_name: str, amp_g: np.ndarray, time_s: np.ndarray, sr_hz: float) -> None:
    savemat(
        path,
        {
            struct_name: {
                "amp": np.asarray(amp_g, dtype=np.float64),
                "t": np.asarray(time_s, dtype=np.float64),
                "sr": float(sr_hz),
            }
        },
    )


def _write_triplet_csv(tmp_path, freqs, H, prefix="H"):
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


def test_load_flight_mat_files_roundtrip(tmp_path):
    fs = 100.0
    t = np.arange(16, dtype=np.float64) / fs
    amp0 = np.linspace(0.0, 1.0, t.size)
    amp1 = -amp0
    p0 = tmp_path / "sensor_a.mat"
    p1 = tmp_path / "sensor_b.mat"
    _write_channel_mat(p0, "sensor_a", amp0, t, fs)
    _write_channel_mat(p1, "sensor_b", amp1, t, fs)

    time_s, acc_g, names = load_flight_mat_files([p0, p1])

    assert np.allclose(time_s, t)
    assert np.allclose(acc_g[:, 0], amp0)
    assert np.allclose(acc_g[:, 1], amp1)
    assert names == ["sensor_a", "sensor_b"]


def test_build_ones_mobility_shape():
    fs = 200.0
    t = np.arange(32, dtype=np.float64) / fs
    freqs_hz, H = frf_io.build_ones_mobility(t, 5)

    assert freqs_hz.shape == (17,)
    assert H.shape == (17, 5, 3)
    assert np.all(H == 1.0 + 0.0j)


def test_pipeline_accepts_mat_flight_data(tmp_path):
    rng = np.random.default_rng(7)
    freqs = np.arange(20.0, 200.0, 5.0)
    nf, ns = len(freqs), 4
    H = (rng.standard_normal((nf, ns, 3)) + 1j * rng.standard_normal((nf, ns, 3))) * 1e-4
    _write_triplet_csv(tmp_path, freqs, H)

    px = tmp_path / "H_x.csv"
    py = tmp_path / "H_y.csv"
    pz = tmp_path / "H_z.csv"
    frf_f, H_load = frf_io.load_mobility_csv_triplet(px, py, pz)

    fs = 400.0
    n = 800
    t = np.arange(n) / fs
    t0, t1 = 0.5, 1.5
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
    mat_paths = []
    for ch_idx in range(ns):
        path = tmp_path / f"ch{ch_idx:02d}.mat"
        _write_channel_mat(path, f"ch{ch_idx:02d}", acc_g[:, ch_idx], t, fs)
        mat_paths.append(path)

    time_s, acc_loaded, names = load_flight_data(mat_paths)
    assert np.allclose(time_s, t)
    assert np.allclose(acc_loaded, acc_g)
    assert names == [f"ch{ch_idx:02d}" for ch_idx in range(ns)]

    res = pipeline.reconstruct_forces(
        time_s=time_s,
        acc_g=acc_loaded,
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
    k = int(np.argmin(np.abs(res.freqs_hz - 55.0)))
    if res.valid_mask[k]:
        rk = np.linalg.norm(res.a_pred_fft[k] - res.a_meas_fft[k]) / (
            np.linalg.norm(res.a_meas_fft[k]) + 1e-12
        )
        assert rk < 2.0
