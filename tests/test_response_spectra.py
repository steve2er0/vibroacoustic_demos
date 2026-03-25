"""Tests for pseudo PSD / CPSD from complex harmonic response lines."""

from pathlib import Path

import numpy as np

from force_recon.response_spectra import (
    auto_psd_from_complex_response,
    infer_uniform_df_hz,
    load_complex_response_csv,
    spectral_matrix_from_complex_response,
    write_auto_psd_csv,
    write_cpsd_csv,
)


def test_load_complex_response_csv_roundtrip(tmp_path: Path):
    path = tmp_path / "response.csv"
    path.write_text(
        "freq_hz,node1_t3_re,node1_t3_im,node2_t3_re,node2_t3_im\n"
        "10.0,1.0,2.0,3.0,4.0\n"
        "11.0,5.0,6.0,7.0,8.0\n"
    )

    freqs_hz, response_complex, names = load_complex_response_csv(path)

    assert np.allclose(freqs_hz, [10.0, 11.0])
    assert np.allclose(response_complex[:, 0], [1.0 + 2.0j, 5.0 + 6.0j])
    assert np.allclose(response_complex[:, 1], [3.0 + 4.0j, 7.0 + 8.0j])
    assert names == ["node1_t3", "node2_t3"]


def test_auto_psd_and_cpsd_scaling_peak():
    response = np.array([[2.0 + 0.0j, 0.0 + 2.0j]], dtype=np.complex128)

    auto = auto_psd_from_complex_response(response, df_hz=1.0, phasor_convention="peak")
    cpsd = spectral_matrix_from_complex_response(response, df_hz=1.0, phasor_convention="peak")

    assert np.allclose(auto, [[2.0, 2.0]])
    assert np.allclose(cpsd[0, 0, 1], -2.0j)
    assert np.allclose(cpsd[0, 1, 0], 2.0j)


def test_auto_psd_scaling_rms():
    response = np.array([[2.0 + 0.0j]], dtype=np.complex128)
    auto = auto_psd_from_complex_response(response, df_hz=0.5, phasor_convention="rms")
    assert np.allclose(auto, [[8.0]])


def test_infer_uniform_df_hz():
    freqs_hz = np.array([10.0, 10.5, 11.0, 11.5])
    assert infer_uniform_df_hz(freqs_hz) == 0.5


def test_write_auto_and_cpsd_csv(tmp_path: Path):
    freqs_hz = np.array([10.0, 11.0])
    response = np.array([[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]])
    auto = auto_psd_from_complex_response(response, df_hz=1.0, phasor_convention="rms")
    cpsd = spectral_matrix_from_complex_response(response, df_hz=1.0, phasor_convention="rms")

    auto_path = tmp_path / "auto.csv"
    cpsd_path = tmp_path / "cpsd.csv"
    write_auto_psd_csv(auto_path, freqs_hz, auto, ["a", "b"])
    write_cpsd_csv(cpsd_path, freqs_hz, cpsd, ["a", "b"])

    auto_text = auto_path.read_text().splitlines()
    cpsd_text = cpsd_path.read_text().splitlines()
    assert auto_text[0] == "freq_hz,a_psd,b_psd"
    assert cpsd_text[0] == "freq_hz,a__a_re,a__a_im,a__b_re,a__b_im,b__b_re,b__b_im"
