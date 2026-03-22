"""Tests for force_recon.inversion (least squares + MAC)."""

import numpy as np
import pytest

from force_recon import inversion


def test_force_recovery_noiseless():
    rng = np.random.default_rng(0)
    n_s, n_f = 8, 50
    F_true = (rng.standard_normal((n_f, 3)) + 1j * rng.standard_normal((n_f, 3))) * 0.5
    A = rng.standard_normal((n_s, 3)) + 1j * rng.standard_normal((n_s, 3))
    freq = np.linspace(1, 100, n_f)
    a = np.zeros((n_f, n_s), dtype=np.complex128)
    for k in range(n_f):
        a[k] = A @ F_true[k]
    A_stack = np.zeros((n_f, n_s, 3), dtype=np.complex128)
    for k in range(n_f):
        A_stack[k] = A
    F_hat = np.zeros_like(F_true)
    for k in range(n_f):
        F_hat[k] = inversion.force_least_squares(A_stack[k], a[k])
    err = np.linalg.norm(F_hat - F_true) / np.linalg.norm(F_true)
    assert err < 1e-10


def test_mac_identity():
    A = np.array([[1 + 0j, 0], [0, 1]], dtype=np.complex128)
    mac = inversion.column_mac_matrix(A)
    assert np.allclose(mac, np.eye(2))


def test_response_mac_perfect():
    a = np.array([1 + 1j, 2 - 0.5j])
    assert inversion.response_mac(a, a) == pytest.approx(1.0)
