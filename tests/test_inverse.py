import numpy as np

from force_recon import conditioning, inverse


def test_recover_force_noiseless():
    rng = np.random.default_rng(42)
    for _ in range(5):
        A = (rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))) * 0.5
        F_true = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        a = A @ F_true
        F_hat = inverse.solve_force_complex(A, a, lam=0.0)
        np.testing.assert_allclose(F_hat, F_true, rtol=1e-5, atol=1e-6)


def test_tikhonov_reduces_norm_when_illposed():
    rng = np.random.default_rng(0)
    A = np.ones((8, 3), dtype=np.complex128)  # rank 1 columns identical
    a = np.ones(8, dtype=np.complex128)
    F0 = inverse.solve_force_complex(A, a, lam=0.0)
    F1 = inverse.solve_force_complex(A, a, lam=1e2)
    assert np.linalg.norm(F1) <= np.linalg.norm(F0) + 1e-9


def test_mac_diagonal_one():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 3)) + 1j * rng.standard_normal((6, 3))
    M = conditioning.mac_column_matrix(A)
    for i in range(3):
        assert abs(M[i, i] - 1.0) < 1e-9
