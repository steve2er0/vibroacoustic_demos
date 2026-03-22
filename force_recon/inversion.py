"""Least-squares force reconstruction and conditioning diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.linalg import cond, lstsq, svd


def force_least_squares(
    A: np.ndarray,
    a: np.ndarray,
    *,
    rcond: float | None = None,
    tikhonov_lambda: float = 0.0,
) -> np.ndarray:
    """
    Solve A @ F ≈ a for F (3,) complex.

    A: (n_s, 3), a: (n_s,) complex.
    Optional Tikhonov: min ||A F - a||² + λ²||F||².
    """
    A = np.asarray(A, dtype=np.complex128)
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    if A.shape[0] != a.shape[0]:
        raise ValueError(f"A rows {A.shape[0]} != len(a) {a.shape[0]}")
    if A.shape[1] != 3:
        raise ValueError("A must have 3 columns (Fx, Fy, Fz)")
    if tikhonov_lambda > 0:
        lam = tikhonov_lambda
        A_aug = np.vstack([A, lam * np.eye(3, dtype=np.complex128)])
        a_aug = np.concatenate([a, np.zeros(3, dtype=np.complex128)])
        return lstsq(A_aug, a_aug, rcond=rcond)[0]
    return lstsq(A, a, rcond=rcond)[0]


def svd_diagnostics(A: np.ndarray) -> tuple[np.ndarray, float]:
    """Return singular values (descending) and 2-norm condition number."""
    A = np.asarray(A, dtype=np.complex128)
    if A.size == 0:
        return np.array([]), float("nan")
    s = svd(A, compute_uv=False)
    try:
        kappa = float(cond(A))
    except (np.linalg.LinAlgError, ValueError):
        kappa = float("inf")
    return s, kappa


def column_mac_matrix(A: np.ndarray) -> np.ndarray:
    """
    3x3 MAC between columns of A (complex).
    MAC_ij = |c_i^H c_j|^2 / (||c_i||^2 ||c_j||^2)
    """
    A = np.asarray(A, dtype=np.complex128)
    n = A.shape[1]
    mac = np.zeros((n, n), dtype=float)
    for i in range(n):
        ci = A[:, i]
        for j in range(n):
            cj = A[:, j]
            num = np.abs(np.vdot(ci, cj)) ** 2
            den = np.vdot(ci, ci).real * np.vdot(cj, cj).real
            mac[i, j] = num / den if den > 0 else 0.0
    return mac


def response_mac(a_meas: np.ndarray, a_pred: np.ndarray) -> float:
    """Scalar MAC between two complex vectors (full-sensor shapes)."""
    a_meas = np.asarray(a_meas, dtype=np.complex128).reshape(-1)
    a_pred = np.asarray(a_pred, dtype=np.complex128).reshape(-1)
    if a_meas.shape != a_pred.shape:
        raise ValueError("Shapes must match")
    num = np.abs(np.vdot(a_meas, a_pred)) ** 2
    den = np.vdot(a_meas, a_meas).real * np.vdot(a_pred, a_pred).real
    return float(num / den) if den > 0 else 0.0


def reconstruct_acceleration(A: np.ndarray, F: np.ndarray) -> np.ndarray:
    """a_pred = A @ F."""
    return A @ F
