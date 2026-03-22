"""Condition numbers, singular values, and column MAC for accelerance matrix A."""

from __future__ import annotations

import numpy as np


def singular_values(A: np.ndarray) -> np.ndarray:
    """Singular values of A (n_s x n_c), descending."""
    A = np.asarray(A, dtype=np.complex128)
    return np.linalg.svd(A, compute_uv=False)


def condition_number(A: np.ndarray) -> float:
    s = singular_values(A)
    if s[-1] <= 0 or not np.isfinite(s[-1]):
        return np.inf
    return float(s[0] / s[-1])


def mac_column_matrix(A: np.ndarray) -> np.ndarray:
    """
    MAC_ij = |c_i^H c_j|^2 / ((c_i^H c_i)(c_j^H c_j)) for columns of A.

    A: (n_s, n_c), typically n_c = 3.
    """
    A = np.asarray(A, dtype=np.complex128)
    n_s, n_c = A.shape
    M = np.zeros((n_c, n_c), dtype=np.float64)
    for i in range(n_c):
        ci = A[:, i]
        nii = np.real(np.vdot(ci, ci))
        if nii <= 0:
            continue
        for j in range(n_c):
            cj = A[:, j]
            njj = np.real(np.vdot(cj, cj))
            if njj <= 0:
                continue
            cross = np.vdot(ci, cj)
            M[i, j] = float(np.abs(cross) ** 2 / (nii * njj))
    return M


def off_diagonal_mac_pairs(M: np.ndarray) -> dict[str, float]:
    """For 3x3 MAC, return MAC_xy, MAC_xz, MAC_yz (indices 0,1,2)."""
    if M.shape[0] < 3:
        return {}
    return {
        "MAC_xy": float(M[0, 1]),
        "MAC_xz": float(M[0, 2]),
        "MAC_yz": float(M[1, 2]),
    }
