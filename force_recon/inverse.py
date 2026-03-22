"""Per-frequency force estimation: damped least squares."""

from __future__ import annotations

import numpy as np


def solve_force_complex(
    A: np.ndarray,
    a: np.ndarray,
    lam: float = 0.0,
) -> np.ndarray:
    """
    Solve A F ≈ a for F (3,) complex.

    Minimizes ||A F - a||^2 + lam^2 ||F||^2 when lam > 0 (Tikhonov).

    A: (n_s, 3), a: (n_s,)
    """
    A = np.asarray(A, dtype=np.complex128)
    a = np.asarray(a, dtype=np.complex128).ravel()
    AH = A.conj().T
    ATA = AH @ A
    ATa = AH @ a
    if lam > 0:
        ATA = ATA + (lam**2) * np.eye(3, dtype=np.complex128)
    try:
        return np.linalg.solve(ATA, ATa)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, a, rcond=None)[0]


def forward_acceleration(A: np.ndarray, F: np.ndarray) -> np.ndarray:
    """a_pred = A @ F."""
    return A @ np.asarray(F, dtype=np.complex128).ravel()


def response_mac(a_meas: np.ndarray, a_pred: np.ndarray) -> float:
    """MAC between two complex vectors (full-sensor stack at one frequency)."""
    x = np.asarray(a_meas, dtype=np.complex128).ravel()
    y = np.asarray(a_pred, dtype=np.complex128).ravel()
    nxx = np.real(np.vdot(x, x))
    nyy = np.real(np.vdot(y, y))
    if nxx <= 0 or nyy <= 0:
        return 0.0
    cross = np.vdot(x, y)
    return float(np.abs(cross) ** 2 / (nxx * nyy))
