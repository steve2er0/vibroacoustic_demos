"""Unit conversions for mobility H and acceleration."""

import numpy as np

# SI constants
LBF_TO_N = 4.4482216152605
IN_TO_M = 0.0254
# Standard gravity for g -> m/s^2
G0_STANDARD = 9.80665


def mobility_imp_to_si(H_imperial: np.ndarray) -> np.ndarray:
    """
    Convert mobility H from (in/s)/lbf to (m/s)/N.

    H_SI = H_imp * (m/in) / (N/lbf) = H_imp * IN_TO_M / LBF_TO_N
    """
    scale = IN_TO_M / LBF_TO_N
    return np.asarray(H_imperial, dtype=np.complex128) * scale


def accelerance_from_mobility(H: np.ndarray, omega_rad_s: np.ndarray) -> np.ndarray:
    """
    A = j * omega * H for H shaped (n_f, n_s, 3) and omega (n_f,) rad/s.
    """
    H = np.asarray(H, dtype=np.complex128)
    w = np.asarray(omega_rad_s, dtype=np.float64).reshape(-1, 1, 1)
    return 1j * w * H


def g_to_m_s2(acc_g: np.ndarray, g0: float = G0_STANDARD) -> np.ndarray:
    return np.asarray(acc_g, dtype=np.float64) * g0
