"""Synthetic simply-supported plate case for GUI checks and regression tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from force_recon import units
from force_recon.interpolate import interp_complex_frequency


@dataclass
class SimplySupportedPlateCase:
    """Self-consistent FRF + flight data generated from a modal plate model."""

    time_s: np.ndarray
    acc_g: np.ndarray
    channel_names: list[str]
    frf_freqs_hz: np.ndarray
    H_imperial: np.ndarray
    H_si: np.ndarray
    true_force_fft: np.ndarray
    force_active_mask: np.ndarray
    t_start: float
    t_end: float
    fft_window: str


def _fraction_points(frac_xy: list[tuple[float, float]], lx: float, ly: float) -> np.ndarray:
    p = np.array(frac_xy, dtype=np.float64)
    p[:, 0] *= lx
    p[:, 1] *= ly
    return p


def _modal_coupling(
    sensor_xy_m: np.ndarray,
    load_xy_m: np.ndarray,
    *,
    length_m: float,
    width_m: float,
    areal_mass_kg_m2: float,
    m_max: int,
    n_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ((m,n) mode index table, coupling tensor) for a simply supported plate."""
    modal_mass = areal_mass_kg_m2 * length_m * width_m / 4.0
    mode_indices: list[tuple[int, int]] = []
    couplings: list[np.ndarray] = []
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            phi_s = np.sin(m * np.pi * sensor_xy_m[:, 0] / length_m) * np.sin(
                n * np.pi * sensor_xy_m[:, 1] / width_m
            )
            phi_f = np.sin(m * np.pi * load_xy_m[:, 0] / length_m) * np.sin(
                n * np.pi * load_xy_m[:, 1] / width_m
            )
            couplings.append(np.outer(phi_s, phi_f) / modal_mass)
            mode_indices.append((m, n))
    mn = np.array(mode_indices, dtype=np.float64)
    return mn, np.array(couplings, dtype=np.complex128)


def generate_simply_supported_plate_case(
    *,
    length_m: float = 0.8,
    width_m: float = 0.6,
    thickness_m: float = 0.004,
    youngs_modulus_pa: float = 70.0e9,
    poisson_ratio: float = 0.33,
    density_kg_m3: float = 2700.0,
    damping_ratio: float = 0.01,
    m_max: int = 6,
    n_max: int = 6,
    fs_hz: float = 1024.0,
    duration_s: float = 4.0,
    frf_df_hz: float = 1.0,
    noise_std_g: float = 0.0,
    seed: int = 7,
) -> SimplySupportedPlateCase:
    """
    Create a deterministic synthetic case with 3 unknown force channels.

    The three channels correspond to three independent point-load cases on the plate
    (stored as Fx/Fy/Fz columns for compatibility with the existing toolchain).
    """
    if duration_s <= 0 or fs_hz <= 0:
        raise ValueError("duration_s and fs_hz must be positive")
    n_time = int(round(duration_s * fs_hz))
    if n_time < 32:
        raise ValueError("Need at least 32 time samples")
    t = np.arange(n_time, dtype=np.float64) / fs_hz
    f_fft = np.fft.rfftfreq(n_time, d=1.0 / fs_hz)

    sensor_xy_m = _fraction_points(
        [
            (0.14, 0.18),
            (0.31, 0.24),
            (0.49, 0.30),
            (0.68, 0.22),
            (0.19, 0.58),
            (0.38, 0.66),
            (0.57, 0.62),
            (0.76, 0.74),
        ],
        length_m,
        width_m,
    )
    load_xy_m = _fraction_points(
        [(0.24, 0.36), (0.53, 0.46), (0.71, 0.28)],
        length_m,
        width_m,
    )
    n_s = sensor_xy_m.shape[0]

    flexural_rigidity = (
        youngs_modulus_pa * thickness_m**3 / (12.0 * (1.0 - poisson_ratio**2))
    )
    areal_mass = density_kg_m3 * thickness_m
    mn, coupling = _modal_coupling(
        sensor_xy_m,
        load_xy_m,
        length_m=length_m,
        width_m=width_m,
        areal_mass_kg_m2=areal_mass,
        m_max=m_max,
        n_max=n_max,
    )
    m_idx = mn[:, 0]
    n_idx = mn[:, 1]
    omega_modes = np.sqrt(flexural_rigidity / areal_mass) * (
        (m_idx * np.pi / length_m) ** 2 + (n_idx * np.pi / width_m) ** 2
    )

    nyquist = fs_hz / 2.0
    frf_freqs_hz = np.arange(0.0, nyquist + 0.5 * frf_df_hz, frf_df_hz)
    H_si = np.zeros((len(frf_freqs_hz), n_s, 3), dtype=np.complex128)
    for k, freq in enumerate(frf_freqs_hz):
        omega = 2.0 * np.pi * freq
        den = omega_modes**2 - omega**2 + 1j * 2.0 * damping_ratio * omega_modes * omega
        receptance = np.tensordot(1.0 / den, coupling, axes=(0, 0))
        H_si[k] = 1j * omega * receptance

    H_imperial = H_si * (units.LBF_TO_N / units.IN_TO_M)

    tone_map = [
        (31.0, np.array([8.0 + 2.5j, -1.0 + 0.6j, 0.7 - 0.4j])),
        (59.0, np.array([-2.4 + 1.0j, 6.5 - 2.0j, 0.8 + 0.2j])),
        (97.0, np.array([1.2 - 0.8j, -1.0 + 0.3j, 7.2 + 1.6j])),
        (143.0, np.array([-1.1j, 1.8 - 0.7j, -4.8 + 1.9j])),
    ]
    true_force_fft = np.zeros((len(f_fft), 3), dtype=np.complex128)
    for f0, vec in tone_map:
        k = int(np.argmin(np.abs(f_fft - f0)))
        true_force_fft[k] += vec

    A_frf = units.accelerance_from_mobility(H_si, 2.0 * np.pi * frf_freqs_hz)
    A_fft = interp_complex_frequency(f_fft, frf_freqs_hz, A_frf, fill_value=0.0 + 0.0j)
    a_fft = np.zeros((len(f_fft), n_s), dtype=np.complex128)
    for k in range(len(f_fft)):
        a_fft[k] = A_fft[k] @ true_force_fft[k]

    acc_m_s2 = np.fft.irfft(a_fft, n=n_time, axis=0)
    acc_g = acc_m_s2 / units.G0_STANDARD
    if noise_std_g > 0:
        rng = np.random.default_rng(seed)
        acc_g = acc_g + rng.normal(scale=noise_std_g, size=acc_g.shape)

    force_active_mask = np.linalg.norm(true_force_fft, axis=1) > 0
    channel_names = [f"plate_s{i+1}" for i in range(n_s)]
    return SimplySupportedPlateCase(
        time_s=t,
        acc_g=acc_g,
        channel_names=channel_names,
        frf_freqs_hz=frf_freqs_hz,
        H_imperial=H_imperial,
        H_si=H_si,
        true_force_fft=true_force_fft,
        force_active_mask=force_active_mask,
        t_start=float(t[0]),
        t_end=float(t[-1]),
        fft_window="boxcar",
    )
