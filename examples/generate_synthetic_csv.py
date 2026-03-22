#!/usr/bin/env python3
"""Generate synthetic flight + transfer CSVs for testing (run from repo root)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from force_recon.units import G0_STANDARD

OUT = Path(__file__).resolve().parent / "synthetic_data"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    fs = 4096.0
    t_end = 2.0
    t = np.arange(0, t_end, 1 / fs)
    n_s = 8
    n_f = len(np.fft.rfftfreq(len(t), d=1 / fs))
    freq = np.fft.rfftfreq(len(t), d=1 / fs)

    # Random complex accelerance A (n_f, n_s, 3) — SI, smooth
    A = np.zeros((n_f, n_s, 3), dtype=np.complex128)
    for i in range(n_s):
        for j in range(3):
            mag = 1e-3 * (1 + 0.1 * rng.standard_normal(n_f))
            ph = rng.uniform(0, 2 * np.pi, n_f)
            A[:, i, j] = mag * np.exp(1j * ph)

    # True force spectrum (smooth)
    F = np.zeros((n_f, 3), dtype=np.complex128)
    for j in range(3):
        F[:, j] = (rng.standard_normal(n_f) + 1j * rng.standard_normal(n_f)) * 5e-2

    a_spec = np.zeros((n_f, n_s), dtype=np.complex128)
    for k in range(n_f):
        a_spec[k] = A[k] @ F[k]

    # Time domain: irfft per channel (real signal)
    acc_m = np.zeros((len(t), n_s))
    for i in range(n_s):
        spec_i = np.zeros(n_f, dtype=np.complex128)
        for k in range(n_f):
            spec_i[k] = a_spec[k, i]
        acc_m[:, i] = np.fft.irfft(spec_i, n=len(t))

    acc_g = acc_m / G0_STANDARD

    df_flight = pd.DataFrame({"time_s": t})
    for i in range(n_s):
        df_flight[f"ch{i+1}"] = acc_g[:, i]
    df_flight.to_csv(OUT / "flight.csv", index=False)

    # Transfer CSVs: store accelerance (not mobility) for simplicity
    for j, name in enumerate(["fx", "fy", "fz"]):
        cols = {"frequency_hz": freq}
        for i in range(n_s):
            cols[f"ch{i}_re"] = A[:, i, j].real
            cols[f"ch{i}_im"] = A[:, i, j].imag
        pd.DataFrame(cols).to_csv(OUT / f"transfer_{name}.csv", index=False)

    print(f"Wrote {OUT}/flight.csv and transfer_fx|fy|fz.csv")


if __name__ == "__main__":
    main()
