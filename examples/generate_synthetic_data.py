#!/usr/bin/env python3
"""Write example mobility + flight CSVs under examples/data/ for the Streamlit app."""

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from force_recon import units

ROOT = Path(__file__).resolve().parent / "data"
NS = 8
FREQS = np.arange(20.0, 2001.0, 1.0)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2025)
    nf = len(FREQS)
    H = (rng.standard_normal((nf, NS, 3)) + 1j * rng.standard_normal((nf, NS, 3))) * 5e-5

    def write_axis(j: int, name: str) -> None:
        rows = []
        for k, fhz in enumerate(FREQS):
            row = [fhz]
            for s in range(NS):
                z = H[k, s, j]
                row.extend([z.real, z.imag])
            rows.append(row)
        arr = np.array(rows)
        hdr = "freq_hz," + ",".join(f"re{s},im{s}" for s in range(NS))
        np.savetxt(ROOT / f"mobility_{name}.csv", arr, delimiter=",", header=hdr, comments="")

    write_axis(0, "Fx")
    write_axis(1, "Fy")
    write_axis(2, "Fz")

    fs = 4096.0
    n = int(fs * 4)
    t = np.arange(n) / fs
    omega = 2 * np.pi * 127.0
    H_si = units.mobility_imp_to_si(H)
    kf = int(np.argmin(np.abs(FREQS - 127.0)))
    A127 = units.accelerance_from_mobility(H_si[None, kf], np.array([2 * np.pi * 127.0]))[0]
    F = np.array([0.8 + 0.1j, -0.2j, 0.05], dtype=np.complex128)
    acc = np.zeros((n, NS))
    for s in range(NS):
        acc[:, s] = np.real((A127[s, :] @ F) * np.exp(1j * omega * t))
    acc_g = acc / units.G0_STANDARD
    cols = ["time_s"] + [f"ch{i}" for i in range(NS)]
    out = np.column_stack([t, acc_g])
    hdr = ",".join(cols)
    np.savetxt(ROOT / "flight_segment.csv", out, delimiter=",", header=hdr, comments="")
    print(f"Wrote {ROOT / 'mobility_Fx.csv'}, …, {ROOT / 'flight_segment.csv'}")


if __name__ == "__main__":
    main()
