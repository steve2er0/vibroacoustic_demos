"""Load accelerance or mobility transfer matrices from CSV / NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_transfer_csv_x_y_z(
    path_x: str | Path,
    path_y: str | Path,
    path_z: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Three CSV files, one per unit force direction (Fx, Fy, Fz).

    Each CSV: first column `frequency_hz` (or `f`), then pairs
    `ch0_re`, `ch0_im`, `ch1_re`, `ch1_im`, ... for each sensor row.

    Returns
    -------
    frequency_hz : (n_f,)
    H_or_A : (n_f, n_s, 3) complex — columns are x, y, z loads.
    """
    paths = [Path(path_x), Path(path_y), Path(path_z)]
    dfs = [pd.read_csv(p) for p in paths]
    freq = dfs[0].iloc[:, 0].to_numpy(dtype=float)
    cols = [c for c in dfs[0].columns[1:]]
    if len(cols) % 2 != 0:
        raise ValueError("Expected re/im pairs after frequency column")
    n_s = len(cols) // 2
    n_f = len(freq)
    out = np.zeros((n_f, n_s, 3), dtype=np.complex128)
    for j, df in enumerate(dfs):
        if not np.allclose(df.iloc[:, 0].to_numpy(), freq):
            raise ValueError(f"Frequency grid mismatch: {paths[0]} vs {paths[j]}")
        for i in range(n_s):
            re_col = df.iloc[:, 1 + 2 * i].to_numpy(dtype=float)
            im_col = df.iloc[:, 2 + 2 * i].to_numpy(dtype=float)
            out[:, i, j] = re_col + 1j * im_col
    return freq, out


def load_transfer_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """NPZ with keys `frequency_hz` and `H` or `A` complex (n_f, n_s, 3)."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    freq = np.asarray(data["frequency_hz"], dtype=float)
    key = "A" if "A" in data.files else "H"
    arr = np.asarray(data[key], dtype=np.complex128)
    return freq, arr


def save_transfer_npz(
    path: str | Path,
    frequency_hz: np.ndarray,
    H_or_A: np.ndarray,
    *,
    name: str = "A",
) -> None:
    np.savez(path, frequency_hz=frequency_hz, **{name: H_or_A})
