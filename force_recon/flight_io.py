"""Load flight CSV or per-channel MAT files (acceleration in g)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
from scipy.io import loadmat


def load_flight_csv(path: str | Path, *, delimiter: str = ",") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    First row: time_s,col0,col1,... (header names optional but recommended).

    Returns time_s (n,), acc_g (n, n_ch), column_names (excluding time).
    """
    path = Path(path)
    with path.open() as f:
        header = f.readline().strip().split(delimiter)
    data = np.loadtxt(path, delimiter=delimiter, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t = np.asarray(data[:, 0], dtype=np.float64)
    acc = np.asarray(data[:, 1:], dtype=np.float64)
    names = header[1:] if len(header) > acc.shape[1] else [f"ch{i}" for i in range(acc.shape[1])]
    if len(names) != acc.shape[1]:
        names = [f"ch{i}" for i in range(acc.shape[1])]
    return t, acc, names


def load_flight_mat_files(
    paths: Sequence[str | Path | BinaryIO],
    *,
    channel_names: Sequence[str] | None = None,
    source_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load per-channel MAT files containing one structure with amp, t, sr fields.

    Each MAT file corresponds to one acceleration channel in g. The file order must
    match the FRF sensor order. Channels are aligned onto their common overlapping
    time span and linearly resampled onto a shared uniform grid using the slowest
    channel sample rate.
    """
    if isinstance(paths, (str, Path)) or hasattr(paths, "read"):
        items = [paths]
    else:
        items = list(paths)
    if not items:
        raise ValueError("Need at least one MAT file")
    if channel_names is not None and len(channel_names) != len(items):
        raise ValueError("channel_names length must match number of MAT files")
    if source_names is not None and len(source_names) != len(items):
        raise ValueError("source_names length must match number of MAT files")

    amp_cols: list[np.ndarray] = []
    time_cols: list[np.ndarray] = []
    sr_values: list[float] = []
    names: list[str] = []
    source_labels: list[str] = []

    for idx, item in enumerate(items):
        source_label = source_names[idx] if source_names is not None else _flight_source_label(item, idx)
        amp_g, time_s, sr_hz = _load_single_channel_mat(item, source_label)
        amp_cols.append(amp_g)
        time_cols.append(time_s)
        sr_values.append(sr_hz)
        source_labels.append(source_label)
        if channel_names is not None:
            names.append(str(channel_names[idx]))
        else:
            names.append(_flight_channel_name(item, idx))

    time_aligned, acc = _align_channels_on_common_timebase(amp_cols, time_cols, sr_values, source_labels)
    return time_aligned, acc, names


def load_flight_data(
    flight_input: str | Path | Sequence[str | Path | BinaryIO],
    *,
    delimiter: str = ",",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load flight data from either a CSV path or ordered per-channel MAT files."""
    if hasattr(flight_input, "read"):
        return load_flight_mat_files([flight_input])
    if isinstance(flight_input, (str, Path)):
        path = Path(flight_input)
        if path.suffix.lower() == ".mat":
            return load_flight_mat_files([path])
        return load_flight_csv(path, delimiter=delimiter)

    items = list(flight_input)
    if not items:
        raise ValueError("Flight input sequence is empty")
    if len(items) == 1 and isinstance(items[0], (str, Path)):
        path = Path(items[0])
        if path.suffix.lower() != ".mat":
            return load_flight_csv(path, delimiter=delimiter)
    return load_flight_mat_files(items)


def _load_single_channel_mat(
    source: str | Path | BinaryIO,
    source_label: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        mat = loadmat(source, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    except NotImplementedError as exc:
        raise ValueError(
            f"{source_label}: unsupported MAT version. Resave without -v7.3 or load via CSV."
        ) from exc
    except Exception as exc:
        raise ValueError(f"{source_label}: could not read MAT file ({exc})") from exc

    candidates = [
        value
        for key, value in mat.items()
        if not key.startswith("__") and _has_required_mat_fields(value)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{source_label}: expected exactly one structure with amp, t, sr fields"
        )

    record = candidates[0]
    amp = np.asarray(_mat_field(record, "amp"), dtype=np.float64).ravel()
    if amp.size == 0 or not np.all(np.isfinite(amp)):
        raise ValueError(f"{source_label}: amp must be a finite numeric vector")

    sr_array = np.asarray(_mat_field(record, "sr"), dtype=np.float64).reshape(-1)
    if sr_array.size != 1:
        raise ValueError(f"{source_label}: sr must be a scalar")
    sr = float(sr_array[0])
    if not np.isfinite(sr) or sr <= 0:
        raise ValueError(f"{source_label}: sr must be a positive finite scalar")

    t_raw = np.asarray(_mat_field(record, "t"), dtype=np.float64).ravel()
    t = t_raw if t_raw.size > 0 else np.arange(amp.size, dtype=np.float64) / sr
    if t.size != amp.size or not np.all(np.isfinite(t)):
        raise ValueError(f"{source_label}: t must be a finite vector matching amp length")
    if t.size > 1:
        sr_inferred = _infer_fs(t, source_label)
        tol = max(1e-9, 1e-6 * max(abs(sr), 1.0))
        if abs(sr_inferred - sr) > tol:
            raise ValueError(f"{source_label}: sr does not match the spacing implied by t")
    return amp, t, sr


def _has_required_mat_fields(value: object) -> bool:
    try:
        _mat_field(value, "amp")
        _mat_field(value, "t")
        _mat_field(value, "sr")
        return True
    except (AttributeError, KeyError, TypeError):
        return False


def _mat_field(value: object, name: str):
    if isinstance(value, dict):
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    raise AttributeError(name)


def _infer_fs(time_s: np.ndarray, source_label: str) -> float:
    if time_s.size < 2:
        raise ValueError(f"{source_label}: need at least two time samples")
    dt = np.diff(np.asarray(time_s, dtype=np.float64).ravel())
    if np.any(dt <= 0):
        raise ValueError(f"{source_label}: time vector must be strictly increasing")
    med = float(np.median(dt))
    return 1.0 / med


def _align_channels_on_common_timebase(
    amp_cols: Sequence[np.ndarray],
    time_cols: Sequence[np.ndarray],
    sr_values: Sequence[float],
    source_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not amp_cols:
        raise ValueError("Need at least one MAT file")

    if len(amp_cols) == 1:
        return np.asarray(time_cols[0], dtype=np.float64), np.asarray(amp_cols[0], dtype=np.float64)[:, None]

    start_common = max(float(time_s[0]) for time_s in time_cols)
    end_common = min(float(time_s[-1]) for time_s in time_cols)
    target_fs = min(float(sr_hz) for sr_hz in sr_values)
    dt = 1.0 / target_fs
    time_tol = max(1e-12, 1e-9 * max(abs(start_common), abs(end_common), 1.0))

    if end_common <= start_common + time_tol:
        raise ValueError(
            "MAT flight channels do not share a common overlapping time span: "
            + ", ".join(
                f"{label} [{float(time_s[0]):.9g}, {float(time_s[-1]):.9g}]"
                for label, time_s in zip(source_labels, time_cols)
            )
        )

    span = end_common - start_common
    n_samples = int(np.floor(span / dt + time_tol / dt)) + 1
    time_common = start_common + np.arange(n_samples, dtype=np.float64) * dt
    time_common = time_common[time_common <= end_common + time_tol]
    if time_common.size < 2:
        raise ValueError(
            "MAT flight channel overlap is too short after alignment; need at least two shared samples"
        )

    acc_cols_resampled: list[np.ndarray] = []
    for amp_g, time_s, label in zip(amp_cols, time_cols, source_labels):
        acc_interp = np.interp(time_common, time_s, amp_g, left=np.nan, right=np.nan)
        if not np.all(np.isfinite(acc_interp)):
            raise ValueError(f"{label}: could not interpolate onto the common time grid")
        acc_cols_resampled.append(acc_interp)

    return time_common, np.column_stack(acc_cols_resampled)


def _flight_source_label(source: str | Path | BinaryIO, idx: int) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    name = getattr(source, "name", None)
    if isinstance(name, str) and name:
        return name
    return f"mat[{idx}]"


def _flight_channel_name(source: str | Path | BinaryIO, idx: int) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).stem
    name = getattr(source, "name", None)
    if isinstance(name, str) and name:
        return Path(name).stem
    return f"ch{idx}"
