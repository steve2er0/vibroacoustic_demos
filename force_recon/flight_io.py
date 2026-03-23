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
    match the FRF sensor order.
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

    time_ref: np.ndarray | None = None
    sr_ref: float | None = None
    acc_cols: list[np.ndarray] = []
    names: list[str] = []

    for idx, item in enumerate(items):
        source_label = source_names[idx] if source_names is not None else _flight_source_label(item, idx)
        amp_g, time_s, sr_hz = _load_single_channel_mat(item, source_label)
        if time_ref is None:
            time_ref = time_s
            sr_ref = sr_hz
        else:
            assert sr_ref is not None
            _validate_matching_timebase(time_ref, sr_ref, time_s, sr_hz, source_label)
        acc_cols.append(amp_g)
        if channel_names is not None:
            names.append(str(channel_names[idx]))
        else:
            names.append(_flight_channel_name(item, idx))

    assert time_ref is not None
    acc = np.column_stack(acc_cols)
    return time_ref, acc, names


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
    med = float(np.median(dt))
    if med <= 0:
        raise ValueError(f"{source_label}: time vector must be strictly increasing")
    return 1.0 / med


def _validate_matching_timebase(
    time_ref: np.ndarray,
    sr_ref: float,
    time_s: np.ndarray,
    sr_hz: float,
    source_label: str,
) -> None:
    if time_s.shape != time_ref.shape:
        raise ValueError(
            f"{source_label}: sample count {time_s.shape[0]} does not match {time_ref.shape[0]}"
        )
    sr_tol = max(1e-9, 1e-6 * max(abs(sr_ref), 1.0))
    if abs(sr_hz - sr_ref) > sr_tol:
        raise ValueError(f"{source_label}: sample rate {sr_hz} does not match {sr_ref}")
    time_tol = max(1e-9, 1e-6 / max(abs(sr_ref), 1.0))
    if not np.allclose(time_s, time_ref, rtol=0.0, atol=time_tol):
        raise ValueError(f"{source_label}: time vector does not match the other channels")


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
