#!/usr/bin/env python3
"""Single-page Streamlit GUI for force reconstruction."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from force_recon import frf_io, pipeline, units
from force_recon.flight_io import load_flight_csv
from force_recon.simply_supported_plate import generate_simply_supported_plate_case
from gui_plotting import draw_conditioning, draw_spectra


@dataclass
class CaseData:
    label: str
    time_s: np.ndarray
    acc_g: np.ndarray
    ch_names: list[str]
    frf_freqs_hz: np.ndarray
    H: np.ndarray
    mobility_is_si_default: bool = False
    fft_window_default: str = "hann"


def _parse_mobility_csv_text(text: str, source: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    f = np.asarray(data[:, 0], dtype=np.float64)
    rest = np.asarray(data[:, 1:], dtype=np.float64)
    if rest.shape[1] % 2 != 0:
        raise ValueError(f"{source}: expected re/im column pairs after frequency column")
    c = rest[:, 0::2] + 1j * rest[:, 1::2]
    return f, c


def _load_triplet_uploads(
    fx_upload,
    fy_upload,
    fz_upload,
    flight_upload,
) -> CaseData:
    if not all([fx_upload, fy_upload, fz_upload, flight_upload]):
        raise ValueError("Upload all four CSV files (Fx, Fy, Fz, flight)")
    fx_f, fx_c = _parse_mobility_csv_text(
        fx_upload.getvalue().decode("utf-8"), "Fx CSV"
    )
    fy_f, fy_c = _parse_mobility_csv_text(
        fy_upload.getvalue().decode("utf-8"), "Fy CSV"
    )
    fz_f, fz_c = _parse_mobility_csv_text(
        fz_upload.getvalue().decode("utf-8"), "Fz CSV"
    )
    if not np.allclose(fx_f, fy_f) or not np.allclose(fx_f, fz_f):
        raise ValueError("Frequency columns must match across Fx/Fy/Fz")
    if fx_c.shape != fy_c.shape or fx_c.shape != fz_c.shape:
        raise ValueError("Mobility sensor column counts must match across Fx/Fy/Fz")
    H = np.stack([fx_c, fy_c, fz_c], axis=2)

    flight_text = flight_upload.getvalue().decode("utf-8")
    with io.StringIO(flight_text) as f:
        header = f.readline().strip().split(",")
    flight_data = np.loadtxt(io.StringIO(flight_text), delimiter=",", skiprows=1)
    if flight_data.ndim == 1:
        flight_data = flight_data.reshape(1, -1)
    t = np.asarray(flight_data[:, 0], dtype=np.float64)
    acc = np.asarray(flight_data[:, 1:], dtype=np.float64)
    names = (
        header[1:]
        if len(header) == acc.shape[1] + 1
        else [f"ch{i}" for i in range(acc.shape[1])]
    )
    return CaseData(
        label="Uploaded CSV data",
        time_s=t,
        acc_g=acc,
        ch_names=names,
        frf_freqs_hz=fx_f,
        H=H,
        mobility_is_si_default=False,
        fft_window_default="hann",
    )


def _load_from_paths(fx: str, fy: str, fz: str, flight: str) -> CaseData:
    if not all([fx.strip(), fy.strip(), fz.strip(), flight.strip()]):
        raise ValueError("Set all four CSV paths")
    frf_f, H = frf_io.load_mobility_csv_triplet(fx.strip(), fy.strip(), fz.strip())
    t, acc, names = load_flight_csv(flight.strip())
    return CaseData(
        label="CSV data from local paths",
        time_s=t,
        acc_g=acc,
        ch_names=names,
        frf_freqs_hz=frf_f,
        H=H,
        mobility_is_si_default=False,
        fft_window_default="hann",
    )


def _load_plate_demo() -> CaseData:
    case = generate_simply_supported_plate_case()
    return CaseData(
        label="Built-in simply supported plate demo",
        time_s=case.time_s,
        acc_g=case.acc_g,
        ch_names=case.channel_names,
        frf_freqs_hz=case.frf_freqs_hz,
        H=case.H_imperial,
        mobility_is_si_default=False,
        fft_window_default=case.fft_window,
    )


def _validate_case(case: CaseData) -> None:
    if case.time_s.ndim != 1:
        raise ValueError("time_s must be a 1D vector")
    if case.acc_g.ndim != 2:
        raise ValueError("acc_g must be shaped (n_samples, n_channels)")
    if case.time_s.shape[0] != case.acc_g.shape[0]:
        raise ValueError("time and acceleration lengths do not match")
    if case.H.shape[1] != case.acc_g.shape[1]:
        raise ValueError(
            f"Sensor mismatch: FRF has {case.H.shape[1]}, flight has {case.acc_g.shape[1]}"
        )
    if case.H.shape[2] != 3:
        raise ValueError("FRF matrix must have 3 load columns")
    if len(case.time_s) < 8:
        raise ValueError("Need at least 8 time samples")


def _csv_bytes_force(res) -> bytes:
    rows = np.column_stack(
        [
            res.freqs_hz,
            res.F_hat[:, 0].real,
            res.F_hat[:, 0].imag,
            res.F_hat[:, 1].real,
            res.F_hat[:, 1].imag,
            res.F_hat[:, 2].real,
            res.F_hat[:, 2].imag,
            res.valid_mask.astype(int),
        ]
    )
    sio = io.StringIO()
    np.savetxt(
        sio,
        rows,
        delimiter=",",
        header="freq_hz,Fx_re,Fx_im,Fy_re,Fy_im,Fz_re,Fz_im,valid",
        comments="",
    )
    return sio.getvalue().encode("utf-8")


def _csv_bytes_diagnostics(res) -> bytes:
    rows = np.column_stack(
        [
            res.freqs_hz,
            res.F_hat[:, 0].real,
            res.F_hat[:, 0].imag,
            res.F_hat[:, 1].real,
            res.F_hat[:, 1].imag,
            res.F_hat[:, 2].real,
            res.F_hat[:, 2].imag,
            res.cond_number,
            res.singular_values[:, 0],
            res.singular_values[:, 1],
            res.singular_values[:, 2],
            res.mac_xy,
            res.mac_xz,
            res.mac_yz,
            res.response_mac,
            res.relative_residual,
            res.valid_mask.astype(int),
        ]
    )
    header = ",".join(
        [
            "freq_hz",
            "Fx_re",
            "Fx_im",
            "Fy_re",
            "Fy_im",
            "Fz_re",
            "Fz_im",
            "cond_number",
            "sigma1",
            "sigma2",
            "sigma3",
            "mac_xy",
            "mac_xz",
            "mac_yz",
            "response_mac",
            "relative_residual",
            "valid",
        ]
    )
    sio = io.StringIO()
    np.savetxt(sio, rows, delimiter=",", header=header, comments="")
    return sio.getvalue().encode("utf-8")


def _median_finite(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64).ravel()
    m = np.isfinite(v)
    if not np.any(m):
        return float("nan")
    return float(np.median(v[m]))


def _show_case_preview(case: CaseData) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    rms = np.sqrt(np.mean(case.acc_g**2, axis=1))
    ax.plot(case.time_s, rms, lw=0.9, label="|acc|_rms (g)")
    for j in range(min(3, case.acc_g.shape[1])):
        ax.plot(case.time_s, case.acc_g[:, j], lw=0.6, alpha=0.5, label=case.ch_names[j])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title("Flight preview")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    st.pyplot(fig, clear_figure=True)


def _show_results(case: CaseData, res, t0: float, t1: float, g0: float, ch_i: int) -> None:
    fig_spec, (ax_f, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
    draw_spectra(
        ax_f,
        ax_a,
        res,
        case.time_s,
        case.acc_g,
        ch_i,
        case.ch_names,
        t0,
        t1,
        g0,
    )
    fig_spec.tight_layout()
    st.pyplot(fig_spec, clear_figure=True)

    fig_cond, axs = plt.subplots(2, 2, figsize=(11, 7))
    draw_conditioning([axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]], res)
    fig_cond.tight_layout()
    st.pyplot(fig_cond, clear_figure=True)

    fig_rr, ax_rr = plt.subplots(1, 1, figsize=(11, 3))
    m = res.valid_mask
    if np.any(m):
        rr = np.maximum(res.relative_residual[m], 1e-16)
        ax_rr.semilogy(res.freqs_hz[m], rr, color="tab:red", lw=1.0)
    ax_rr.set_xlabel("Hz")
    ax_rr.set_ylabel("||r|| / ||a||")
    ax_rr.set_title("Relative residual")
    ax_rr.grid(True, alpha=0.3)
    fig_rr.tight_layout()
    st.pyplot(fig_rr, clear_figure=True)

    if np.any(m):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valid bins", f"{int(np.sum(m))}/{len(m)}")
        c2.metric("Median κ(A)", f"{_median_finite(res.cond_number[m]):.3g}")
        c3.metric("Median response MAC", f"{_median_finite(res.response_mac[m]):.4f}")
        c4.metric("Median rel. residual", f"{_median_finite(res.relative_residual[m]):.3e}")
    else:
        st.warning("No valid reconstruction bins. Check time window and FRF frequency range.")

    c5, c6 = st.columns(2)
    c5.download_button(
        "Download F_hat_spectrum.csv",
        data=_csv_bytes_force(res),
        file_name="F_hat_spectrum.csv",
        mime="text/csv",
    )
    c6.download_button(
        "Download reconstruction_diagnostics.csv",
        data=_csv_bytes_diagnostics(res),
        file_name="reconstruction_diagnostics.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Force Reconstruction", layout="wide")
    st.title("Force Reconstruction")
    st.caption("Single-page interface for force inversion and diagnostics.")

    if "case" not in st.session_state:
        st.session_state.case = None
    if "res" not in st.session_state:
        st.session_state.res = None

    with st.sidebar:
        st.header("Data source")
        source = st.radio(
            "Choose input source",
            ["Built-in plate demo", "Local CSV paths", "Upload CSV files"],
            index=0,
        )

        try:
            if source == "Built-in plate demo":
                if st.button("Load demo case", use_container_width=True):
                    case = _load_plate_demo()
                    _validate_case(case)
                    st.session_state.case = case
                    st.session_state.res = None
                    st.success("Plate demo loaded")
            elif source == "Local CSV paths":
                default_root = Path(__file__).resolve().parent / "examples" / "data"
                fx = st.text_input(
                    "Fx CSV",
                    value=str(default_root / "mobility_Fx.csv"),
                )
                fy = st.text_input(
                    "Fy CSV",
                    value=str(default_root / "mobility_Fy.csv"),
                )
                fz = st.text_input(
                    "Fz CSV",
                    value=str(default_root / "mobility_Fz.csv"),
                )
                fl = st.text_input(
                    "Flight CSV",
                    value=str(default_root / "flight_segment.csv"),
                )
                if st.button("Load local files", use_container_width=True):
                    case = _load_from_paths(fx, fy, fz, fl)
                    _validate_case(case)
                    st.session_state.case = case
                    st.session_state.res = None
                    st.success("Local CSVs loaded")
            else:
                fx_up = st.file_uploader("Fx CSV", type=["csv"], key="fx_upload")
                fy_up = st.file_uploader("Fy CSV", type=["csv"], key="fy_upload")
                fz_up = st.file_uploader("Fz CSV", type=["csv"], key="fz_upload")
                fl_up = st.file_uploader("Flight CSV", type=["csv"], key="flight_upload")
                if st.button("Load uploaded files", use_container_width=True):
                    case = _load_triplet_uploads(fx_up, fy_up, fz_up, fl_up)
                    _validate_case(case)
                    st.session_state.case = case
                    st.session_state.res = None
                    st.success("Uploaded CSVs loaded")
        except Exception as e:
            st.error(f"Load error: {e}")

    case = st.session_state.case
    if case is None:
        st.info("Load a dataset from the sidebar to start.")
        return

    st.subheader(case.label)
    fs_hz = 1.0 / np.median(np.diff(case.time_s))
    st.write(
        f"Samples: {len(case.time_s)}, fs≈{fs_hz:.2f} Hz, channels: {case.acc_g.shape[1]}, "
        f"FRF bins: {len(case.frf_freqs_hz)}"
    )

    _show_case_preview(case)

    t_min = float(case.time_s[0])
    t_max = float(case.time_s[-1])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        t0 = st.number_input("t_start (s)", value=t_min, min_value=t_min, max_value=t_max)
    with col2:
        t1 = st.number_input("t_end (s)", value=t_max, min_value=t_min, max_value=t_max)
    with col3:
        lam = st.number_input("Tikhonov lambda", min_value=0.0, value=1e-7, format="%.3e")
    with col4:
        g0 = st.number_input("g0", min_value=1.0, value=float(units.G0_STANDARD))

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        mobility_si = st.checkbox("H is SI (m/s)/N", value=case.mobility_is_si_default)
    with col6:
        fft_window = st.selectbox(
            "FFT window",
            options=["hann", "boxcar"],
            index=0 if case.fft_window_default == "hann" else 1,
        )
    with col7:
        apply_band = st.checkbox("Apply fmin/fmax", value=True)
    with col8:
        ch_i = st.selectbox(
            "PSD channel",
            options=list(range(case.acc_g.shape[1])),
            format_func=lambda i: f"#{i} ({case.ch_names[i]})",
            index=0,
        )

    f_min, f_max = None, None
    if apply_band:
        c9, c10 = st.columns(2)
        with c9:
            f_min = st.number_input("fmin (Hz)", min_value=0.0, value=5.0)
        with c10:
            nyq = float(fs_hz / 2.0)
            f_max = st.number_input("fmax (Hz)", min_value=0.0, value=min(220.0, nyq))

    run_col, _ = st.columns([1, 4])
    with run_col:
        run_now = st.button("Run reconstruction", type="primary")

    if run_now:
        if t1 <= t0:
            st.error("Need t_end > t_start")
            return
        if f_min is not None and f_max is not None and f_max < f_min:
            st.error("Need fmax >= fmin")
            return
        try:
            st.session_state.res = pipeline.reconstruct_forces(
                time_s=case.time_s,
                acc_g=case.acc_g,
                t_start=float(t0),
                t_end=float(t1),
                frf_freqs_hz=case.frf_freqs_hz,
                H_imperial=case.H,
                g0=float(g0),
                tikhonov_lambda=float(lam),
                mobility_is_si=bool(mobility_si),
                f_min_hz=float(f_min) if f_min is not None else None,
                f_max_hz=float(f_max) if f_max is not None else None,
                fft_window=fft_window,
            )
        except Exception as e:
            st.error(f"Run error: {e}")
            st.session_state.res = None
            return
        st.success("Reconstruction complete")

    if st.session_state.res is not None:
        st.subheader("Results")
        _show_results(case, st.session_state.res, float(t0), float(t1), float(g0), int(ch_i))


if __name__ == "__main__":
    main()
