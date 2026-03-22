#!/usr/bin/env python3
"""
Interactive force-reconstruction GUI (cross-platform).

- **Time slice:** drag horizontally on the flight preview (matplotlib SpanSelector).
- **macOS:** `macosx` backend — no Tk window (avoids Tcl/Tk abort on some Xcode Pythons).
  File picks use **AppleScript** only (still no `import tkinter` on Darwin).
- **Windows / Linux:** `TkAgg` + **Tkinter** used only for native file dialogs (usually fine on Windows).

Run: ``python3 gui_app.py`` (see ``gui_app.py`` for CLI / legacy Tk routing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if sys.platform == "darwin":
    matplotlib.use("macosx")
else:
    matplotlib.use("TkAgg")

import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, SpanSelector, TextBox

import matplotlib.pyplot as plt

from force_recon import frf_io, pipeline, units
from force_recon.export_nastran import (
    write_force_spectrum_csv,
    write_reconstruction_diagnostics_csv,
)
from force_recon.flight_io import load_flight_csv
from force_recon.simply_supported_plate import generate_simply_supported_plate_case
from gui_file_dialogs import choose_open_file, choose_save_file
from gui_plotting import draw_conditioning, draw_spectra


class InteractiveForceReconGUI:
    def __init__(self) -> None:
        self.paths: dict[str, Path | None] = {
            "fx": None,
            "fy": None,
            "fz": None,
            "flight": None,
        }
        self.time_s: np.ndarray | None = None
        self.acc_g: np.ndarray | None = None
        self.ch_names: list[str] = []
        self.frf_f: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.res = None

        self.t0: float = 0.0
        self.t1: float = 1.0
        self.g0: float = units.G0_STANDARD
        self.lam: float = 0.0
        self.f_min_hz: float | None = 0.0
        self.f_max_hz: float | None = None
        self.fft_window: str = "hann"
        self.mobility_si: bool = False
        self.ch_i: int = 0

        self.window_options = ["hann", "boxcar"]

        self.span: SpanSelector | None = None
        self.fig_setup = plt.figure("Force reconstruction — setup", figsize=(13.5, 8.0))
        self.ax_prev = self.fig_setup.add_axes([0.06, 0.34, 0.90, 0.60])
        self._build_controls()
        self._status(
            "Load CSVs (or built-in plate demo), drag the orange span, then run reconstruction."
        )

    def _build_controls(self) -> None:
        fig = self.fig_setup

        ax_files = fig.add_axes([0.06, 0.26, 0.17, 0.06])
        self.btn_files = Button(ax_files, "Select CSV files…")
        self.btn_files.on_clicked(self._on_files)

        ax_demo = fig.add_axes([0.24, 0.26, 0.16, 0.06])
        self.btn_demo = Button(ax_demo, "Load Plate Demo")
        self.btn_demo.on_clicked(self._on_demo)

        ax_full = fig.add_axes([0.41, 0.26, 0.12, 0.06])
        self.btn_full = Button(ax_full, "Use Full Time")
        self.btn_full.on_clicked(self._on_full_window)

        ax_run = fig.add_axes([0.54, 0.26, 0.16, 0.06])
        self.btn_run = Button(ax_run, "Run reconstruction")
        self.btn_run.on_clicked(self._on_run)

        ax_save = fig.add_axes([0.71, 0.26, 0.12, 0.06])
        self.btn_save = Button(ax_save, "Save F̂ CSV…")
        self.btn_save.on_clicked(self._on_save)

        ax_save_diag = fig.add_axes([0.84, 0.26, 0.12, 0.06])
        self.btn_save_diag = Button(ax_save_diag, "Save diagnostics…")
        self.btn_save_diag.on_clicked(self._on_save_diagnostics)

        ax_t0 = fig.add_axes([0.06, 0.17, 0.11, 0.055])
        self.tb_t0 = TextBox(ax_t0, "t0 ", initial="0")
        self.tb_t0.on_submit(self._on_t0_text)

        ax_t1 = fig.add_axes([0.19, 0.17, 0.11, 0.055])
        self.tb_t1 = TextBox(ax_t1, "t1 ", initial="1")
        self.tb_t1.on_submit(self._on_t1_text)

        ax_g0 = fig.add_axes([0.32, 0.17, 0.11, 0.055])
        self.tb_g0 = TextBox(ax_g0, "g0 ", initial=str(self.g0))

        ax_lam = fig.add_axes([0.45, 0.17, 0.11, 0.055])
        self.tb_lam = TextBox(ax_lam, "λ ", initial="0")

        ax_fmin = fig.add_axes([0.58, 0.17, 0.11, 0.055])
        self.tb_fmin = TextBox(ax_fmin, "fmin ", initial="0")

        ax_fmax = fig.add_axes([0.71, 0.17, 0.11, 0.055])
        self.tb_fmax = TextBox(ax_fmax, "fmax ", initial="")

        ax_ch = fig.add_axes([0.84, 0.17, 0.12, 0.055])
        self.tb_ch = TextBox(ax_ch, "PSD ch# ", initial="0")
        self.tb_ch.on_submit(self._on_ch_text)

        rax = fig.add_axes([0.06, 0.05, 0.16, 0.10])
        rax.set_title("H units", fontsize=8)
        self.chk_si = CheckButtons(rax, ["SI (m/s)/N"], [False])
        self.chk_si.on_clicked(self._on_chk_si)

        ax_win = fig.add_axes([0.24, 0.03, 0.17, 0.12])
        ax_win.set_title("FFT window", fontsize=8)
        self.rb_window = RadioButtons(ax_win, self.window_options, active=0)
        self.rb_window.on_clicked(self._on_window_choice)

        ax_prev_ch = fig.add_axes([0.44, 0.08, 0.06, 0.06])
        self.btn_prev_ch = Button(ax_prev_ch, "◀ ch")
        self.btn_prev_ch.on_clicked(self._on_prev_channel)

        ax_next_ch = fig.add_axes([0.51, 0.08, 0.06, 0.06])
        self.btn_next_ch = Button(ax_next_ch, "ch ▶")
        self.btn_next_ch.on_clicked(self._on_next_channel)

        self.ax_channel_label = fig.add_axes([0.60, 0.06, 0.36, 0.08])
        self.ax_channel_label.set_axis_off()
        self.channel_label = self.ax_channel_label.text(
            0.0, 0.5, "PSD channel: #0", va="center", fontsize=10
        )

    def _status(self, msg: str) -> None:
        self.fig_setup.suptitle(msg, fontsize=10, y=0.02, color="0.2")
        self.fig_setup.canvas.draw_idle()

    def _on_chk_si(self, _label: str) -> None:
        self.mobility_si = self.chk_si.get_status()[0]

    def _on_t0_text(self, s: str) -> None:
        try:
            self.t0 = float(s)
        except ValueError:
            pass

    def _on_t1_text(self, s: str) -> None:
        try:
            self.t1 = float(s)
        except ValueError:
            pass

    def _on_window_choice(self, label: str) -> None:
        if label in self.window_options:
            self.fft_window = label

    def _on_ch_text(self, s: str) -> None:
        try:
            self._set_channel_index(max(0, int(float(s))), update_box=False)
        except ValueError:
            pass

    def _on_prev_channel(self, _event=None) -> None:
        self._set_channel_index(self.ch_i - 1, update_box=True)

    def _on_next_channel(self, _event=None) -> None:
        self._set_channel_index(self.ch_i + 1, update_box=True)

    def _set_channel_index(self, idx: int, *, update_box: bool) -> None:
        n_ch = self.acc_g.shape[1] if self.acc_g is not None else None
        if n_ch is not None:
            idx = max(0, min(idx, n_ch - 1))
        else:
            idx = max(0, idx)
        self.ch_i = idx
        if update_box and self.tb_ch.text.strip() != str(idx):
            self.tb_ch.set_val(str(idx))
        name = self.ch_names[idx] if self.ch_names and idx < len(self.ch_names) else f"ch{idx}"
        self.channel_label.set_text(f"PSD channel: #{idx} ({name})")
        self.fig_setup.canvas.draw_idle()

    def _set_time_window(self, t0: float, t1: float) -> None:
        self.t0, self.t1 = float(t0), float(t1)
        self.tb_t0.set_val(f"{self.t0:.6g}")
        self.tb_t1.set_val(f"{self.t1:.6g}")

    @staticmethod
    def _parse_optional_float(text: str) -> float | None:
        s = text.strip().lower()
        if s in {"", "none", "inf", "+inf"}:
            return None
        return float(s)

    def _load_case_arrays(
        self,
        *,
        frf_freqs_hz: np.ndarray,
        H_mobility: np.ndarray,
        time_s: np.ndarray,
        acc_g: np.ndarray,
        channel_names: list[str],
        source_label: str,
    ) -> None:
        t = np.asarray(time_s, dtype=np.float64).ravel()
        acc = np.asarray(acc_g, dtype=np.float64)
        if acc.ndim == 1:
            acc = acc.reshape(-1, 1)
        H = np.asarray(H_mobility, dtype=np.complex128)
        if len(t) != acc.shape[0]:
            self._status(
                f"Load error: time length {len(t)} does not match acceleration rows {acc.shape[0]}."
            )
            return
        if acc.shape[1] != H.shape[1]:
            self._status(
                f"Mismatch: FRF has {H.shape[1]} sensors, flight has {acc.shape[1]} channels."
            )
            return
        if len(t) < 8:
            self._status("Need at least 8 time samples.")
            return
        self.frf_f = np.asarray(frf_freqs_hz, dtype=np.float64).ravel()
        self.H = H
        self.time_s = t
        self.acc_g = acc
        self.ch_names = (
            channel_names
            if len(channel_names) == acc.shape[1]
            else [f"ch{i}" for i in range(acc.shape[1])]
        )
        self.res = None
        self._plot_preview()
        span = 0.1 * (float(t[-1]) - float(t[0]))
        t0 = float(t[0]) + span
        t1 = float(t[-1]) - span
        if t1 <= t0:
            t0, t1 = float(t[0]), float(t[-1])
        self._set_time_window(t0, t1)
        self._set_channel_index(0, update_box=True)
        fs = 1.0 / np.median(np.diff(t))
        self._status(
            (
                f"{source_label}: {len(t)} samples, fs≈{fs:.1f} Hz, {acc.shape[1]} channels. "
                "Drag preview span or edit t0/t1."
            )
        )

    def _on_files(self, _event=None) -> None:
        self._status("Choose Fx, Fy, Fz, then flight…")
        self.fig_setup.canvas.draw_idle()
        fx = choose_open_file("Mobility CSV — unit Fx")
        if not fx:
            self._status("Cancelled.")
            return
        fy = choose_open_file("Mobility CSV — unit Fy")
        if not fy:
            self._status("Cancelled.")
            return
        fz = choose_open_file("Mobility CSV — unit Fz")
        if not fz:
            self._status("Cancelled.")
            return
        fl = choose_open_file("Flight CSV (time + channels in g)")
        if not fl:
            self._status("Cancelled.")
            return
        self.paths["fx"], self.paths["fy"], self.paths["fz"], self.paths["flight"] = (
            fx,
            fy,
            fz,
            fl,
        )
        try:
            frf_f, H = frf_io.load_mobility_csv_triplet(fx, fy, fz)
            t, acc, names = load_flight_csv(fl)
        except Exception as e:
            self._status(f"Load error: {e}")
            return
        self._load_case_arrays(
            frf_freqs_hz=frf_f,
            H_mobility=H,
            time_s=t,
            acc_g=acc,
            channel_names=names,
            source_label="Loaded CSV data",
        )

    def _on_demo(self, _event=None) -> None:
        try:
            case = generate_simply_supported_plate_case()
        except Exception as e:
            self._status(f"Demo case error: {e}")
            return
        if self.chk_si.get_status()[0]:
            self.chk_si.set_active(0)
        self.tb_lam.set_val("1e-7")
        self.tb_fmin.set_val("5")
        self.tb_fmax.set_val("220")
        if self.rb_window.value_selected != case.fft_window:
            self.rb_window.set_active(self.window_options.index(case.fft_window))
        self.fft_window = case.fft_window
        self.paths = {"fx": None, "fy": None, "fz": None, "flight": None}
        self._load_case_arrays(
            frf_freqs_hz=case.frf_freqs_hz,
            H_mobility=case.H_imperial,
            time_s=case.time_s,
            acc_g=case.acc_g,
            channel_names=case.channel_names,
            source_label="Built-in simply supported plate demo",
        )
        self._set_time_window(case.t_start, case.t_end)
        self._status(
            "Built-in plate demo loaded. Window=boxcar, λ=1e-7, band=5..220 Hz. Click Run."
        )

    def _on_full_window(self, _event=None) -> None:
        if self.time_s is None:
            self._status("Load data first.")
            return
        self._set_time_window(float(self.time_s[0]), float(self.time_s[-1]))
        self._status("Time window set to full record.")

    def _plot_preview(self) -> None:
        assert self.time_s is not None and self.acc_g is not None
        self.ax_prev.clear()
        t = self.time_s
        acc = self.acc_g
        # RMS across channels for a clear envelope
        rms = np.sqrt(np.mean(acc**2, axis=1))
        self.ax_prev.plot(t, rms, "C0", lw=0.8, label="|acc|_rms (g)")
        for j in range(min(3, acc.shape[1])):
            self.ax_prev.plot(t, acc[:, j], alpha=0.35, lw=0.5, label=self.ch_names[j])
        self.ax_prev.set_xlabel("Time (s)")
        self.ax_prev.set_ylabel("Acceleration (g)")
        self.ax_prev.set_title("Flight preview — drag to select reconstruction window")
        self.ax_prev.legend(loc="upper right", fontsize=7)
        self.ax_prev.grid(True, alpha=0.3)

        if self.span is not None:
            self.span.disconnect_events()
        span_kw = dict(
            useblit=True,
            props=dict(alpha=0.25, facecolor="tab:orange"),
            interactive=True,
        )
        try:
            self.span = SpanSelector(
                self.ax_prev,
                self._on_span,
                "horizontal",
                drag_from_anywhere=True,
                **span_kw,
            )
        except TypeError:
            self.span = SpanSelector(
                self.ax_prev,
                self._on_span,
                "horizontal",
                **span_kw,
            )
        self.fig_setup.canvas.draw_idle()

    def _on_span(self, xmin: float, xmax: float) -> None:
        self._set_time_window(float(min(xmin, xmax)), float(max(xmin, xmax)))
        self.fig_setup.canvas.draw_idle()

    def _read_float_boxes(self) -> bool:
        try:
            self.g0 = float(self.tb_g0.text)
            self.lam = float(self.tb_lam.text)
            self.t0 = float(self.tb_t0.text)
            self.t1 = float(self.tb_t1.text)
            self.f_min_hz = self._parse_optional_float(self.tb_fmin.text)
            self.f_max_hz = self._parse_optional_float(self.tb_fmax.text)
            self._set_channel_index(max(0, int(float(self.tb_ch.text))), update_box=False)
        except ValueError:
            self._status("Invalid numeric field in t0, t1, g0, λ, channel, fmin, or fmax.")
            return False
        if self.f_min_hz is not None and self.f_min_hz < 0:
            self._status("fmin must be >= 0.")
            return False
        if (
            self.f_min_hz is not None
            and self.f_max_hz is not None
            and self.f_max_hz < self.f_min_hz
        ):
            self._status("Need fmax >= fmin.")
            return False
        return True

    def _on_run(self, _event=None) -> None:
        if not self._read_float_boxes():
            return
        if self.time_s is None or self.acc_g is None or self.frf_f is None or self.H is None:
            self._status("Load CSV files first.")
            return
        if self.t1 <= self.t0:
            self._status("Need t₁ > t₀.")
            return
        n_ch = self.acc_g.shape[1]
        if self.ch_i >= n_ch:
            self.ch_i = n_ch - 1
            self.tb_ch.set_val(str(self.ch_i))
        self.mobility_si = self.chk_si.get_status()[0]
        try:
            self.res = pipeline.reconstruct_forces(
                time_s=self.time_s,
                acc_g=self.acc_g,
                t_start=self.t0,
                t_end=self.t1,
                frf_freqs_hz=self.frf_f,
                H_imperial=self.H,
                g0=self.g0,
                tikhonov_lambda=self.lam,
                mobility_is_si=self.mobility_si,
                f_min_hz=self.f_min_hz,
                f_max_hz=self.f_max_hz,
                fft_window=self.fft_window,
            )
        except Exception as e:
            self._status(f"Run error: {e}")
            return
        self._show_results()
        valid = int(np.sum(self.res.valid_mask))
        self._status(
            (
                f"Done: reconstructed {valid}/{len(self.res.freqs_hz)} bins "
                f"(window={self.fft_window})."
            )
        )

    @staticmethod
    def _finite_median(x: np.ndarray) -> float:
        v = np.asarray(x, dtype=np.float64).ravel()
        m = np.isfinite(v)
        if not np.any(m):
            return float("nan")
        return float(np.median(v[m]))

    def _summary_lines(self, res) -> list[str]:
        m = res.valid_mask
        n_valid = int(np.sum(m))
        if n_valid == 0:
            return ["No valid reconstruction bins.", "Check FRF range/time window/band limits."]
        f_valid = res.freqs_hz[m]
        med_cond = self._finite_median(res.cond_number[m])
        med_mac = self._finite_median(res.response_mac[m])
        med_rr = self._finite_median(res.relative_residual[m])
        return [
            f"Valid bins: {n_valid} / {len(res.freqs_hz)}",
            f"Valid range: {float(np.min(f_valid)):.1f} .. {float(np.max(f_valid)):.1f} Hz",
            f"Median κ(A): {med_cond:.3g}",
            f"Median response MAC: {med_mac:.4f}",
            f"Median relative residual: {med_rr:.3e}",
            f"FFT window: {self.fft_window}",
            f"Band limits: fmin={self.f_min_hz}, fmax={self.f_max_hz}",
            f"PSD channel: #{self.ch_i} ({self.ch_names[self.ch_i]})",
        ]

    def _show_results(self) -> None:
        res = self.res
        assert res is not None and self.time_s is not None and self.acc_g is not None

        fig = plt.figure("Force reconstruction — results", figsize=(12.5, 11.0))
        fig.clf()
        ax_f = fig.add_subplot(4, 2, 1)
        ax_a = fig.add_subplot(4, 2, 2)
        ax_c = [
            fig.add_subplot(4, 2, 3),
            fig.add_subplot(4, 2, 4),
            fig.add_subplot(4, 2, 5),
            fig.add_subplot(4, 2, 6),
        ]
        ax_rr = fig.add_subplot(4, 2, 7)
        ax_summary = fig.add_subplot(4, 2, 8)
        draw_spectra(
            ax_f,
            ax_a,
            res,
            self.time_s,
            self.acc_g,
            self.ch_i,
            self.ch_names,
            self.t0,
            self.t1,
            self.g0,
        )
        draw_conditioning(ax_c, res)

        m = res.valid_mask
        ax_rr.clear()
        if np.any(m):
            rr = np.maximum(res.relative_residual[m], 1e-16)
            ax_rr.semilogy(res.freqs_hz[m], rr, color="tab:red")
        ax_rr.set_xlabel("Hz")
        ax_rr.set_ylabel("||r|| / ||a||")
        ax_rr.set_title("Relative residual")
        ax_rr.grid(True, alpha=0.3)

        ax_summary.clear()
        ax_summary.set_axis_off()
        ax_summary.set_title("Run summary", loc="left")
        lines = self._summary_lines(res)
        y = 0.95
        for line in lines:
            ax_summary.text(0.0, y, line, transform=ax_summary.transAxes, va="top")
            y -= 0.12

        fig.tight_layout()
        fig.canvas.draw_idle()
        try:
            fig.canvas.manager.window.lift()
        except Exception:
            pass

    def _on_save(self, _event=None) -> None:
        if self.res is None:
            self._status("Run reconstruction first.")
            return
        path = choose_save_file("F_hat_spectrum.csv")
        if not path:
            return
        write_force_spectrum_csv(
            self.res.freqs_hz,
            self.res.F_hat,
            path,
            valid_mask=self.res.valid_mask,
        )
        self._status(f"Saved {path}")

    def _on_save_diagnostics(self, _event=None) -> None:
        if self.res is None:
            self._status("Run reconstruction first.")
            return
        path = choose_save_file("reconstruction_diagnostics.csv")
        if not path:
            return
        write_reconstruction_diagnostics_csv(
            self.res.freqs_hz,
            self.res.F_hat,
            self.res.cond_number,
            self.res.singular_values,
            self.res.mac_xy,
            self.res.mac_xz,
            self.res.mac_yz,
            self.res.response_mac,
            self.res.relative_residual,
            self.res.valid_mask,
            path,
        )
        self._status(f"Saved {path}")


def run_gui() -> int:
    InteractiveForceReconGUI()
    print("Close all plot windows to quit.")
    plt.show()
    return 0


def main() -> int:
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
