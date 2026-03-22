"""Tkinter + TkAgg GUI (Linux/Windows, or macOS with FORCE_RECON_GUI=tk)."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from force_recon import frf_io, pipeline, units
from force_recon.export_nastran import write_force_spectrum_csv
from force_recon.flight_io import load_flight_csv
from gui_plotting import draw_conditioning, draw_spectra


class ForceReconApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Force reconstruction")
        self.geometry("1100x820")

        self.path_fx = tk.StringVar()
        self.path_fy = tk.StringVar()
        self.path_fz = tk.StringVar()
        self.path_flight = tk.StringVar()
        self.g0 = tk.DoubleVar(value=units.G0_STANDARD)
        self.t0 = tk.DoubleVar(value=0.0)
        self.t1 = tk.DoubleVar(value=1.0)
        self.lam = tk.DoubleVar(value=0.0)
        self.mobility_si = tk.BooleanVar(value=False)

        self.time_s: np.ndarray | None = None
        self.acc_g: np.ndarray | None = None
        self.ch_names: list[str] = []
        self.res = None

        self._build_controls()
        self._build_plots()

    def _browse(self, var: tk.StringVar) -> None:
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            var.set(p)

    def _build_controls(self) -> None:
        lf = ttk.LabelFrame(
            self,
            text="Files (mobility: freq_hz,re0,im0,… ; flight: time_s + channels in g)",
        )
        lf.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        def row(r: int, label: str, var: tk.StringVar) -> None:
            ttk.Label(lf, text=label).grid(row=r, column=0, sticky=tk.W, padx=4, pady=2)
            ttk.Entry(lf, textvariable=var, width=70).grid(row=r, column=1, sticky=tk.EW, padx=4)
            ttk.Button(lf, text="Browse…", command=lambda: self._browse(var)).grid(
                row=r, column=2, padx=4
            )

        row(0, "Fx CSV", self.path_fx)
        row(1, "Fy CSV", self.path_fy)
        row(2, "Fz CSV", self.path_fz)
        row(3, "Flight CSV", self.path_flight)
        lf.columnconfigure(1, weight=1)

        opts = ttk.Frame(self)
        opts.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        ttk.Checkbutton(opts, text="H already SI (m/s)/N", variable=self.mobility_si).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Label(opts, text="g₀").pack(side=tk.LEFT)
        ttk.Entry(opts, textvariable=self.g0, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(opts, text="t_start (s)").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(opts, textvariable=self.t0, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(opts, text="t_end (s)").pack(side=tk.LEFT)
        ttk.Entry(opts, textvariable=self.t1, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(opts, text="Tikhonov λ").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(opts, textvariable=self.lam, width=10).pack(side=tk.LEFT, padx=4)

        self.channel_combo = ttk.Combobox(opts, state="readonly", width=20)
        self.channel_combo.pack(side=tk.RIGHT, padx=8)
        ttk.Label(opts, text="PSD channel:").pack(side=tk.RIGHT)

        btns = ttk.Frame(self)
        btns.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        ttk.Button(btns, text="Load files", command=self._load_files).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Run reconstruction", command=self._run).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Save F̂ CSV…", command=self._save_csv).pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(self, text="Choose CSVs and click Load files, then Run reconstruction.")
        self.status.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=2)

    def _build_plots(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        tab1 = ttk.Frame(nb)
        tab2 = ttk.Frame(nb)
        nb.add(tab1, text="Spectra")
        nb.add(tab2, text="Conditioning")

        self.fig_spec = Figure(figsize=(10, 4), dpi=100)
        self.ax_fx = self.fig_spec.add_subplot(121)
        self.ax_ax = self.fig_spec.add_subplot(122)
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=tab1)
        self.canvas_spec.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_spec, tab1).update()

        self.fig_cond = Figure(figsize=(10, 5), dpi=100)
        self.ax_c = [
            self.fig_cond.add_subplot(221),
            self.fig_cond.add_subplot(222),
            self.fig_cond.add_subplot(223),
            self.fig_cond.add_subplot(224),
        ]
        self.canvas_cond = FigureCanvasTkAgg(self.fig_cond, master=tab2)
        self.canvas_cond.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_cond, tab2).update()

    def _load_files(self) -> None:
        px, py, pz, pf = (
            self.path_fx.get().strip(),
            self.path_fy.get().strip(),
            self.path_fz.get().strip(),
            self.path_flight.get().strip(),
        )
        if not all([px, py, pz, pf]):
            messagebox.showwarning("Missing files", "Set all four CSV paths.")
            return
        try:
            _, H = frf_io.load_mobility_csv_triplet(px, py, pz)
            t, acc, names = load_flight_csv(pf)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        if acc.shape[1] != H.shape[1]:
            messagebox.showerror(
                "Mismatch",
                f"FRF sensors={H.shape[1]}, flight channels={acc.shape[1]}",
            )
            return
        self.time_s = t
        self.acc_g = acc
        self.ch_names = names
        self.channel_combo["values"] = names
        if names:
            self.channel_combo.current(0)
        fs = 1.0 / np.median(np.diff(t))
        self.status.config(
            text=f"Loaded: {len(t)} samples, fs≈{fs:.1f} Hz, {H.shape[1]} channels."
        )

    def _run(self) -> None:
        if self.time_s is None or self.acc_g is None:
            messagebox.showwarning("No data", "Load files first.")
            return
        px, py, pz = self.path_fx.get().strip(), self.path_fy.get().strip(), self.path_fz.get().strip()
        if not all([px, py, pz]):
            messagebox.showwarning("Missing FRF", "Set Fx, Fy, Fz CSV paths.")
            return
        try:
            frf_f, H_imp = frf_io.load_mobility_csv_triplet(px, py, pz)
            self.res = pipeline.reconstruct_forces(
                time_s=self.time_s,
                acc_g=self.acc_g,
                t_start=float(self.t0.get()),
                t_end=float(self.t1.get()),
                frf_freqs_hz=frf_f,
                H_imperial=H_imp,
                g0=float(self.g0.get()),
                tikhonov_lambda=float(self.lam.get()),
                mobility_is_si=bool(self.mobility_si.get()),
            )
        except Exception as e:
            messagebox.showerror("Run error", str(e))
            return
        self._redraw()
        self.status.config(text="Reconstruction done. Use Save F̂ CSV or inspect tabs.")

    def _redraw(self) -> None:
        res = self.res
        if res is None or self.time_s is None or self.acc_g is None:
            return
        g0 = float(self.g0.get())
        t0, t1 = float(self.t0.get()), float(self.t1.get())
        ch_i = max(0, self.channel_combo.current())

        draw_spectra(
            self.ax_fx,
            self.ax_ax,
            res,
            self.time_s,
            self.acc_g,
            ch_i,
            self.ch_names,
            t0,
            t1,
            g0,
        )
        self.fig_spec.tight_layout()
        self.canvas_spec.draw()

        draw_conditioning(self.ax_c, res)
        self.fig_cond.tight_layout()
        self.canvas_cond.draw()

    def _save_csv(self) -> None:
        if self.res is None:
            messagebox.showwarning("Nothing to save", "Run reconstruction first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="F_hat_spectrum.csv",
        )
        if not path:
            return
        write_force_spectrum_csv(
            self.res.freqs_hz,
            self.res.F_hat,
            path,
            valid_mask=self.res.valid_mask,
        )
        messagebox.showinfo("Saved", path)


def main() -> None:
    app = ForceReconApp()
    app.mainloop()
