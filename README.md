# vibroacoustic_demos

Vibroacoustic demos plus a **force reconstruction** toolkit (`force_recon/`).

## Force reconstruction

Estimate three translational interface forces from **flight accelerations (g)** and **mobility** **H** = *v/F* from three NASTRAN-style CSVs (unit **Fx**, **Fy**, **Fz**).

### Setup

```bash
python3 -m pip install -r requirements.txt
# editable install (if you use pyproject optional extras):
# python3 -m pip install -e ".[gui,dev]"
```

### GUI (desktop, self-contained)

```bash
python3 gui_app.py
```

- **All platforms:** **interactive Matplotlib** window — load four CSVs, then **drag the orange band** on the flight preview to set the reconstruction **time window** (or type **t₀ / t₁**). **Run reconstruction** opens a second window with spectra and conditioning. **Save F̂ CSV** uses a native save dialog.
- Added workflow controls: **Use Full Time**, **PSD channel step buttons**, **FFT window selection** (`hann` / `boxcar`), and optional **frequency-band limits** (`fmin`, `fmax`).
- Added diagnostics export: **Save diagnostics…** writes per-frequency conditioning/MAC/residual metrics to CSV.
- Added one-click validation dataset: **Load Plate Demo** generates a built-in simply-supported plate case (modal synthetic FRF + flight).
- **macOS:** `macosx` backend + **AppleScript** file dialogs (no Tk import — avoids Tcl abort on some Xcode Pythons).
- **Windows / Linux:** `TkAgg` + **Tk** only for open/save dialogs (not the main plot window).
- **Headless / script:** `python3 gui_app.py --fx … --fy … --fz … --flight … --t0 … --t1 … [--save out.csv]` (see `--help`).
- **Legacy:** full Tk embedded UI: `FORCE_RECON_GUI=tk python3 gui_app.py`

### GUI (recommended: Streamlit web app)

```bash
python3 -m pip install streamlit
python3 gui_app.py --web
# equivalent:
# FORCE_RECON_GUI=web python3 gui_app.py
# streamlit run gui_streamlit.py
```

- Single-page workflow: load data, run inversion, inspect force/accel spectra + conditioning + residual on one page.
- No hidden secondary plot windows; better behavior on macOS and remote sessions.
- Supports:
  - Built-in simply supported plate demo dataset
  - Local CSV paths
  - Direct CSV uploads (Fx/Fy/Fz + flight)
- Downloads:
  - `F_hat_spectrum.csv`
  - `reconstruction_diagnostics.csv`

Upload:

- Three mobility CSVs: `freq_hz,re0,im0,re1,im1,...` per file (same frequencies and sensor order).
- Flight CSV: `time_s,ch0,ch1,...` in **g**.

Options: **g₀**, time window, Tikhonov **λ**, checkbox if **H** is already SI.

### Example synthetic data

```bash
python3 examples/generate_synthetic_data.py
```

Then in the GUI use `examples/data/mobility_Fx.csv` (and **Fy** / **Fz**) plus `flight_segment.csv`, window e.g. **0.5–3.5 s**.

### Tests

```bash
python3 examples/generate_synthetic_csv.py   # creates examples/synthetic_data/ for integration test
python3 -m pytest tests/ -q
```

(All tests should pass; **9 passed** including the simply-supported plate reconstruction case.)

### Library usage

```python
from force_recon import frf_io, pipeline

frf_f, H = frf_io.load_mobility_csv_triplet("H_x.csv", "H_y.csv", "H_z.csv")
res = pipeline.reconstruct_forces(
    time_s=t,
    acc_g=acc_g,
    t_start=0.0,
    t_end=1.0,
    frf_freqs_hz=frf_f,
    H_imperial=H,
    mobility_is_si=False,
)
# res.F_hat — complex (n_f, 3), res.a_meas_fft, conditioning, MAC, …
```

Exports: `force_recon.export_nastran.write_force_spectrum_csv`, `tabrnd1_snippet`, `tabled1_re_im_snippets`.
