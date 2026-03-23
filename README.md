# vibroacoustic_demos

Vibroacoustic demos plus a **force reconstruction** toolkit (`force_recon/`).

## Force reconstruction

Estimate three translational interface forces from **flight accelerations (g)** and **mobility** **H** = *v/F* from three NASTRAN-style CSVs (unit **Fx**, **Fy**, **Fz**). Flight data can come from a single CSV or ordered per-channel MATLAB `.mat` files. For workflow/plumbing tests, Python can also synthesize a dummy **H = ones()** mobility tensor from the flight timebase.

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

- **All platforms:** **interactive Matplotlib** window — load three FRF CSVs plus either a flight CSV or ordered per-channel `.mat` files, then **drag the orange band** on the flight preview to set the reconstruction **time window** (or type **t₀ / t₁**). **Run reconstruction** opens a second window with spectra and conditioning. **Save F̂ CSV** uses a native save dialog.
- Added a dummy-mobility option: **Use H = ones()** if you want to exercise the flight-data workflow before real FRFs are available.
- Added workflow controls: **Use Full Time**, **PSD channel step buttons**, **FFT window selection** (`hann` / `boxcar`), and optional **frequency-band limits** (`fmin`, `fmax`).
- Added diagnostics export: **Save diagnostics…** writes per-frequency conditioning/MAC/residual metrics to CSV.
- Added one-click validation dataset: **Load Plate Demo** generates a built-in simply-supported plate case (modal synthetic FRF + flight).
- **macOS:** `macosx` backend + **AppleScript** file dialogs (no Tk import — avoids Tcl abort on some Xcode Pythons).
- **Windows / Linux:** `TkAgg` + **Tk** only for open/save dialogs (not the main plot window).
- **Headless / script:** `python3 gui_app.py --fx … --fy … --fz … --flight … --t0 … --t1 … [--save out.csv]`, `python3 gui_app.py --fx … --fy … --fz … --flight-mat ch01.mat ch02.mat ch03.mat --t0 … --t1 …`, or `python3 gui_app.py --ones-h --flight-mat ch01.mat ch02.mat ch03.mat --t0 … --t1 …` (see `--help`).
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
  - Local file paths for flight CSV or ordered per-channel MAT files
  - Direct uploads for Fx/Fy/Fz CSV plus either a flight CSV or flight MAT files
  - Optional dummy mobility: **H = ones()**
- Downloads:
  - `F_hat_spectrum.csv`
  - `reconstruction_diagnostics.csv`

Upload:

- Three mobility CSVs: `freq_hz,re0,im0,re1,im1,...` per file (same frequencies and sensor order).
- Flight CSV: `time_s,ch0,ch1,...` in **g**.
- Or ordered per-channel flight MAT files, one per channel, each with one structure containing `amp`, `t`, and `sr`.

Options: **g₀**, time window, Tikhonov **λ**, checkbox if **H** is already SI.

Python library helper for dummy FRFs:

```python
from force_recon import frf_io

frf_f, H_dummy = frf_io.build_ones_mobility(time_s=t, n_sensors=acc_g.shape[1])
```

### Example synthetic data

```bash
python3 examples/generate_synthetic_data.py
```

Then in the GUI use `examples/data/mobility_Fx.csv` (and **Fy** / **Fz**) plus `flight_segment.csv`, window e.g. **0.5–3.5 s**.

### MATLAB (flight CSV or per-channel MAT workflow)

`matlab/reconstruct_forces_from_flight_data.m` is the preferred MATLAB entrypoint for the real pipeline. It loads the same **Fx/Fy/Fz mobility CSVs** as the Python app and accepts either:

- A flight CSV: `time_s,ch0,ch1,...` in **g**
- An ordered list of per-channel `.mat` files, one file per flight channel

For `.mat` input, each file must contain exactly one structure with fields:

- `amp`: acceleration samples in **g**
- `t`: time vector in seconds
- `sr`: sample rate in Hz

The `.mat` file order must match the FRF sensor order. Mixed sample rates and
different channel lengths are allowed; the loaders align the channels over their
common overlapping time span and linearly resample onto the slowest channel rate.

The MATLAB pipeline runs the same windowed frequency-by-frequency inversion and can optionally write:

- `F_hat_spectrum.csv`
- `reconstruction_diagnostics.csv`

When `plot_results=true`, MATLAB now also opens a measured-versus-predicted acceleration
comparison for a selected response channel, using the reconstructed forces to synthesize
the model response.

Example:

```matlab
opts = struct( ...
    't_start', 0.5, ...
    't_end', 3.5, ...
    'fft_window', 'hann', ...
    'tikhonov_lambda', 0.0, ...
    'plot_channel_idx', 1, ...
    'show_progress', true, ...
    'progress_interval_sec', 2.0, ...
    'plot_results', true);

[result, inputs] = reconstruct_forces_from_flight_data( ...
    'mobility_Fx.csv', 'mobility_Fy.csv', 'mobility_Fz.csv', 'flight_segment.csv', opts);
```

Per-channel `.mat` example:

```matlab
flightMatFiles = {
    'accel_ch01.mat'
    'accel_ch02.mat'
    'accel_ch03.mat'
};

[result, inputs] = reconstruct_forces_from_flight_data( ...
    'mobility_Fx.csv', 'mobility_Fy.csv', 'mobility_Fz.csv', flightMatFiles, opts);
```

For longer MATLAB runs, `show_progress=true` prints stage-level timings plus periodic
solve-loop progress with elapsed time and ETA.

Repo-local runner:

```matlab
run('matlab/run_force_reconstruction_flight_example.m')
```

If you want the repo sample CSVs for MATLAB, generate them first with:

```bash
python3 examples/generate_synthetic_data.py
```

### Tests

```bash
python3 examples/generate_synthetic_csv.py   # creates examples/synthetic_data/ for integration test
python3 -m pytest tests/ -q
```

(All tests should pass; **12 passed** including the simply-supported plate reconstruction case, Python MAT-flight import coverage, and the dummy-`H=ones()` helper.)

### Library usage

```python
from force_recon import frf_io, pipeline
from force_recon.flight_io import load_flight_data

frf_f, H = frf_io.load_mobility_csv_triplet("H_x.csv", "H_y.csv", "H_z.csv")
t, acc_g, ch_names = load_flight_data(
    ["accel_ch01.mat", "accel_ch02.mat", "accel_ch03.mat"]
)
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
