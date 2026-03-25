# vibroacoustic_demos

Vibroacoustic demos plus a **force reconstruction** toolkit (`force_recon/`).

## Force reconstruction

Estimate interface forces from **flight accelerations (g)** and **mobility** **H** = *v/F* from NASTRAN-style CSVs. The standard Python workflow uses three translational unit-load files (**Fx**, **Fy**, **Fz**). The MATLAB flight pipeline also supports an arbitrary ordered list of mobility CSVs, which is useful for distributed interface models such as individual bolt loads. Flight data can come from a single CSV or ordered per-channel MATLAB `.mat` files. For workflow/plumbing tests, Python can also synthesize a dummy **H = ones()** mobility tensor from the flight timebase.

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

`matlab/reconstruct_forces_from_flight_data.m` is the preferred MATLAB entrypoint for the real pipeline. It accepts either the legacy **Fx/Fy/Fz** mobility triplet or an ordered list of one or more mobility CSVs, plus either:

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
- a NASTRAN `TABLED1` include file with real/imaginary force spectra in `N` or `lbf`
- a SOL111 replay-deck skeleton with `DAREA`, `RLOAD1`, `DLOAD`, and the `TABLED1` include

When `plot_results=true`, MATLAB now also opens a measured-versus-predicted acceleration
comparison for a selected response channel, using the reconstructed forces to synthesize
the model response. You can also enable all-channel overlays and a lambda sweep summary
to help pick a stable regularization level. The PSD overlays are plotted in `g^2/Hz`,
and the MATLAB all-channel view can also show per-channel PSD subplots plus a
channel-by-frequency dB error map. The PSD legends include measured/predicted RMS,
the measured PSD can be bracketed with `+/- 3 dB` guide lines, and the PSD axes/range
can be controlled with `plot_psd_xscale`, `plot_psd_yscale`, `plot_psd_fmin_hz`,
and `plot_psd_fmax_hz`. If your SOL111 FRFs are exported on coarse lines such as
`1 Hz`, set `solve_on_frf_grid=true` to solve only on those FRF lines instead of
every flight FFT bin. You can also iterate on the inverse problem by setting
`active_channel_idx` to solve with only a selected subset of response channels.
If your PSD plots start too high in frequency, increase `psd_nperseg`; the first
nonzero Welch bin is approximately `fs / psd_nperseg`.

Example:

```matlab
opts = struct( ...
    't_start', 0.5, ...
    't_end', 3.5, ...
    'fft_window', 'hann', ...
    'tikhonov_lambda', 0.0, ...
    'highpass_hz', 2.0, ...
    'highpass_order', 4, ...
    'solve_on_frf_grid', true, ...
    'active_channel_idx', [1 2 3], ...
    'plot_channel_idx', 1, ...
    'plot_all_channels', true, ...
    'plot_psd_error_map', true, ...
    'plot_psd_xscale', 'log', ...
    'plot_psd_yscale', 'log', ...
    'plot_psd_fmin_hz', 10.0, ...
    'plot_psd_fmax_hz', 3000.0, ...
    'psd_nperseg', 2048, ...
    'plot_lambda_sweep', true, ...
    'save_nastran_tabled1', 'reconstructed_forces_nastran.inc', ...
    'save_nastran_replay_bdf', 'reconstructed_forces_replay.bdf', ...
    'nastran_force_unit', 'lbf', ...
    'nastran_table_id_start', 1001, ...
    'nastran_grid_ids', 5000, ...
    'nastran_spc_sid', 1, ...
    'nastran_method_sid', 10, ...
    'nastran_freq_sid', 30, ...
    'nastran_sdamping_sid', 20, ...
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

Arbitrary load-case example for a 4-bolt `Fz` model:

```matlab
mobilityPaths = {
    'bolt01_Fz.csv'
    'bolt02_Fz.csv'
    'bolt03_Fz.csv'
    'bolt04_Fz.csv'
};

[result, inputs] = reconstruct_forces_from_flight_data( ...
    mobilityPaths, flightMatFiles, opts);
```

For longer MATLAB runs, `show_progress=true` prints stage-level timings plus periodic
solve-loop progress with elapsed time and ETA.

For flight data with low-frequency vehicle motion or bias drift, you can enable the
optional zero-phase high-pass preprocessing with `highpass_hz` and `highpass_order`.

Repo-local runner:

```matlab
run('matlab/run_force_reconstruction_flight_example.m')
```

If you want the repo sample CSVs for MATLAB, generate them first with:

```bash
python3 examples/generate_synthetic_data.py
```

### Workflow and terminology

This section summarizes the main concepts and options used by the Python and MATLAB workflows.

#### End-to-end workflow

1. Load the mobility files.
   - Standard Python / legacy MATLAB case: three files for unit `Fx`, `Fy`, and `Fz`
   - General MATLAB case: any ordered list of mobility CSVs representing the load cases you want to solve
2. Load flight acceleration data from either:
   - one CSV with `time_s,ch0,ch1,...`
   - or ordered per-channel MAT files with `amp`, `t`, and `sr`
3. Optionally apply a zero-phase high-pass filter to remove mean, drift, or low-frequency vehicle motion.
4. Select the analysis time window with `t_start` and `t_end`.
5. Convert the mobility FRFs to accelerance with:
   \[
   A(\omega) = j \omega H(\omega)
   \]
6. Compute the windowed one-sided FFT of the measured flight accelerations.
7. Interpolate the FRFs onto the FFT frequency grid.
8. Solve the force reconstruction independently at each frequency bin.
9. Reconstruct the model-predicted acceleration from the solved forces.
10. Review diagnostics such as residual, response MAC, singular values, and condition number.

#### Input formats and units

- Mobility CSV format:
  - `freq_hz,re0,im0,re1,im1,...`
  - one file per load case
  - in the standard 3-axis workflow: one file for `Fx`, one for `Fy`, one for `Fz`
  - real/imaginary pairs correspond to response channels in the same order as the flight channels
- Flight CSV format:
  - `time_s,ch0,ch1,ch2,...`
  - acceleration channels are expected in `g`
- Flight MAT format:
  - one file per channel
  - each file contains one structure with fields:
    - `amp`: acceleration in `g`
    - `t`: time vector in seconds
    - `sr`: sample rate in Hz
  - mixed sample rates are allowed; the loaders align all channels over the common overlap and resample to the slowest channel rate

If your NASTRAN model is in `lbf` and `in`, the expected mobility units are:

- mobility: `(in/s)/lbf`
- displacement FRF: `in/lbf`
- accelerance: `(in/s^2)/lbf`

This toolkit normally expects mobility as input. If you already have SI mobility, set `mobility_is_si=true` in MATLAB or the equivalent GUI option on the Python side.

The reconstructed force spectrum is solved internally in `N`. If you want to replay it
in an `lbf-in` SOL111 model, export with:

- `save_nastran_tabled1 = 'your_file.inc'`
- `nastran_force_unit = 'lbf'`

The MATLAB exporter writes one real/imaginary `TABLED1` pair per reconstructed load case.
Use all reconstructed load cases together in the same SOL111 replay run so the original
complex phase relationship is preserved.

If you also set `save_nastran_replay_bdf`, MATLAB writes a SOL111 replay-deck skeleton
that references the exported `TABLED1` file and combines all reconstructed loads with
one `DLOAD`. For the standard `Fx/Fy/Fz` case, a scalar `nastran_grid_ids` value is
expanded automatically to one grid with components `1,2,3`. For a distributed load model,
provide one `nastran_grid_ids` entry and one `nastran_components` entry per reconstructed
load case.

For workflow testing without measured FRFs, the Python GUIs and CLI also support a dummy `H = ones()` mode.

#### What the inversion is solving

At each frequency bin, the code solves:

\[
a_k \approx A_k F_k
\]

where:

- `A_k` is the complex accelerance matrix at one frequency, shaped `n_sensors x n_loads`
- `a_k` is the measured complex acceleration vector across sensors
- `F_k` is the unknown complex load vector for the chosen force/load cases

Because the system is typically overdetermined, the code uses a complex least-squares solve. The key step is the Hermitian transpose:

\[
A^H = \overline{A}^T
\]

not a plain transpose. This matters because the FRFs and FFT data are complex-valued, so the correct least-squares normal equations are:

\[
(A^H A) F = A^H a
\]

The MATLAB implementation uses `A'`, which is the Hermitian transpose for complex data, and the Python implementation uses `A.conj().T`.

#### What `lambda` means

`lambda` or `tikhonov_lambda` is the regularization parameter in the damped least-squares solve:

\[
(A^H A + \lambda^2 I) F = A^H a
\]

Interpretation:

- `lambda = 0`: plain least squares
- small `lambda`: closer fit to the measured data, but more sensitive to noisy or ill-conditioned bins
- larger `lambda`: more stable and smoother force estimates, but more biased and less able to exactly match the measured response

Practical guidance:

- start with `0`, `1e-9`, `1e-8`, `1e-7`, `1e-6`, `1e-5`
- choose the smallest value that removes unstable force spikes without materially hurting the measured-vs-predicted response match
- the MATLAB lambda sweep plot is a quick way to compare median response MAC, residual, and force magnitude across candidate values

#### Meaning of the diagnostics

- `relative_residual`:
  - \[
    \|a_{pred} - a_{meas}\| / \|a_{meas}\|
    \]
  - lower is better
  - measures how closely the reconstructed model response matches the measured acceleration at each frequency
- `response_mac`:
  - scalar MAC between the full measured and predicted sensor vectors at one frequency
  - ranges from `0` to `1`
  - values near `1` mean the predicted multi-sensor response shape matches the measured shape well
- `cond_number`:
  - the matrix condition number of `A`
  - larger values mean the inversion is more sensitive to small errors or noise
  - very large values often indicate that the solved forces may be unstable unless regularized
- `singular_values`:
  - singular values of `A`
  - show how much independent force-direction information is available at that frequency
  - a very small trailing singular value usually indicates weak observability in one load direction
- `max_column_mac`:
  - largest off-diagonal column-to-column MAC in the load-case matrix at that frequency
  - values near `1` mean at least two load cases produce very similar sensor response patterns, which makes force separation harder
- `mac_xy`, `mac_xz`, `mac_yz`:
  - legacy pairwise column MAC values retained for the standard 3-load `Fx/Fy/Fz` case

#### Meaning of the main options

- `t_start`, `t_end`:
  - time window used for the FFT and reconstruction
- `fft_window`:
  - window applied before the FFT, typically `hann` or `boxcar`
- `mobility_is_si`:
  - set `true` only if the input FRFs are already in `(m/s)/N`
- `skip_zero_hz`:
  - skips the DC bin, which is usually the right choice for this type of frequency-domain inversion
- `f_min_hz`, `f_max_hz`:
  - optional frequency limits applied after the FFT grid is formed
- `active_channel_idx`:
  - optional MATLAB-only list of active response channels, or a logical mask, used to subset both the flight data and FRF sensor columns before inversion
  - useful for leave-one-out tests or for dropping suspect channels to see how the reconstruction changes
- `psd_nperseg`, `psd_noverlap`:
  - optional MATLAB-only Welch PSD controls for the measured/predicted comparison plots
  - the first nonzero PSD bin is approximately `fs / psd_nperseg`, so higher `psd_nperseg` gives lower-frequency resolution
- `load_case_names`:
  - optional labels for the mobility files / reconstructed loads in MATLAB plots and CSV exports
- `solve_on_frf_grid`:
  - when `false`, the solve runs on the native flight FFT grid and the FRFs are interpolated onto it
  - when `true`, the solve runs only on the FRF frequency lines, which is useful when the model is exported on coarse spacing such as `1 Hz` from SOL111
- `highpass_hz`, `highpass_order`:
  - optional zero-phase high-pass preprocessing
  - useful for removing bias, gravity leakage, drift, or low-frequency vehicle rigid-body motion
  - a 4-pole filter at `2 Hz` is a reasonable starting point when low-frequency vehicle motion is contaminating the data
- `plot_channel_idx`:
  - selected channel for the single measured-vs-predicted comparison figure in MATLAB
  - when `active_channel_idx` is used, this refers to the active-channel subset
- `plot_all_channels`:
  - overlays all measured and predicted channels in MATLAB
- `plot_lambda_sweep`:
  - enables the MATLAB lambda sweep summary figure
- `lambda_sweep_values`:
  - lambda values evaluated in the MATLAB sweep
- `lambda_sweep_max_bins`:
  - limits how many frequency bins are used in the sweep summary so large datasets stay responsive
- `show_progress`, `progress_interval_sec`:
  - console progress/timing output for MATLAB, including solve-loop ETA updates
- `save_nastran_tabled1`:
  - optional MATLAB-only path for a NASTRAN include file containing one real/imaginary `TABLED1` pair per reconstructed load case
  - only the valid solved frequency bins are written
- `save_nastran_replay_bdf`:
  - optional MATLAB-only path for a SOL111 replay-deck skeleton
  - if `save_nastran_tabled1` is empty, MATLAB automatically writes a companion `*_tables.inc` file next to the replay deck
- `nastran_force_unit`:
  - force unit used in the MATLAB NASTRAN export
  - use `'lbf'` for an `lbf-in` model or `'N'` for an SI model
- `nastran_table_id_start`:
  - first `TABLED1` ID used by the MATLAB NASTRAN export
  - each load case consumes two IDs: one for the real part and one for the imaginary part
- `nastran_grid_ids`:
  - grid IDs used on the replay `DAREA` entries
  - provide one scalar for the standard 3-load `Fx/Fy/Fz` case, or one value per reconstructed load case
- `nastran_components`:
  - component IDs used on the replay `DAREA` entries
  - default = `[1 2 3]` for the standard 3-load case
- `nastran_darea_scales`:
  - optional `DAREA` scale factors for the replay deck
  - default = `1.0` for each reconstructed load case because the `TABLED1` values already contain the reconstructed force amplitude
- `nastran_darea_sid_start`, `nastran_rload_sid_start`, `nastran_dload_sid`:
  - starting IDs used for the replay deck `DAREA`, `RLOAD1`, and `DLOAD` entries
- `nastran_subcase_id`, `nastran_spc_sid`, `nastran_method_sid`, `nastran_freq_sid`, `nastran_sdamping_sid`:
  - case-control IDs written into the replay deck skeleton
  - adjust these to match your model setup
- `nastran_title`, `nastran_model_include`:
  - title and optional model `INCLUDE` path written into the replay deck skeleton

#### Measured versus predicted acceleration

Both the Python app and MATLAB workflow compute the model-predicted response after the force solve:

\[
a_{pred} = A F
\]

That predicted acceleration is compared against the measured acceleration in two ways:

- time-domain overlay after inverse FFT of the predicted one-sided spectrum
- PSD overlay for the selected channel and, in MATLAB, optionally for all channels

This is the main sanity check that the reconstructed forces are physically consistent with the measured response.

### Tests

```bash
python3 examples/generate_synthetic_csv.py   # creates examples/synthetic_data/ for integration test
python3 -m pytest tests/ -q
```

(All tests should pass; **21 passed** including the simply-supported plate reconstruction case, MAT-flight import coverage, NASTRAN export formatting checks, and pseudo PSD/CPSD response utilities.)

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

### Pseudo PSD / CPSD from replayed complex response

If you replay the reconstructed loads in deterministic `SOL111` and export complex response
as CSV in the same style

- `freq_hz,re0,im0,re1,im1,...`

you can build a pseudo auto-PSD and cross-PSD estimate with:

```bash
python3 compute_psd_from_complex_response.py replay_response.csv
```

This writes:

- `<input>_auto_psd.csv`
- `<input>_cpsd.csv`

By default the script assumes the complex harmonic response is a **peak-amplitude** phasor,
which is the usual interpretation for deterministic `SOL111` output. In that case the
one-sided pseudo spectral matrix is

\[
S_{xx}(f_k) \approx \frac{X(f_k) X(f_k)^H}{2 \Delta f}
\]

If your complex response is already in RMS form, use:

```bash
python3 compute_psd_from_complex_response.py replay_response.csv --phasor-convention rms
```

The output units are the square of the input response units per Hz. For example:

- input in `g` -> output in `g^2/Hz`
- input in `in/s^2` -> output in `(in/s^2)^2/Hz`

MATLAB has the same workflow:

```matlab
opts = struct();
opts.phasor_convention = 'peak';   % usual SOL111 interpretation
result = compute_psd_from_complex_response_csv('replay_response.csv', opts);
```

By default MATLAB writes:

- `replay_response_auto_psd.csv`
- `replay_response_cpsd.csv`

Set `opts.save_auto_psd_csv = ''` or `opts.save_cpsd_csv = ''` to skip either output.
