% run_force_reconstruction_flight_example.m
%
% Example driver for the real flight-data MATLAB pipeline.
% Replace the default CSV flight input with your own CSV or ordered .mat files.
% The legacy example below uses Fx/Fy/Fz mobility CSVs, but you can also pass
% an ordered cell array of arbitrary mobility CSVs to reconstruct multiple bolt
% or interface load cases.

clear;
clc;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
dataDir = fullfile(repoRoot, 'examples', 'data');

fxPath = fullfile(dataDir, 'mobility_Fx.csv');
fyPath = fullfile(dataDir, 'mobility_Fy.csv');
fzPath = fullfile(dataDir, 'mobility_Fz.csv');
mobilityInput = {fxPath, fyPath, fzPath};

% Replace mobilityInput with any ordered list of load-case CSVs if you want
% to reconstruct more than the legacy Fx/Fy/Fz set, for example:
% mobilityInput = {
%     '/path/to/bolt01_Fz.csv'
%     '/path/to/bolt02_Fz.csv'
%     '/path/to/bolt03_Fz.csv'
%     '/path/to/bolt04_Fz.csv'
% };

flightPath = fullfile(dataDir, 'flight_segment.csv');
flightInput = flightPath;

% For per-channel .mat flight inputs, pass an ordered cell array instead.
% Each .mat file must contain one structure with fields:
%   amp  acceleration samples in g
%   t    time vector in seconds
%   sr   sample rate in Hz
% Different channel sample rates are allowed; the loader aligns the channels
% over their common overlap and resamples onto the slowest channel rate.
% flightInput = {
%     '/path/to/ch01.mat'
%     '/path/to/ch02.mat'
%     '/path/to/ch03.mat'
% };

requiredPaths = mobilityInput(:).';
if ischar(flightInput) || (isstring(flightInput) && isscalar(flightInput))
    requiredPaths{end + 1} = char(flightInput);
elseif isstring(flightInput)
    requiredPaths = [requiredPaths, cellstr(flightInput(:).')];
elseif iscell(flightInput)
    requiredPaths = [requiredPaths, flightInput(:).'];
end
missingMask = ~cellfun(@isfile, requiredPaths);
if any(missingMask)
    missingPaths = strjoin(requiredPaths(missingMask), newline);
    error(['Required input files were not found.' newline ...
        'Run "python3 examples/generate_synthetic_data.py" from the repo root for the sample CSV set,' newline ...
        'or replace the file paths in this script with your real flight-data CSVs.' newline newline ...
        missingPaths]);
end

opts = struct();
opts.t_start = 0.5;
opts.t_end = 3.5;
opts.fft_window = 'hann';
opts.tikhonov_lambda = 0.0;
% Set true to solve only on the FRF frequency lines (for example, 1 Hz SOL111 data).
opts.solve_on_frf_grid = false;
% Optional channel subset for leave-one-out / sensor-screening studies.
% opts.active_channel_idx = [1 2 3 4 6 7 8];
opts.plot_results = true;
opts.plot_channel_idx = 1;
opts.plot_all_channels = true;
opts.plot_psd_error_map = true;
opts.plot_psd_xscale = 'log';
opts.plot_psd_yscale = 'log';
opts.plot_psd_fmin_hz = 10.0;
opts.plot_psd_fmax_hz = 3000.0;
% Increase psd_nperseg if the first nonzero PSD bin is too high in frequency.
% For example, fs / psd_nperseg ~= 10 Hz gives a first visible bin near 10 Hz.
opts.psd_nperseg = 2048;
opts.plot_lambda_sweep = true;
opts.verbose = true;
opts.show_progress = true;
opts.progress_interval_sec = 2.0;
% Optional preprocessing for low-frequency vehicle motion / bias drift.
% opts.highpass_hz = 2.0;
% opts.highpass_order = 4;

% Uncomment these to write CSV outputs from MATLAB.
% opts.save_force_csv = fullfile(repoRoot, 'F_hat_spectrum_matlab.csv');
% opts.save_diagnostics_csv = fullfile(repoRoot, 'reconstruction_diagnostics_matlab.csv');
% opts.recovery_mobility_paths = {
%     '/path/to/recovery_Fx.csv'
%     '/path/to/recovery_Fy.csv'
%     '/path/to/recovery_Fz.csv'
% };
% opts.save_recovery_psd_csv = fullfile(repoRoot, 'recovery_predicted_psd_matlab.csv');
% opts.recovery_channel_names = {'grid123_t3' 'grid456_t3'};
%
% Uncomment these to write a NASTRAN TABLED1 include file in model units.
% For an lbf-in model, set nastran_force_unit = 'lbf' and keep all
% reconstructed load cases active together in the same SOL111 replay run.
% opts.save_nastran_tabled1 = fullfile(repoRoot, 'reconstructed_forces_nastran.inc');
% opts.nastran_force_unit = 'lbf';
% opts.nastran_table_id_start = 1001;
%
% Uncomment these to also write a SOL111 replay-deck skeleton. For the
% standard Fx/Fy/Fz case, you can provide one grid ID and MATLAB will map
% the three reconstructed loads to components 1/2/3 on that grid.
% opts.save_nastran_replay_bdf = fullfile(repoRoot, 'reconstructed_forces_replay.bdf');
% opts.nastran_grid_ids = 5000;
% opts.nastran_spc_sid = 1;
% opts.nastran_method_sid = 10;
% opts.nastran_freq_sid = 30;
% opts.nastran_sdamping_sid = 20;
% opts.nastran_title = 'Reconstructed Force Replay';
% opts.nastran_model_include = 'your_model.bdf';

[result, inputs] = reconstruct_forces_from_flight_data( ...
    mobilityInput, flightInput, opts);

fprintf('Processed %d flight channels from %.3f s to %.3f s.\n', ...
    numel(inputs.channel_names), result.t_start, result.t_end);
