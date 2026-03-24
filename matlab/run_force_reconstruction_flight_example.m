% run_force_reconstruction_flight_example.m
%
% Example driver for the real flight-data MATLAB pipeline.
% Replace the default CSV flight input with your own CSV or ordered .mat files.

clear;
clc;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
dataDir = fullfile(repoRoot, 'examples', 'data');

fxPath = fullfile(dataDir, 'mobility_Fx.csv');
fyPath = fullfile(dataDir, 'mobility_Fy.csv');
fzPath = fullfile(dataDir, 'mobility_Fz.csv');
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

requiredPaths = {fxPath, fyPath, fzPath};
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
opts.plot_results = true;
opts.plot_channel_idx = 1;
opts.plot_all_channels = true;
opts.plot_psd_error_map = true;
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

[result, inputs] = reconstruct_forces_from_flight_data( ...
    fxPath, fyPath, fzPath, flightInput, opts);

fprintf('Processed %d flight channels from %.3f s to %.3f s.\n', ...
    numel(inputs.channel_names), result.t_start, result.t_end);
