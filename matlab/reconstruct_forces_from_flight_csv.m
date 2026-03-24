function [result, inputs] = reconstruct_forces_from_flight_csv(pathFx, pathFy, pathFz, flightInput, opts)
%RECONSTRUCT_FORCES_FROM_FLIGHT_CSV Reconstruct 3-axis interface forces from flight input data.
%
%   [result, inputs] = reconstruct_forces_from_flight_csv(pathFx, pathFy, pathFz, flightInput)
%   [result, inputs] = reconstruct_forces_from_flight_csv(pathFx, pathFy, pathFz, flightInput, opts)
%
% Inputs
%   pathFx, pathFy, pathFz
%       Mobility CSVs with columns:
%       freq_hz,re0,im0,re1,im1,...
%   flightInput
%       Either:
%       1) A flight CSV path with columns:
%          time_s,ch0,ch1,...
%       2) An ordered cell array / string array of per-channel .mat files.
%          Each .mat file must contain exactly one structure with fields:
%          amp   acceleration samples in g
%          t     time vector [s]
%          sr    sample rate [Hz]
%          The order of the .mat files must match the FRF sensor order.
%          Channels may have different sample rates and lengths; they are
%          aligned over their common overlapping time span and linearly
%          resampled onto the slowest channel sample rate.
%   opts (optional struct)
%       t_start               analysis start time [s], default = first sample
%       t_end                 analysis end time [s], default = last sample
%       g0                    standard gravity, default = 9.80665
%       tikhonov_lambda       force regularization lambda, default = 0
%       highpass_hz           optional high-pass cutoff [Hz], default = []
%       highpass_order        high-pass filter order, default = 4
%       mobility_is_si        true if H is already (m/s)/N, default = false
%       skip_zero_hz          skip the DC bin, default = true
%       f_min_hz              optional lower analysis limit
%       f_max_hz              optional upper analysis limit
%       fft_window            'hann' or 'boxcar', default = 'hann'
%       plot_results          plot reconstructed forces + diagnostics, default = false
%       plot_channel_idx      1-based channel index for acceleration comparison plots, default = 1
%       plot_all_channels     also plot all-channel measured/predicted overlays, default = true
%       plot_lambda_sweep     plot lambda sweep summary, default = false
%       lambda_sweep_values   lambda values for the sweep, default = [0 1e-9 1e-8 1e-7 1e-6 1e-5]
%       lambda_sweep_max_bins maximum bins used in the lambda sweep summary, default = 2000
%       verbose               print summary metrics, default = true
%       show_progress         print stage progress and timings, default = verbose
%       progress_interval_sec progress update period during solve loop, default = 2.0
%       save_force_csv        optional output path for F_hat CSV
%       save_diagnostics_csv  optional output path for diagnostics CSV
%
% Outputs
%   result
%       Struct aligned with the Python pipeline fields:
%       freqs_hz, F_hat, a_meas_fft, a_pred_fft, cond_number,
%       singular_values, mac_xy, mac_xz, mac_yz, response_mac,
%       relative_residual, valid_mask, fs_hz, t_start, t_end,
%       selected measured/predicted accelerations, preprocessing info,
%       and optional lambda-sweep diagnostics
%   inputs
%       Loaded source data and resolved options. When plot_results=true,
%       MATLAB also compares the measured and reconstructed acceleration
%       for opts.plot_channel_idx in time and PSD form.
%
% Example
%   opts = struct('t_start', 0.5, 't_end', 3.5, 'fft_window', 'hann', ...
%       'tikhonov_lambda', 0.0, 'plot_results', true);
%   [result, inputs] = reconstruct_forces_from_flight_csv( ...
%       'mobility_Fx.csv', 'mobility_Fy.csv', 'mobility_Fz.csv', 'flight.csv', opts);
%
%   matFiles = {'ch01.mat', 'ch02.mat', 'ch03.mat'};
%   [result, inputs] = reconstruct_forces_from_flight_csv( ...
%       'mobility_Fx.csv', 'mobility_Fy.csv', 'mobility_Fz.csv', matFiles, opts);

    if nargin < 5 || isempty(opts)
        opts = struct();
    end
    if ~isstruct(opts)
        error('opts must be a struct when provided.');
    end

    progress_enabled = resolve_preload_progress_option(opts);
    overall_timer = tic;
    timing = struct();

    if progress_enabled
        fprintf('\n');
        fprintf('Starting force reconstruction from flight input\n');
        fprintf('---------------------------------------------\n');
    end

    assert_file_exists(pathFx, 'pathFx');
    assert_file_exists(pathFy, 'pathFy');
    assert_file_exists(pathFz, 'pathFz');
    assert_flight_input_exists(flightInput, 'flightInput');

    stage_timer = tic;
    log_progress(progress_enabled, 'Loading mobility CSV triplet...\n');
    [frf_freqs_hz, H_input] = load_mobility_csv_triplet(pathFx, pathFy, pathFz);
    timing.load_mobility_sec = toc(stage_timer);
    log_progress(progress_enabled, 'Loaded mobility CSVs: %d frequencies, %d sensors in %s.\n', ...
        numel(frf_freqs_hz), size(H_input, 2), format_duration(timing.load_mobility_sec));

    stage_timer = tic;
    log_progress(progress_enabled, 'Loading flight input...\n');
    [time_s, acc_g, channel_names, flight_input_type] = load_flight_input(flightInput, progress_enabled);
    timing.load_flight_sec = toc(stage_timer);
    log_progress(progress_enabled, 'Loaded %s flight input: %d samples, %d channels in %s.\n', ...
        upper(flight_input_type), numel(time_s), size(acc_g, 2), format_duration(timing.load_flight_sec));

    opts = resolve_options(opts, time_s);
    progress_enabled = opts.show_progress;
    if ~isempty(opts.f_min_hz) && ~isempty(opts.f_max_hz) && opts.f_max_hz < opts.f_min_hz
        error('opts.f_max_hz must be >= opts.f_min_hz.');
    end
    if opts.t_end < opts.t_start
        error('opts.t_end must be >= opts.t_start.');
    end

    if size(H_input, 2) ~= size(acc_g, 2)
        error('H has %d sensors, but the flight input has %d channels.', size(H_input, 2), size(acc_g, 2));
    end
    if ~isscalar(opts.plot_channel_idx) || ~isnumeric(opts.plot_channel_idx) || ...
            ~isfinite(opts.plot_channel_idx) || opts.plot_channel_idx < 1 || ...
            opts.plot_channel_idx > size(acc_g, 2) || opts.plot_channel_idx ~= round(opts.plot_channel_idx)
        error('opts.plot_channel_idx must be an integer from 1 to %d.', size(acc_g, 2));
    end

    acc_g_raw = acc_g;
    stage_timer = tic;
    [acc_g, preprocessing] = preprocess_flight_acceleration(time_s, acc_g_raw, opts, progress_enabled);
    timing.preprocess_sec = toc(stage_timer);
    if preprocessing.highpass_enabled
        log_progress(progress_enabled, ...
            'Applied %d-pole high-pass filter at %.3f Hz (%s) in %s.\n', ...
            preprocessing.highpass_order, preprocessing.highpass_hz, preprocessing.method, ...
            format_duration(timing.preprocess_sec));
    else
        log_progress(progress_enabled, 'No high-pass filter applied.\n');
    end

    time_mask = time_s >= opts.t_start & time_s <= opts.t_end;
    time_sel = time_s(time_mask);
    acc_sel_g = acc_g(time_mask, :);

    if numel(time_sel) < 8
        error('Time slice too short for FFT. Need at least 8 samples after windowing.');
    end

    selected_duration = 0.0;
    if numel(time_sel) > 1
        selected_duration = time_sel(end) - time_sel(1);
    end
    log_progress(progress_enabled, ...
        'Selected analysis window %.6f to %.6f s: %d samples across %.6f s.\n', ...
        opts.t_start, opts.t_end, numel(time_sel), selected_duration);

    stage_timer = tic;
    log_progress(progress_enabled, 'Computing windowed FFT for %d channels...\n', size(acc_sel_g, 2));
    fs_hz = infer_fs(time_sel);
    acc_sel_si = acc_sel_g * opts.g0;
    [freqs_hz, a_meas_fft] = complex_rfft_matrix(acc_sel_si, fs_hz, opts.fft_window);
    timing.fft_sec = toc(stage_timer);
    log_progress(progress_enabled, 'Computed FFT: %d frequency bins at %.3f Hz sample rate in %s.\n', ...
        numel(freqs_hz), fs_hz, format_duration(timing.fft_sec));

    stage_timer = tic;
    log_progress(progress_enabled, 'Converting mobility to accelerance and interpolating onto FFT grid...\n');
    H_si = H_input;
    if ~opts.mobility_is_si
        H_si = mobility_imp_to_si(H_input);
    end

    A_frf = accelerance_from_mobility(H_si, 2.0 * pi * frf_freqs_hz);
    A_fft = interp_complex_frequency(freqs_hz, frf_freqs_hz, A_frf, complex(NaN, NaN));
    timing.interp_sec = toc(stage_timer);
    log_progress(progress_enabled, 'Prepared FRFs on FFT grid in %s.\n', format_duration(timing.interp_sec));

    n_freq = numel(freqs_hz);
    n_sensors = size(a_meas_fft, 2);

    F_hat = complex(zeros(n_freq, 3));
    a_pred_fft = complex(zeros(n_freq, n_sensors));
    cond_number = NaN(n_freq, 1);
    singular_values = NaN(n_freq, 3);
    mac_xy = NaN(n_freq, 1);
    mac_xz = NaN(n_freq, 1);
    mac_yz = NaN(n_freq, 1);
    response_mac_values = NaN(n_freq, 1);
    relative_residual = NaN(n_freq, 1);
    valid_mask = false(n_freq, 1);

    solve_mask = true(n_freq, 1);
    if opts.skip_zero_hz
        solve_mask = solve_mask & freqs_hz > 0;
    end
    if ~isempty(opts.f_min_hz)
        solve_mask = solve_mask & freqs_hz >= opts.f_min_hz;
    end
    if ~isempty(opts.f_max_hz)
        solve_mask = solve_mask & freqs_hz <= opts.f_max_hz;
    end
    finite_mask = all(isfinite(reshape(A_fft, n_freq, [])), 2);
    solve_indices = find(solve_mask & finite_mask);
    n_solve = numel(solve_indices);

    stage_timer = tic;
    last_progress_elapsed = -Inf;
    if n_solve == 0
        log_progress(progress_enabled, 'No valid frequency bins were available for reconstruction.\n');
    else
        log_progress(progress_enabled, 'Solving %d frequency bins...\n', n_solve);
    end

    for solve_idx = 1:n_solve
        k = solve_indices(solve_idx);
        fk = freqs_hz(k);
        Ak = reshape(A_fft(k, :, :), n_sensors, 3);

        ak = a_meas_fft(k, :).';
        [mac_xy(k), mac_xz(k), mac_yz(k)] = column_mac_pairs(Ak);

        s = svd(Ak, 'econ');
        singular_values(k, 1:numel(s)) = s(:).';
        if s(end) > 0 && isfinite(s(end))
            cond_number(k) = s(1) / s(end);
        else
            cond_number(k) = Inf;
        end

        Fk = solve_force_complex(Ak, ak, opts.tikhonov_lambda);
        a_pred = Ak * Fk;

        F_hat(k, :) = Fk.';
        a_pred_fft(k, :) = a_pred.';

        denom = norm(ak);
        if denom > 0
            relative_residual(k) = norm(a_pred - ak) / denom;
        else
            relative_residual(k) = 0.0;
        end

        response_mac_values(k) = response_mac(ak, a_pred);
        valid_mask(k) = true;

        if progress_enabled
            elapsed_solve_sec = toc(stage_timer);
            if solve_idx == 1 || solve_idx == n_solve || ...
                    (elapsed_solve_sec - last_progress_elapsed) >= opts.progress_interval_sec
                eta_sec = 0.0;
                if solve_idx < n_solve
                    eta_sec = elapsed_solve_sec * (n_solve - solve_idx) / solve_idx;
                end
                fprintf('  Solve %d/%d (%.1f%%) | f = %.3f Hz | elapsed %s | ETA %s\n', ...
                    solve_idx, n_solve, 100.0 * solve_idx / n_solve, fk, ...
                    format_duration(elapsed_solve_sec), format_duration(eta_sec));
                last_progress_elapsed = elapsed_solve_sec;
            end
        end
    end
    timing.solve_sec = toc(stage_timer);
    if n_solve > 0
        log_progress(progress_enabled, 'Completed frequency solve in %s.\n', format_duration(timing.solve_sec));
    end

    acc_pred_sel_si = real(irfft_matrix(a_pred_fft, numel(time_sel)));
    acc_pred_sel_g = acc_pred_sel_si / opts.g0;

    lambda_sweep = struct();
    if opts.plot_lambda_sweep
        stage_timer = tic;
        log_progress(progress_enabled, 'Evaluating lambda sweep diagnostics...\n');
        lambda_sweep = evaluate_lambda_sweep(A_fft, a_meas_fft, freqs_hz, solve_indices, n_sensors, opts, progress_enabled);
        timing.lambda_sweep_sec = toc(stage_timer);
        if isfield(lambda_sweep, 'n_eval_bins') && lambda_sweep.n_eval_bins > 0
            log_progress(progress_enabled, 'Evaluated lambda sweep using %d bins in %s.\n', ...
                lambda_sweep.n_eval_bins, format_duration(timing.lambda_sweep_sec));
        else
            log_progress(progress_enabled, 'Skipped lambda sweep because no valid bins were available.\n');
        end
    end

    result = struct();
    result.freqs_hz = freqs_hz;
    result.F_hat = F_hat;
    result.a_meas_fft = a_meas_fft;
    result.a_pred_fft = a_pred_fft;
    result.cond_number = cond_number;
    result.singular_values = singular_values;
    result.mac_xy = mac_xy;
    result.mac_xz = mac_xz;
    result.mac_yz = mac_yz;
    result.response_mac = response_mac_values;
    result.relative_residual = relative_residual;
    result.valid_mask = valid_mask;
    result.fs_hz = fs_hz;
    result.t_start = opts.t_start;
    result.t_end = opts.t_end;
    result.time_sel_s = time_sel;
    result.acc_meas_sel_g = acc_sel_g;
    result.acc_meas_sel_si = acc_sel_si;
    result.acc_pred_sel_g = acc_pred_sel_g;
    result.acc_pred_sel_si = acc_pred_sel_si;
    result.preprocessing = preprocessing;
    result.lambda_sweep = lambda_sweep;
    timing.reconstruction_sec = toc(overall_timer);
    result.timing = timing;

    inputs = struct();
    inputs.path_fx = pathFx;
    inputs.path_fy = pathFy;
    inputs.path_fz = pathFz;
    inputs.path_flight = flightInput;
    inputs.flight_input = flightInput;
    inputs.flight_input_type = flight_input_type;
    inputs.frf_freqs_hz = frf_freqs_hz;
    inputs.H_input = H_input;
    inputs.acc_g_raw = acc_g_raw;
    inputs.time_s = time_s;
    inputs.acc_g = acc_g;
    inputs.channel_names = channel_names;
    inputs.options = opts;
    inputs.preprocessing = preprocessing;
    inputs.timing = timing;

    if ~isempty(opts.save_force_csv)
        stage_timer = tic;
        log_progress(progress_enabled, 'Writing force spectrum CSV: %s\n', opts.save_force_csv);
        write_force_spectrum_csv(result.freqs_hz, result.F_hat, result.valid_mask, opts.save_force_csv);
        log_progress(progress_enabled, 'Wrote force spectrum CSV in %s.\n', format_duration(toc(stage_timer)));
    end
    if ~isempty(opts.save_diagnostics_csv)
        stage_timer = tic;
        log_progress(progress_enabled, 'Writing diagnostics CSV: %s\n', opts.save_diagnostics_csv);
        write_reconstruction_diagnostics_csv(result, opts.save_diagnostics_csv);
        log_progress(progress_enabled, 'Wrote diagnostics CSV in %s.\n', format_duration(toc(stage_timer)));
    end
    if opts.plot_results
        stage_timer = tic;
        log_progress(progress_enabled, 'Rendering MATLAB figures...\n');
        plot_reconstruction_results(result, inputs);
        log_progress(progress_enabled, 'Rendered figures in %s.\n', format_duration(toc(stage_timer)));
    end

    timing.total_sec = toc(overall_timer);
    result.timing = timing;
    inputs.timing = timing;
    if opts.verbose
        print_summary(result);
    end
    log_progress(progress_enabled, 'Total runtime: %s.\n', format_duration(timing.total_sec));
end


function opts = resolve_options(opts, time_s)
    defaults = struct();
    defaults.t_start = time_s(1);
    defaults.t_end = time_s(end);
    defaults.g0 = 9.80665;
    defaults.tikhonov_lambda = 0.0;
    defaults.highpass_hz = [];
    defaults.highpass_order = 4;
    defaults.mobility_is_si = false;
    defaults.skip_zero_hz = true;
    defaults.f_min_hz = [];
    defaults.f_max_hz = [];
    defaults.fft_window = 'hann';
    defaults.plot_results = false;
    defaults.plot_channel_idx = 1;
    defaults.plot_all_channels = true;
    defaults.plot_lambda_sweep = false;
    defaults.lambda_sweep_values = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5];
    defaults.lambda_sweep_max_bins = 2000;
    defaults.verbose = true;
    defaults.show_progress = [];
    defaults.progress_interval_sec = 2.0;
    defaults.save_force_csv = '';
    defaults.save_diagnostics_csv = '';

    names = fieldnames(defaults);
    for idx = 1:numel(names)
        name = names{idx};
        if ~isfield(opts, name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
    end

    if isempty(opts.show_progress)
        opts.show_progress = logical(opts.verbose);
    else
        opts.show_progress = logical(opts.show_progress);
    end
    opts.plot_all_channels = logical(opts.plot_all_channels);
    opts.plot_lambda_sweep = logical(opts.plot_lambda_sweep);

    if ~isempty(opts.highpass_hz)
        if ~isscalar(opts.highpass_hz) || ~isnumeric(opts.highpass_hz) || ...
                ~isfinite(opts.highpass_hz) || opts.highpass_hz < 0
            error('opts.highpass_hz must be empty or a non-negative finite scalar.');
        end
        if opts.highpass_hz == 0
            opts.highpass_hz = [];
        end
    end
    if ~isscalar(opts.highpass_order) || ~isnumeric(opts.highpass_order) || ...
            ~isfinite(opts.highpass_order) || opts.highpass_order < 1 || ...
            opts.highpass_order ~= round(opts.highpass_order)
        error('opts.highpass_order must be a positive integer.');
    end
    if ~isnumeric(opts.lambda_sweep_values) || any(~isfinite(opts.lambda_sweep_values(:))) || ...
            any(opts.lambda_sweep_values(:) < 0)
        error('opts.lambda_sweep_values must be a numeric vector of non-negative finite values.');
    end
    opts.lambda_sweep_values = unique(double(opts.lambda_sweep_values(:)).', 'stable');
    if ~isscalar(opts.lambda_sweep_max_bins) || ~isnumeric(opts.lambda_sweep_max_bins) || ...
            ~isfinite(opts.lambda_sweep_max_bins) || opts.lambda_sweep_max_bins < 1 || ...
            opts.lambda_sweep_max_bins ~= round(opts.lambda_sweep_max_bins)
        error('opts.lambda_sweep_max_bins must be a positive integer.');
    end
    if ~isscalar(opts.progress_interval_sec) || ~isnumeric(opts.progress_interval_sec) || ...
            ~isfinite(opts.progress_interval_sec) || opts.progress_interval_sec <= 0
        error('opts.progress_interval_sec must be a positive finite scalar.');
    end
end


function [acc_out_g, preprocessing] = preprocess_flight_acceleration(time_s, acc_in_g, opts, progress_enabled)
    acc_out_g = acc_in_g;
    preprocessing = struct();
    preprocessing.highpass_enabled = false;
    preprocessing.highpass_hz = [];
    preprocessing.highpass_order = opts.highpass_order;
    preprocessing.method = 'none';
    preprocessing.fs_hz = infer_fs(time_s);

    if isempty(opts.highpass_hz)
        return;
    end

    if opts.highpass_hz >= 0.5 * preprocessing.fs_hz
        error('opts.highpass_hz must be below Nyquist (%.6f Hz).', 0.5 * preprocessing.fs_hz);
    end

    if progress_enabled
        fprintf('Applying high-pass filter at %.3f Hz (order %d)...\n', opts.highpass_hz, opts.highpass_order);
    end

    if exist('butter', 'file') == 2 && exist('filtfilt', 'file') == 2
        try
            [b, a] = butter(opts.highpass_order, opts.highpass_hz / (0.5 * preprocessing.fs_hz), 'high');
            acc_out_g = filtfilt(b, a, double(acc_in_g));
            preprocessing.method = 'filtfilt';
        catch
            acc_out_g = zero_phase_fft_highpass(acc_in_g, preprocessing.fs_hz, opts.highpass_hz, opts.highpass_order);
            preprocessing.method = 'fft_butterworth';
        end
    else
        acc_out_g = zero_phase_fft_highpass(acc_in_g, preprocessing.fs_hz, opts.highpass_hz, opts.highpass_order);
        preprocessing.method = 'fft_butterworth';
    end

    preprocessing.highpass_enabled = true;
    preprocessing.highpass_hz = opts.highpass_hz;
end


function filtered = zero_phase_fft_highpass(x, fs_hz, cutoff_hz, order)
    if isvector(x)
        x = x(:);
    end

    n = size(x, 1);
    n_channels = size(x, 2);
    freqs_hz = (0:floor(n / 2)).' * (fs_hz / n);
    transfer = zeros(numel(freqs_hz), 1);
    positive_mask = freqs_hz > 0;
    transfer(positive_mask) = 1.0 ./ sqrt(1.0 + (cutoff_hz ./ freqs_hz(positive_mask)) .^ (2 * order));

    if mod(n, 2) == 0
        mirrored = transfer(end-1:-1:2);
    else
        mirrored = transfer(end:-1:2);
    end
    transfer_full = [transfer; mirrored];

    filtered = zeros(n, n_channels);
    for channel_idx = 1:n_channels
        X = fft(double(x(:, channel_idx)));
        filtered(:, channel_idx) = real(ifft(X .* transfer_full));
    end
end


function lambda_sweep = evaluate_lambda_sweep(A_fft, a_meas_fft, freqs_hz, solve_indices, n_sensors, opts, progress_enabled)
    lambda_sweep = struct();
    lambda_sweep.lambda_values = opts.lambda_sweep_values(:);
    lambda_sweep.n_eval_bins = 0;
    lambda_sweep.eval_freqs_hz = [];
    lambda_sweep.median_response_mac = [];
    lambda_sweep.median_relative_residual = [];
    lambda_sweep.median_force_norm = [];

    if isempty(solve_indices) || isempty(opts.lambda_sweep_values)
        return;
    end

    n_eval = min(numel(solve_indices), opts.lambda_sweep_max_bins);
    eval_positions = unique(round(linspace(1, numel(solve_indices), n_eval)));
    eval_indices = solve_indices(eval_positions);
    lambda_values = opts.lambda_sweep_values(:);

    lambda_sweep.n_eval_bins = numel(eval_indices);
    lambda_sweep.eval_freqs_hz = freqs_hz(eval_indices);
    lambda_sweep.median_response_mac = NaN(numel(lambda_values), 1);
    lambda_sweep.median_relative_residual = NaN(numel(lambda_values), 1);
    lambda_sweep.median_force_norm = NaN(numel(lambda_values), 1);

    for lambda_idx = 1:numel(lambda_values)
        lam = lambda_values(lambda_idx);
        if progress_enabled
            fprintf('  Lambda sweep %d/%d: lambda = %s\n', ...
                lambda_idx, numel(lambda_values), format_lambda_value(lam));
        end

        response_mac_values = NaN(numel(eval_indices), 1);
        relative_residual_values = NaN(numel(eval_indices), 1);
        force_norm_values = NaN(numel(eval_indices), 1);

        for eval_idx = 1:numel(eval_indices)
            k = eval_indices(eval_idx);
            Ak = reshape(A_fft(k, :, :), n_sensors, 3);
            ak = a_meas_fft(k, :).';
            Fk = solve_force_complex(Ak, ak, lam);
            a_pred = Ak * Fk;

            response_mac_values(eval_idx) = response_mac(ak, a_pred);
            denom = norm(ak);
            if denom > 0
                relative_residual_values(eval_idx) = norm(a_pred - ak) / denom;
            else
                relative_residual_values(eval_idx) = 0.0;
            end
            force_norm_values(eval_idx) = norm(Fk);
        end

        lambda_sweep.median_response_mac(lambda_idx) = median_finite(response_mac_values);
        lambda_sweep.median_relative_residual(lambda_idx) = median_finite(relative_residual_values);
        lambda_sweep.median_force_norm(lambda_idx) = median_finite(force_norm_values);
    end
end


function assert_file_exists(pathValue, argName)
    if ~isfile(pathValue)
        error('%s does not exist: %s', argName, pathValue);
    end
end


function assert_flight_input_exists(flightInput, argName)
    if ischar(flightInput) || (isstring(flightInput) && isscalar(flightInput))
        assert_file_exists(char(flightInput), argName);
        return;
    end

    if isstring(flightInput)
        flightInput = cellstr(flightInput(:));
    end

    if ~iscell(flightInput) || isempty(flightInput)
        error('%s must be a CSV path or a non-empty list of .mat file paths.', argName);
    end

    for idx = 1:numel(flightInput)
        pathValue = flightInput{idx};
        if ~(ischar(pathValue) || (isstring(pathValue) && isscalar(pathValue)))
            error('%s{%d} must be a file path.', argName, idx);
        end
        assert_file_exists(char(pathValue), sprintf('%s{%d}', argName, idx));
    end
end


function [freqs_hz, H] = load_mobility_csv_triplet(pathFx, pathFy, pathFz)
    paths = {pathFx, pathFy, pathFz};
    freqs_hz = [];
    stacks = cell(1, 3);

    for idx = 1:numel(paths)
        data = read_numeric_csv(paths{idx}, 1);
        if mod(size(data, 2) - 1, 2) ~= 0
            error('Expected freq + re/im pairs in %s.', paths{idx});
        end
        freq_this = data(:, 1);
        complex_cols = complex(data(:, 2:2:end), data(:, 3:2:end));

        if isempty(freqs_hz)
            freqs_hz = freq_this;
        elseif ~isequal(size(freqs_hz), size(freq_this)) || any(abs(freqs_hz - freq_this) > 1e-9)
            error('Frequency column mismatch across mobility CSVs.');
        end

        stacks{idx} = complex_cols;
    end

    H = cat(3, stacks{1}, stacks{2}, stacks{3});
end


function [time_s, acc_g, names, input_type] = load_flight_input(flightInput, show_progress)
    if nargin < 2
        show_progress = false;
    end

    if ischar(flightInput) || (isstring(flightInput) && isscalar(flightInput))
        pathValue = char(flightInput);
        [~, ~, ext] = fileparts(pathValue);
        if strcmpi(ext, '.mat')
            [time_s, acc_g, names] = load_flight_mat_files({pathValue}, show_progress);
            input_type = 'mat';
        else
            [time_s, acc_g, names] = load_flight_csv(pathValue);
            input_type = 'csv';
        end
        return;
    end

    if isstring(flightInput)
        flightInput = cellstr(flightInput(:));
    end

    if iscell(flightInput)
        [time_s, acc_g, names] = load_flight_mat_files(flightInput, show_progress);
        input_type = 'mat';
        return;
    end

    error('Unsupported flightInput type. Use a CSV path or an ordered list of .mat file paths.');
end


function [time_s, acc_g, names] = load_flight_csv(pathFlight)
    header = read_header_fields(pathFlight);
    data = read_numeric_csv(pathFlight, 1);
    if size(data, 2) < 2
        error('Flight CSV must contain time plus at least one channel.');
    end

    time_s = data(:, 1);
    acc_g = data(:, 2:end);

    if numel(header) == size(data, 2)
        names = header(2:end);
    else
        names = default_channel_names(size(acc_g, 2));
    end
end


function [time_s, acc_g, names] = load_flight_mat_files(matPaths, show_progress)
    if nargin < 2
        show_progress = false;
    end

    if isstring(matPaths)
        matPaths = cellstr(matPaths(:));
    end
    if isempty(matPaths)
        error('At least one .mat flight channel file is required.');
    end

    n_channels = numel(matPaths);
    names = cell(1, n_channels);
    amp_cols = cell(1, n_channels);
    time_cols = cell(1, n_channels);
    sr_values = zeros(1, n_channels);
    source_labels = cell(1, n_channels);

    for idx = 1:n_channels
        pathValue = char(matPaths{idx});
        [~, ~, ext] = fileparts(pathValue);
        if ~strcmpi(ext, '.mat')
            error('Per-channel flight inputs must be .mat files. Received: %s', pathValue);
        end

        if show_progress
            fprintf('  Loading flight MAT channel %d/%d: %s\n', idx, n_channels, pathValue);
        end
        [amp_g, t_this, sr_this] = load_single_channel_mat(pathValue);
        [~, baseName, ~] = fileparts(pathValue);
        names{idx} = baseName;
        amp_cols{idx} = amp_g;
        time_cols{idx} = t_this;
        sr_values(idx) = sr_this;
        source_labels{idx} = pathValue;
    end

    [time_s, acc_g] = align_flight_channels(amp_cols, time_cols, sr_values, source_labels);
    if show_progress
        fprintf('  Aligned %d channel(s) to %.3f Hz common grid: %d samples from %.6f to %.6f s.\n', ...
            n_channels, min(sr_values), numel(time_s), time_s(1), time_s(end));
    end
end


function [amp_g, time_s, sr_hz] = load_single_channel_mat(pathValue)
    loaded = load(pathValue);
    fieldNames = fieldnames(loaded);
    if isempty(fieldNames)
        error('No variables were found in %s.', pathValue);
    end

    selectedField = '';
    for idx = 1:numel(fieldNames)
        candidate = loaded.(fieldNames{idx});
        if isstruct(candidate) && isscalar(candidate) && has_required_flight_fields(candidate)
            if ~isempty(selectedField)
                error('Expected exactly one flight-data structure in %s.', pathValue);
            end
            selectedField = fieldNames{idx};
        end
    end

    if isempty(selectedField)
        error('Could not find a structure with fields amp, t, sr in %s.', pathValue);
    end

    data = loaded.(selectedField);
    if ~isnumeric(data.amp) || ~isvector(data.amp)
        error('Field "amp" in %s must be a numeric vector.', pathValue);
    end
    amp_g = double(data.amp(:));
    if isempty(amp_g) || any(~isfinite(amp_g))
        error('Field "amp" in %s must be a finite numeric vector.', pathValue);
    end

    if ~isnumeric(data.sr) || numel(data.sr) ~= 1
        error('Field "sr" in %s must be a numeric scalar.', pathValue);
    end
    sr_hz = double(data.sr);
    if ~isfinite(sr_hz) || sr_hz <= 0
        error('Field "sr" in %s must be a positive finite scalar.', pathValue);
    end

    if ~isempty(data.t) && (~isnumeric(data.t) || ~isvector(data.t))
        error('Field "t" in %s must be a numeric vector.', pathValue);
    end
    time_s = double(data.t(:));
    if isempty(time_s)
        time_s = (0:numel(amp_g)-1).' / sr_hz;
    end
    if numel(time_s) ~= numel(amp_g) || any(~isfinite(time_s))
        error('Field "t" in %s must be a finite vector with the same length as "amp".', pathValue);
    end
    if numel(time_s) > 1
        inferred_sr = infer_fs(time_s);
        sr_tol = max(1e-9, 1e-6 * max(abs(sr_hz), 1.0));
        if abs(inferred_sr - sr_hz) > sr_tol
            error('Field "sr" in %s does not match the spacing in field "t".', pathValue);
        end
    end
end


function tf = has_required_flight_fields(value)
    tf = isfield(value, 'amp') && isfield(value, 't') && isfield(value, 'sr');
end


function [time_s, acc_g] = align_flight_channels(amp_cols, time_cols, sr_values, source_labels)
    n_channels = numel(amp_cols);
    if n_channels == 0
        error('At least one .mat flight channel file is required.');
    end

    if n_channels == 1
        time_s = time_cols{1};
        acc_g = amp_cols{1};
        return;
    end

    start_times = cellfun(@(t) t(1), time_cols);
    end_times = cellfun(@(t) t(end), time_cols);
    start_common = max(start_times);
    end_common = min(end_times);
    target_fs = min(sr_values);
    dt = 1.0 / target_fs;
    time_tol = max(1e-12, 1e-9 * max([abs(start_common), abs(end_common), 1.0]));

    if end_common <= start_common + time_tol
        ranges = cell(1, n_channels);
        for idx = 1:n_channels
            ranges{idx} = sprintf('%s [%.9g, %.9g]', source_labels{idx}, time_cols{idx}(1), time_cols{idx}(end));
        end
        error('MAT flight channels do not share a common overlapping time span: %s', strjoin(ranges, ', '));
    end

    n_samples = floor((end_common - start_common) / dt + time_tol / dt) + 1;
    time_s = start_common + (0:n_samples-1).' * dt;
    time_s = time_s(time_s <= end_common + time_tol);
    if numel(time_s) < 2
        error('MAT flight channel overlap is too short after alignment; need at least two shared samples.');
    end

    acc_g = zeros(numel(time_s), n_channels);
    for idx = 1:n_channels
        acc_interp = interp1(time_cols{idx}, amp_cols{idx}, time_s, 'linear');
        if any(~isfinite(acc_interp))
            error('Channel file %s could not be interpolated onto the common time grid.', source_labels{idx});
        end
        acc_g(:, idx) = acc_interp;
    end
end


function fields = read_header_fields(pathValue)
    fid = fopen(pathValue, 'r');
    if fid < 0
        error('Could not open %s for reading.', pathValue);
    end
    cleanup = onCleanup(@() fclose(fid));
    first_line = fgetl(fid);
    if ~ischar(first_line)
        fields = {};
        return;
    end

    first_line = regexprep(first_line, '^\xEF\xBB\xBF', '');
    raw_fields = strsplit(strtrim(first_line), ',');
    fields = cellfun(@strtrim, raw_fields, 'UniformOutput', false);
end


function names = default_channel_names(n_channels)
    names = arrayfun(@(k) sprintf('ch%d', k - 1), 1:n_channels, 'UniformOutput', false);
end


function data = read_numeric_csv(pathValue, header_lines)
    try
        data = readmatrix(pathValue, 'NumHeaderLines', header_lines);
    catch
        data = dlmread(pathValue, ',', header_lines, 0);
    end

    if isvector(data)
        data = reshape(data, 1, []);
    end

    finite_row_mask = any(isfinite(data), 2);
    data = data(finite_row_mask, :);

    if isempty(data)
        error('No numeric data found in %s.', pathValue);
    end
end


function fs_hz = infer_fs(time_s)
    if numel(time_s) < 2
        error('Need at least two time samples to infer sample rate.');
    end

    dt = diff(time_s(:));
    if any(dt <= 0)
        error('Time vector must be strictly increasing.');
    end
    med_dt = median(dt);
    fs_hz = 1.0 / med_dt;
end


function [freqs_hz, spectrum] = complex_rfft_matrix(x, fs_hz, window_name)
    if isvector(x)
        x = x(:);
    end

    n = size(x, 1);
    n_channels = size(x, 2);
    win = get_window_vector(window_name, n);

    window_energy = sum(win .^ 2);
    if window_energy > 0
        scale = sqrt(n / window_energy);
    else
        scale = 1.0;
    end

    freqs_hz = (0:floor(n / 2)).' * (fs_hz / n);
    spectrum = complex(zeros(numel(freqs_hz), n_channels));

    for channel_idx = 1:n_channels
        u = x(:, channel_idx) .* win * scale;
        X = fft(u);
        spectrum(:, channel_idx) = X(1:numel(freqs_hz));
    end
end


function win = get_window_vector(window_name, n)
    switch lower(window_name)
        case {'boxcar', 'rect', 'rectangular'}
            win = ones(n, 1);
        case 'hann'
            if n == 1
                win = 1;
            else
                idx = (0:n-1).';
                win = 0.5 - 0.5 * cos(2.0 * pi * idx / n);
            end
        otherwise
            error('Unsupported fft_window "%s". Use "hann" or "boxcar".', window_name);
    end
end


function H_si = mobility_imp_to_si(H_imperial)
    lbf_to_n = 4.4482216152605;
    in_to_m = 0.0254;
    H_si = H_imperial * (in_to_m / lbf_to_n);
end


function A = accelerance_from_mobility(H, omega_rad_s)
    omega = reshape(omega_rad_s(:), [], 1, 1);
    A = 1i * omega .* H;
end


function out = interp_complex_frequency(f_target_hz, f_source_hz, values, fill_value)
    original_size = size(values);
    n_source = original_size(1);
    n_columns = prod(original_size(2:end));

    values_2d = reshape(values, n_source, n_columns);
    out_2d = complex(zeros(numel(f_target_hz), n_columns));

    for col_idx = 1:n_columns
        real_part = interp1(f_source_hz, real(values_2d(:, col_idx)), f_target_hz, 'linear', NaN);
        imag_part = interp1(f_source_hz, imag(values_2d(:, col_idx)), f_target_hz, 'linear', NaN);
        out_2d(:, col_idx) = complex(real_part, imag_part);
    end

    if ~(isnan(real(fill_value)) || isnan(imag(fill_value)))
        bad_mask = isnan(real(out_2d)) | isnan(imag(out_2d));
        out_2d(bad_mask) = fill_value;
    end

    out = reshape(out_2d, [numel(f_target_hz), original_size(2:end)]);
end


function [mac_xy, mac_xz, mac_yz] = column_mac_pairs(A)
    M = zeros(3, 3);
    for i = 1:3
        ci = A(:, i);
        nii = real(ci' * ci);
        if nii <= 0
            continue;
        end
        for j = 1:3
            cj = A(:, j);
            njj = real(cj' * cj);
            if njj <= 0
                continue;
            end
            cross = ci' * cj;
            M(i, j) = abs(cross) .^ 2 / (nii * njj);
        end
    end

    mac_xy = M(1, 2);
    mac_xz = M(1, 3);
    mac_yz = M(2, 3);
end


function F = solve_force_complex(A, a, lam)
    AH = A';
    ATA = AH * A;
    ATa = AH * a;

    if lam > 0
        ATA = ATA + (lam ^ 2) * eye(3);
    end

    if rcond(ATA) > 1e-12
        F = ATA \ ATa;
    else
        F = A \ a;
    end
end


function value = response_mac(a_meas, a_pred)
    nxx = real(a_meas' * a_meas);
    nyy = real(a_pred' * a_pred);
    if nxx <= 0 || nyy <= 0
        value = 0.0;
        return;
    end

    cross = a_meas' * a_pred;
    value = abs(cross) .^ 2 / (nxx * nyy);
end


function print_summary(result)
    valid = result.valid_mask;
    fprintf('\n');
    fprintf('Force reconstruction from flight input\n');
    fprintf('-------------------------------------\n');
    if isfield(result, 'preprocessing') && result.preprocessing.highpass_enabled
        fprintf('High-pass filter            : %d-pole %.3f Hz (%s)\n', ...
            result.preprocessing.highpass_order, result.preprocessing.highpass_hz, result.preprocessing.method);
    end
    fprintf('Valid frequency bins         : %d / %d\n', nnz(valid), numel(valid));
    fprintf('Median response MAC          : %.6f\n', median_finite(result.response_mac(valid)));
    fprintf('Median relative residual     : %.6f\n', median_finite(result.relative_residual(valid)));
    fprintf('Median condition number      : %.6f\n', median_finite(result.cond_number(valid)));
    if isfield(result, 'lambda_sweep') && isfield(result.lambda_sweep, 'n_eval_bins') && result.lambda_sweep.n_eval_bins > 0
        fprintf('Lambda sweep eval bins       : %d\n', result.lambda_sweep.n_eval_bins);
    end
    if isfield(result, 'timing')
        if isfield(result.timing, 'solve_sec')
            fprintf('Solve time                  : %s\n', format_duration(result.timing.solve_sec));
        end
        if isfield(result.timing, 'total_sec')
            fprintf('Total runtime               : %s\n', format_duration(result.timing.total_sec));
        elseif isfield(result.timing, 'reconstruction_sec')
            fprintf('Reconstruction runtime      : %s\n', format_duration(result.timing.reconstruction_sec));
        end
    end
    fprintf('\n');
end


function value = median_finite(x)
    finite_values = x(isfinite(x));
    if isempty(finite_values)
        value = NaN;
    else
        value = median(finite_values);
    end
end


function enabled = resolve_preload_progress_option(opts)
    enabled = true;
    if isfield(opts, 'show_progress') && ~isempty(opts.show_progress)
        enabled = logical(opts.show_progress);
        return;
    end
    if isfield(opts, 'verbose') && ~isempty(opts.verbose)
        enabled = logical(opts.verbose);
    end
end


function log_progress(enabled, varargin)
    if enabled
        fprintf(varargin{:});
    end
end


function text = format_duration(seconds)
    if ~isfinite(seconds)
        text = 'n/a';
        return;
    end

    seconds = max(0.0, double(seconds));
    if seconds < 60.0
        text = sprintf('%.1f s', seconds);
        return;
    end

    minutes = floor(seconds / 60.0);
    rem_seconds = seconds - 60.0 * minutes;
    if seconds < 3600.0
        text = sprintf('%dm %.1fs', minutes, rem_seconds);
        return;
    end

    hours = floor(minutes / 60.0);
    rem_minutes = minutes - 60.0 * hours;
    text = sprintf('%dh %dm %.1fs', hours, rem_minutes, rem_seconds);
end


function plot_reconstruction_results(result, inputs)
    force_names = {'Fx', 'Fy', 'Fz'};
    valid = result.valid_mask;
    freq = result.freqs_hz;

    figure('Name', 'Force Reconstruction from Flight CSV', 'Color', 'w');

    for force_idx = 1:3
        subplot(2, 3, force_idx);
        if any(valid)
            plot(freq(valid), abs(result.F_hat(valid, force_idx)), 'LineWidth', 1.25);
        else
            plot(freq, abs(result.F_hat(:, force_idx)), 'LineWidth', 1.25);
        end
        xlabel('Frequency (Hz)');
        ylabel('|F| (N)');
        title(sprintf('%s magnitude', force_names{force_idx}));
        grid on;
    end

    subplot(2, 3, 4);
    plot(freq(valid), result.response_mac(valid), 'b-', 'LineWidth', 1.25);
    xlabel('Frequency (Hz)');
    ylabel('MAC');
    title('Response MAC');
    grid on;
    ylim([0, 1.05]);

    subplot(2, 3, 5);
    plot(freq(valid), result.relative_residual(valid), 'm-', 'LineWidth', 1.25);
    xlabel('Frequency (Hz)');
    ylabel('Relative residual');
    title('Prediction residual');
    grid on;

    subplot(2, 3, 6);
    cond_values = result.cond_number(valid);
    cond_values = max(cond_values, 1.0);
    semilogy(freq(valid), cond_values, 'k-', 'LineWidth', 1.25);
    xlabel('Frequency (Hz)');
    ylabel('Condition number');
    title('Matrix conditioning');
    grid on;

    plot_single_channel_comparison(result, inputs);
    if inputs.options.plot_all_channels
        plot_all_channel_comparison(result, inputs);
    end
    if inputs.options.plot_lambda_sweep && isfield(result, 'lambda_sweep') && ...
            isfield(result.lambda_sweep, 'n_eval_bins') && result.lambda_sweep.n_eval_bins > 0
        plot_lambda_sweep_figure(result.lambda_sweep);
    end
end


function channel_name = resolve_plot_channel_name(channel_names, channel_idx)
    if iscell(channel_names) && numel(channel_names) >= channel_idx && ~isempty(channel_names{channel_idx})
        channel_name = channel_names{channel_idx};
    else
        channel_name = sprintf('ch%d', channel_idx - 1);
    end
end


function [time_plot, data_plot] = downsample_series_for_plot(time_s, data, max_points)
    n = numel(time_s);
    time_s = time_s(:);
    if isvector(data)
        data = data(:);
    elseif size(data, 1) ~= n && size(data, 2) == n
        data = data.';
    end

    if n <= max_points
        time_plot = time_s;
        data_plot = data;
        return;
    end

    idx = unique(round(linspace(1, n, max_points)));
    time_plot = time_s(idx);
    if isvector(data)
        data_plot = data(idx);
    else
        data_plot = data(idx, :);
    end
end


function plot_single_channel_comparison(result, inputs)
    plot_channel_idx = inputs.options.plot_channel_idx;
    channel_name = resolve_plot_channel_name(inputs.channel_names, plot_channel_idx);
    acc_meas_g = result.acc_meas_sel_g(:, plot_channel_idx);
    acc_pred_g = result.acc_pred_sel_g(:, plot_channel_idx);
    acc_meas_si = result.acc_meas_sel_si(:, plot_channel_idx);
    acc_pred_si = result.acc_pred_sel_si(:, plot_channel_idx);

    [time_plot, acc_meas_plot_g] = downsample_series_for_plot(result.time_sel_s, acc_meas_g, 5000);
    [~, acc_pred_plot_g] = downsample_series_for_plot(result.time_sel_s, acc_pred_g, 5000);

    nperseg = default_psd_nperseg(numel(result.time_sel_s));
    [freq_meas_psd, psd_meas] = welch_psd_1d(acc_meas_si, result.fs_hz, nperseg, [], inputs.options.fft_window);
    [freq_pred_psd, psd_pred] = welch_psd_1d(acc_pred_si, result.fs_hz, nperseg, [], inputs.options.fft_window);

    figure('Name', 'Measured vs Predicted Acceleration', 'Color', 'w');

    subplot(2, 1, 1);
    plot(time_plot, acc_meas_plot_g, 'b-', 'LineWidth', 1.0);
    hold on;
    plot(time_plot, acc_pred_plot_g, 'r--', 'LineWidth', 1.0);
    hold off;
    xlabel('Time (s)');
    ylabel('Acceleration (g)');
    title(sprintf('Measured vs Predicted Acceleration — %s', channel_name));
    legend('Measured', 'Predicted', 'Location', 'best');
    grid on;

    subplot(2, 1, 2);
    loglog(freq_meas_psd, max(psd_meas, 1e-30), 'b-', 'LineWidth', 1.0);
    hold on;
    loglog(freq_pred_psd, max(psd_pred, 1e-30), 'r--', 'LineWidth', 1.0);
    hold off;
    xlabel('Frequency (Hz)');
    ylabel('(m/s^2)^2/Hz');
    title(sprintf('Acceleration PSD — %s', channel_name));
    legend('Measured', 'Predicted', 'Location', 'best');
    grid on;
end


function plot_all_channel_comparison(result, inputs)
    n_channels = size(result.acc_meas_sel_g, 2);
    if n_channels == 0
        return;
    end

    [time_plot, acc_meas_plot_g] = downsample_series_for_plot(result.time_sel_s, result.acc_meas_sel_g, 3000);
    [~, acc_pred_plot_g] = downsample_series_for_plot(result.time_sel_s, result.acc_pred_sel_g, 3000);
    nperseg = default_psd_nperseg(numel(result.time_sel_s));
    colors = lines(max(n_channels, 1));
    legend_handles = gobjects(n_channels, 1);

    figure('Name', 'All Channel Acceleration Comparison', 'Color', 'w');

    subplot(2, 1, 1);
    hold on;
    for channel_idx = 1:n_channels
        legend_handles(channel_idx) = plot(time_plot, acc_meas_plot_g(:, channel_idx), '-', ...
            'Color', colors(channel_idx, :), 'LineWidth', 0.9);
        plot(time_plot, acc_pred_plot_g(:, channel_idx), '--', ...
            'Color', colors(channel_idx, :), 'LineWidth', 0.9);
    end
    hold off;
    xlabel('Time (s)');
    ylabel('Acceleration (g)');
    title('All channels: solid = measured, dashed = predicted');
    grid on;
    maybe_add_channel_legend(legend_handles, inputs.channel_names);

    subplot(2, 1, 2);
    hold on;
    for channel_idx = 1:n_channels
        [freq_meas_psd, psd_meas] = welch_psd_1d(result.acc_meas_sel_si(:, channel_idx), result.fs_hz, nperseg, [], inputs.options.fft_window);
        [freq_pred_psd, psd_pred] = welch_psd_1d(result.acc_pred_sel_si(:, channel_idx), result.fs_hz, nperseg, [], inputs.options.fft_window);
        loglog(freq_meas_psd, max(psd_meas, 1e-30), '-', 'Color', colors(channel_idx, :), 'LineWidth', 0.9);
        loglog(freq_pred_psd, max(psd_pred, 1e-30), '--', 'Color', colors(channel_idx, :), 'LineWidth', 0.9);
    end
    hold off;
    xlabel('Frequency (Hz)');
    ylabel('(m/s^2)^2/Hz');
    title('All channel PSDs: solid = measured, dashed = predicted');
    grid on;
end


function plot_lambda_sweep_figure(lambda_sweep)
    lambda_values = lambda_sweep.lambda_values(:);
    lambda_plot = lambda_values;
    positive_mask = lambda_plot > 0;
    if any(positive_mask)
        min_positive = min(lambda_plot(positive_mask));
    else
        min_positive = 1e-12;
    end
    lambda_plot(~positive_mask) = min_positive / 3.0;
    tick_labels = arrayfun(@format_lambda_value, lambda_values, 'UniformOutput', false);

    figure('Name', 'Lambda Sweep Diagnostics', 'Color', 'w');

    subplot(3, 1, 1);
    semilogx(lambda_plot, lambda_sweep.median_response_mac, 'o-', 'LineWidth', 1.2, 'MarkerSize', 6);
    ylabel('Median MAC');
    title(sprintf('Lambda sweep using %d evaluation bins', lambda_sweep.n_eval_bins));
    grid on;
    set(gca, 'XTick', lambda_plot, 'XTickLabel', tick_labels);

    subplot(3, 1, 2);
    semilogy(lambda_plot, max(lambda_sweep.median_relative_residual, 1e-16), 'o-', 'LineWidth', 1.2, 'MarkerSize', 6);
    ylabel('Median residual');
    grid on;
    set(gca, 'XTick', lambda_plot, 'XTickLabel', tick_labels);

    subplot(3, 1, 3);
    semilogy(lambda_plot, max(lambda_sweep.median_force_norm, 1e-16), 'o-', 'LineWidth', 1.2, 'MarkerSize', 6);
    xlabel('\lambda');
    ylabel('Median ||F||');
    grid on;
    set(gca, 'XTick', lambda_plot, 'XTickLabel', tick_labels);
end


function maybe_add_channel_legend(legend_handles, channel_names)
    if numel(legend_handles) > 10
        return;
    end

    labels = cell(numel(legend_handles), 1);
    for idx = 1:numel(legend_handles)
        labels{idx} = resolve_plot_channel_name(channel_names, idx);
    end
    legend(legend_handles, labels, 'Location', 'eastoutside');
end


function nperseg = default_psd_nperseg(n_time)
    nperseg = min(256, max(32, floor(n_time / 4)));
end


function text = format_lambda_value(value)
    if value == 0
        text = '0';
    else
        text = sprintf('%.0e', value);
    end
end


function [freqs_hz, pxx] = welch_psd_1d(x, fs_hz, nperseg, noverlap, window_name)
    x = double(x(:));
    n = numel(x);
    if n == 0
        error('Need at least one sample for PSD.');
    end

    if nargin < 3 || isempty(nperseg)
        nperseg = min(256, max(32, floor(n / 4)));
    end
    nperseg = min(max(1, round(nperseg)), n);

    if nargin < 4 || isempty(noverlap)
        noverlap = floor(nperseg / 2);
    end
    noverlap = round(noverlap);
    if noverlap < 0 || noverlap >= nperseg
        error('noverlap must satisfy 0 <= noverlap < nperseg.');
    end

    if nargin < 5 || isempty(window_name)
        window_name = 'hann';
    end

    step = nperseg - noverlap;
    starts = 1:step:(n - nperseg + 1);
    if isempty(starts)
        starts = 1;
    end

    win = get_window_vector(window_name, nperseg);
    win_energy = sum(win .^ 2);
    if win_energy <= 0
        error('Window energy must be positive.');
    end

    n_freq = floor(nperseg / 2) + 1;
    freqs_hz = (0:n_freq-1).' * (fs_hz / nperseg);
    pxx = zeros(n_freq, 1);

    for start_idx = starts
        segment = x(start_idx:start_idx + nperseg - 1);
        segment = segment - mean(segment);
        X = fft(segment .* win, nperseg);
        P = abs(X(1:n_freq)) .^ 2 / (fs_hz * win_energy);
        if mod(nperseg, 2) == 0
            if n_freq > 2
                P(2:end-1) = 2.0 * P(2:end-1);
            end
        else
            if n_freq > 1
                P(2:end) = 2.0 * P(2:end);
            end
        end
        pxx = pxx + P;
    end

    pxx = pxx / numel(starts);
end


function x = irfft_matrix(one_sided_spectrum, n_time)
    if isvector(one_sided_spectrum)
        one_sided_spectrum = one_sided_spectrum(:);
    end

    if mod(n_time, 2) == 0
        mirrored = conj(one_sided_spectrum(end-1:-1:2, :));
    else
        mirrored = conj(one_sided_spectrum(end:-1:2, :));
    end

    full_spectrum = [one_sided_spectrum; mirrored];
    x = real(ifft(full_spectrum, n_time, 1));
end


function write_force_spectrum_csv(freqs_hz, F_hat, valid_mask, path_out)
    fid = fopen(path_out, 'w');
    if fid < 0
        error('Could not open %s for writing.', path_out);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, 'freq_hz,Fx_re,Fx_im,Fy_re,Fy_im,Fz_re,Fz_im,valid\n');
    for k = 1:numel(freqs_hz)
        fprintf(fid, '%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%d\n', ...
            freqs_hz(k), ...
            real(F_hat(k, 1)), imag(F_hat(k, 1)), ...
            real(F_hat(k, 2)), imag(F_hat(k, 2)), ...
            real(F_hat(k, 3)), imag(F_hat(k, 3)), ...
            logical_to_int(valid_mask(k)));
    end
end


function write_reconstruction_diagnostics_csv(result, path_out)
    fid = fopen(path_out, 'w');
    if fid < 0
        error('Could not open %s for writing.', path_out);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, ['freq_hz,Fx_re,Fx_im,Fy_re,Fy_im,Fz_re,Fz_im,cond_number,' ...
        'sigma1,sigma2,sigma3,mac_xy,mac_xz,mac_yz,response_mac,relative_residual,valid\n']);
    for k = 1:numel(result.freqs_hz)
        fprintf(fid, ['%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,' ...
            '%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%d\n'], ...
            result.freqs_hz(k), ...
            real(result.F_hat(k, 1)), imag(result.F_hat(k, 1)), ...
            real(result.F_hat(k, 2)), imag(result.F_hat(k, 2)), ...
            real(result.F_hat(k, 3)), imag(result.F_hat(k, 3)), ...
            result.cond_number(k), ...
            result.singular_values(k, 1), result.singular_values(k, 2), result.singular_values(k, 3), ...
            result.mac_xy(k), result.mac_xz(k), result.mac_yz(k), ...
            result.response_mac(k), result.relative_residual(k), ...
            logical_to_int(result.valid_mask(k)));
    end
end


function out = logical_to_int(value)
    if value
        out = 1;
    else
        out = 0;
    end
end
