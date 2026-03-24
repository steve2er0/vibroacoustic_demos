function [result, inputs] = reconstruct_forces_from_flight_csv(varargin)
%RECONSTRUCT_FORCES_FROM_FLIGHT_CSV Reconstruct interface forces from flight input data.
%
%   Legacy form:
%   [result, inputs] = reconstruct_forces_from_flight_csv(pathFx, pathFy, pathFz, flightInput)
%   [result, inputs] = reconstruct_forces_from_flight_csv(pathFx, pathFy, pathFz, flightInput, opts)
%
%   General form:
%   [result, inputs] = reconstruct_forces_from_flight_csv(mobilityPaths, flightInput)
%   [result, inputs] = reconstruct_forces_from_flight_csv(mobilityPaths, flightInput, opts)
%
% Inputs
%   pathFx, pathFy, pathFz
%       Legacy 3-file mobility CSV inputs for unit Fx, Fy, Fz.
%   mobilityPaths
%       Ordered cell array / string array of one or more mobility CSVs.
%       Each CSV uses columns:
%       freq_hz,re0,im0,re1,im1,...
%       The order of the files defines the reconstructed load-case order.
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
%       active_channel_idx    optional active-channel list or logical mask
%                             in the original flight/FRF channel order,
%                             default = all channels
%       load_case_names       optional names for plots / CSV export, default = file basenames
%       solve_on_frf_grid     solve only at the FRF frequency lines, default = false
%       fft_window            'hann' or 'boxcar', default = 'hann'
%       plot_results          plot reconstructed forces + diagnostics, default = false
%       plot_channel_idx      1-based channel index for acceleration comparison plots, default = 1
%       plot_all_channels     also plot all-channel measured/predicted overlays, default = true
%       plot_psd_error_map    plot all-channel PSD dB-difference map, default = true
%       plot_psd_xscale       'log' or 'linear', default = 'log'
%       plot_psd_yscale       'log' or 'linear', default = 'log'
%       plot_psd_fmin_hz      PSD plot lower x limit [Hz], default = 10
%       plot_psd_fmax_hz      PSD plot upper x limit [Hz], default = 3000
%       psd_nperseg           optional Welch segment length for PSD plots
%       psd_noverlap          optional Welch overlap for PSD plots
%       plot_lambda_sweep     plot lambda sweep summary, default = false
%       lambda_sweep_values   lambda values for the sweep, default = [0 1e-9 1e-8 1e-7 1e-6 1e-5]
%       lambda_sweep_max_bins maximum bins used in the lambda sweep summary, default = 2000
%       verbose               print summary metrics, default = true
%       show_progress         print stage progress and timings, default = verbose
%       progress_interval_sec progress update period during solve loop, default = 2.0
%       save_force_csv        optional output path for F_hat CSV
%       save_diagnostics_csv  optional output path for diagnostics CSV
%       save_nastran_tabled1  optional output path for NASTRAN TABLED1 include
%       save_nastran_replay_bdf optional output path for a SOL111 replay-deck skeleton
%       nastran_force_unit    force units for NASTRAN export: 'N' or 'lbf',
%                             default = 'N'
%       nastran_table_id_start starting TABLED1 ID for NASTRAN export,
%                              default = 1001
%       nastran_grid_ids      scalar grid ID for the legacy 3-load case, or a
%                             vector of grid IDs with one entry per load case
%       nastran_components    optional component list for replay export.
%                             Default = [1 2 3] for the legacy 3-load case
%       nastran_darea_scales  optional DAREA scales for replay export,
%                             default = 1.0 for each load case
%       nastran_darea_sid_start starting DAREA SID for replay export,
%                               default = 101
%       nastran_rload_sid_start starting RLOAD1 SID for replay export,
%                               default = 201
%       nastran_dload_sid     DLOAD SID for replay export, default = 40
%       nastran_subcase_id    replay subcase ID, default = 1
%       nastran_spc_sid       SPC SID referenced in the replay deck, default = 1
%       nastran_method_sid    METHOD SID referenced in the replay deck, default = 10
%       nastran_freq_sid      FREQ SID referenced in the replay deck, default = 30
%       nastran_sdamping_sid  optional SDAMPING SID for the replay deck, default = []
%       nastran_title         replay deck title, default = 'Reconstructed Force Replay'
%       nastran_model_include optional model INCLUDE path in the replay deck
%
% Outputs
%   result
%       Struct aligned with the legacy Python-style fields plus MATLAB-only
%       load-case metadata:
%       freqs_hz, F_hat, a_meas_fft, a_pred_fft, cond_number,
%       singular_values, column_mac, max_column_mac, mac_xy, mac_xz,
%       mac_yz, response_mac, load_case_names, active_channel_idx,
%       relative_residual, valid_mask, fs_hz, t_start, t_end,
%       selected measured/predicted accelerations, preprocessing info,
%       optional NASTRAN-export force spectrum in opts.nastran_force_unit,
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
%   mobilityPaths = {'bolt01_Fz.csv', 'bolt02_Fz.csv', 'bolt03_Fz.csv', 'bolt04_Fz.csv'};
%   [result, inputs] = reconstruct_forces_from_flight_csv( ...
%       mobilityPaths, 'flight.csv', opts);
%
%   matFiles = {'ch01.mat', 'ch02.mat', 'ch03.mat'};
%   [result, inputs] = reconstruct_forces_from_flight_csv( ...
%       'mobility_Fx.csv', 'mobility_Fy.csv', 'mobility_Fz.csv', matFiles, opts);

    [mobility_paths, flightInput, opts, is_legacy_triplet] = parse_reconstruction_inputs(varargin{:});
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

    assert_path_list_exists(mobility_paths, 'mobilityPaths');
    assert_flight_input_exists(flightInput, 'flightInput');

    stage_timer = tic;
    log_progress(progress_enabled, 'Loading %d mobility CSV file(s)...\n', numel(mobility_paths));
    [frf_freqs_hz, H_input] = load_mobility_csv_stack(mobility_paths);
    load_case_names = resolve_load_case_names(opts, mobility_paths, size(H_input, 3), is_legacy_triplet);
    timing.load_mobility_sec = toc(stage_timer);
    log_progress(progress_enabled, 'Loaded mobility CSVs: %d frequencies, %d sensors, %d load cases in %s.\n', ...
        numel(frf_freqs_hz), size(H_input, 2), size(H_input, 3), format_duration(timing.load_mobility_sec));

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
    total_channel_count = size(acc_g, 2);
    active_channel_idx = resolve_active_channel_indices(opts.active_channel_idx, total_channel_count);
    active_channel_names = channel_names(active_channel_idx);
    if numel(active_channel_idx) < total_channel_count
        log_progress(progress_enabled, 'Using %d/%d active channels for reconstruction.\n', ...
            numel(active_channel_idx), total_channel_count);
    else
        log_progress(progress_enabled, 'Using all %d channels for reconstruction.\n', total_channel_count);
    end
    H_input = H_input(:, active_channel_idx, :);
    acc_g = acc_g(:, active_channel_idx);
    channel_names = active_channel_names;
    opts.plot_channel_idx = resolve_plot_channel_option(opts.plot_channel_idx, active_channel_idx);

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
    [flight_fft_freqs_hz, a_meas_fft_flight] = complex_rfft_matrix(acc_sel_si, fs_hz, opts.fft_window);
    timing.fft_sec = toc(stage_timer);
    log_progress(progress_enabled, 'Computed FFT: %d frequency bins at %.3f Hz sample rate in %s.\n', ...
        numel(flight_fft_freqs_hz), fs_hz, format_duration(timing.fft_sec));

    stage_timer = tic;
    log_progress(progress_enabled, 'Converting mobility to accelerance and preparing the solve grid...\n');
    H_si = H_input;
    if ~opts.mobility_is_si
        H_si = mobility_imp_to_si(H_input);
    end

    A_frf = accelerance_from_mobility(H_si, 2.0 * pi * frf_freqs_hz);
    [freqs_hz, a_meas_fft, A_fft, solve_grid_source] = prepare_reconstruction_grid( ...
        flight_fft_freqs_hz, a_meas_fft_flight, frf_freqs_hz, A_frf, opts);
    timing.interp_sec = toc(stage_timer);
    if strcmp(solve_grid_source, 'frf')
        log_progress(progress_enabled, 'Prepared FRFs on the FRF grid: %d solve bins in %s.\n', ...
            numel(freqs_hz), format_duration(timing.interp_sec));
    else
        log_progress(progress_enabled, 'Prepared FRFs on the flight FFT grid: %d solve bins in %s.\n', ...
            numel(freqs_hz), format_duration(timing.interp_sec));
    end

    n_freq = numel(freqs_hz);
    n_sensors = size(a_meas_fft, 2);
    n_loads = size(A_fft, 3);
    n_singular = min(n_sensors, n_loads);

    F_hat = complex(zeros(n_freq, n_loads));
    a_pred_fft = complex(zeros(n_freq, n_sensors));
    cond_number = NaN(n_freq, 1);
    singular_values = NaN(n_freq, n_singular);
    column_mac = NaN(n_freq, n_loads, n_loads);
    max_column_mac = NaN(n_freq, 1);
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
    finite_mask = all(isfinite(reshape(A_fft, n_freq, [])), 2) & all(isfinite(a_meas_fft), 2);
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
        Ak = reshape(A_fft(k, :, :), n_sensors, n_loads);

        ak = a_meas_fft(k, :).';
        column_mac_k = column_mac_matrix(Ak);
        column_mac(k, :, :) = column_mac_k;
        max_column_mac(k) = max_offdiag_column_mac(column_mac_k);
        [mac_xy(k), mac_xz(k), mac_yz(k)] = legacy_column_mac_pairs(column_mac_k);

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

    if strcmp(solve_grid_source, 'frf')
        a_pred_fft_timegrid = interp_complex_frequency(flight_fft_freqs_hz, freqs_hz, a_pred_fft, complex(0.0, 0.0));
    else
        a_pred_fft_timegrid = a_pred_fft;
    end

    acc_pred_sel_si = real(irfft_matrix(a_pred_fft_timegrid, numel(time_sel)));
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
    result.flight_fft_freqs_hz = flight_fft_freqs_hz;
    result.a_meas_fft_flight = a_meas_fft_flight;
    result.a_pred_fft_flight = a_pred_fft_timegrid;
    result.solve_grid_source = solve_grid_source;
    result.cond_number = cond_number;
    result.singular_values = singular_values;
    result.column_mac = column_mac;
    result.max_column_mac = max_column_mac;
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
    result.channel_names = channel_names;
    result.active_channel_idx = active_channel_idx(:);
    result.n_channels_total = total_channel_count;
    result.load_case_names = load_case_names;
    result.preprocessing = preprocessing;
    result.lambda_sweep = lambda_sweep;
    result.F_hat_nastran = convert_force_from_si(result.F_hat, opts.nastran_force_unit);
    result.nastran_force_unit = upper(opts.nastran_force_unit);
    timing.reconstruction_sec = toc(overall_timer);
    result.timing = timing;

    inputs = struct();
    inputs.mobility_paths = mobility_paths;
    inputs.load_case_names = load_case_names;
    if is_legacy_triplet
        inputs.path_fx = mobility_paths{1};
        inputs.path_fy = mobility_paths{2};
        inputs.path_fz = mobility_paths{3};
    end
    inputs.path_flight = flightInput;
    inputs.flight_input = flightInput;
    inputs.flight_input_type = flight_input_type;
    inputs.frf_freqs_hz = frf_freqs_hz;
    inputs.solve_grid_source = solve_grid_source;
    inputs.H_input = H_input;
    inputs.acc_g_raw = acc_g_raw;
    inputs.time_s = time_s;
    inputs.acc_g = acc_g;
    inputs.channel_names = channel_names;
    inputs.active_channel_idx = active_channel_idx(:);
    inputs.n_channels_total = total_channel_count;
    inputs.options = opts;
    inputs.preprocessing = preprocessing;
    inputs.timing = timing;

    if ~isempty(opts.save_force_csv)
        stage_timer = tic;
        log_progress(progress_enabled, 'Writing force spectrum CSV: %s\n', opts.save_force_csv);
        write_force_spectrum_csv(result.freqs_hz, result.F_hat, result.valid_mask, ...
            result.load_case_names, opts.save_force_csv);
        log_progress(progress_enabled, 'Wrote force spectrum CSV in %s.\n', format_duration(toc(stage_timer)));
    end
    if ~isempty(opts.save_diagnostics_csv)
        stage_timer = tic;
        log_progress(progress_enabled, 'Writing diagnostics CSV: %s\n', opts.save_diagnostics_csv);
        write_reconstruction_diagnostics_csv(result, opts.save_diagnostics_csv);
        log_progress(progress_enabled, 'Wrote diagnostics CSV in %s.\n', format_duration(toc(stage_timer)));
    end
    if ~isempty(opts.save_nastran_tabled1)
        stage_timer = tic;
        log_progress(progress_enabled, 'Writing NASTRAN TABLED1 include: %s\n', opts.save_nastran_tabled1);
        write_nastran_force_tabled1( ...
            result.freqs_hz, result.F_hat, result.valid_mask, result.load_case_names, ...
            opts.save_nastran_tabled1, opts.nastran_force_unit, opts.nastran_table_id_start);
        log_progress(progress_enabled, 'Wrote NASTRAN TABLED1 include in %s.\n', ...
            format_duration(toc(stage_timer)));
    end
    if ~isempty(opts.save_nastran_replay_bdf)
        stage_timer = tic;
        log_progress(progress_enabled, 'Writing NASTRAN replay deck: %s\n', opts.save_nastran_replay_bdf);
        nastran_tabled1_path = resolve_nastran_tabled1_output_path(opts.save_nastran_tabled1, opts.save_nastran_replay_bdf);
        if isempty(opts.save_nastran_tabled1)
            write_nastran_force_tabled1( ...
                result.freqs_hz, result.F_hat, result.valid_mask, result.load_case_names, ...
                nastran_tabled1_path, opts.nastran_force_unit, opts.nastran_table_id_start);
            log_progress(progress_enabled, 'Auto-wrote companion NASTRAN TABLED1 include: %s\n', nastran_tabled1_path);
        end
        [nastran_grid_ids, nastran_components, nastran_darea_scales] = resolve_nastran_replay_dofs(opts, result);
        write_nastran_replay_bdf( ...
            opts.save_nastran_replay_bdf, nastran_tabled1_path, result.load_case_names, ...
            nastran_grid_ids, nastran_components, nastran_darea_scales, opts);
        log_progress(progress_enabled, 'Wrote NASTRAN replay deck in %s.\n', ...
            format_duration(toc(stage_timer)));
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
    defaults.active_channel_idx = [];
    defaults.load_case_names = {};
    defaults.solve_on_frf_grid = false;
    defaults.fft_window = 'hann';
    defaults.plot_results = false;
    defaults.plot_channel_idx = 1;
    defaults.plot_all_channels = true;
    defaults.plot_psd_error_map = true;
    defaults.plot_psd_xscale = 'log';
    defaults.plot_psd_yscale = 'log';
    defaults.plot_psd_fmin_hz = 10.0;
    defaults.plot_psd_fmax_hz = 3000.0;
    defaults.psd_nperseg = [];
    defaults.psd_noverlap = [];
    defaults.plot_lambda_sweep = false;
    defaults.lambda_sweep_values = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5];
    defaults.lambda_sweep_max_bins = 2000;
    defaults.verbose = true;
    defaults.show_progress = [];
    defaults.progress_interval_sec = 2.0;
    defaults.save_force_csv = '';
    defaults.save_diagnostics_csv = '';
    defaults.save_nastran_tabled1 = '';
    defaults.save_nastran_replay_bdf = '';
    defaults.nastran_force_unit = 'N';
    defaults.nastran_table_id_start = 1001;
    defaults.nastran_grid_ids = [];
    defaults.nastran_components = [];
    defaults.nastran_darea_scales = [];
    defaults.nastran_darea_sid_start = 101;
    defaults.nastran_rload_sid_start = 201;
    defaults.nastran_dload_sid = 40;
    defaults.nastran_subcase_id = 1;
    defaults.nastran_spc_sid = 1;
    defaults.nastran_method_sid = 10;
    defaults.nastran_freq_sid = 30;
    defaults.nastran_sdamping_sid = [];
    defaults.nastran_title = 'Reconstructed Force Replay';
    defaults.nastran_model_include = '';

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
    opts.plot_psd_error_map = logical(opts.plot_psd_error_map);
    opts.plot_lambda_sweep = logical(opts.plot_lambda_sweep);
    opts.solve_on_frf_grid = logical(opts.solve_on_frf_grid);
    opts.plot_psd_xscale = validate_axis_scale_option(opts.plot_psd_xscale, 'opts.plot_psd_xscale');
    opts.plot_psd_yscale = validate_axis_scale_option(opts.plot_psd_yscale, 'opts.plot_psd_yscale');
    opts.load_case_names = normalize_optional_name_list(opts.load_case_names, 'opts.load_case_names');
    opts.nastran_force_unit = validate_force_unit_option(opts.nastran_force_unit, 'opts.nastran_force_unit');

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
    if ~isscalar(opts.plot_psd_fmin_hz) || ~isnumeric(opts.plot_psd_fmin_hz) || ...
            ~isfinite(opts.plot_psd_fmin_hz) || opts.plot_psd_fmin_hz < 0
        error('opts.plot_psd_fmin_hz must be a non-negative finite scalar.');
    end
    if ~isscalar(opts.plot_psd_fmax_hz) || ~isnumeric(opts.plot_psd_fmax_hz) || ...
            ~isfinite(opts.plot_psd_fmax_hz) || opts.plot_psd_fmax_hz <= 0
        error('opts.plot_psd_fmax_hz must be a positive finite scalar.');
    end
    if opts.plot_psd_fmax_hz < opts.plot_psd_fmin_hz
        error('opts.plot_psd_fmax_hz must be >= opts.plot_psd_fmin_hz.');
    end
    if strcmp(opts.plot_psd_xscale, 'log') && opts.plot_psd_fmin_hz <= 0
        error('opts.plot_psd_fmin_hz must be > 0 when opts.plot_psd_xscale is "log".');
    end
    if ~isempty(opts.psd_nperseg)
        if ~isscalar(opts.psd_nperseg) || ~isnumeric(opts.psd_nperseg) || ...
                ~isfinite(opts.psd_nperseg) || opts.psd_nperseg < 1 || ...
                opts.psd_nperseg ~= round(opts.psd_nperseg)
            error('opts.psd_nperseg must be empty or a positive integer.');
        end
    end
    if ~isempty(opts.psd_noverlap)
        if ~isscalar(opts.psd_noverlap) || ~isnumeric(opts.psd_noverlap) || ...
                ~isfinite(opts.psd_noverlap) || opts.psd_noverlap < 0 || ...
                opts.psd_noverlap ~= round(opts.psd_noverlap)
            error('opts.psd_noverlap must be empty or a non-negative integer.');
        end
        if ~isempty(opts.psd_nperseg) && opts.psd_noverlap >= opts.psd_nperseg
            error('opts.psd_noverlap must be less than opts.psd_nperseg.');
        end
    end
    if ~isscalar(opts.progress_interval_sec) || ~isnumeric(opts.progress_interval_sec) || ...
            ~isfinite(opts.progress_interval_sec) || opts.progress_interval_sec <= 0
        error('opts.progress_interval_sec must be a positive finite scalar.');
    end
    if ~isscalar(opts.nastran_table_id_start) || ~isnumeric(opts.nastran_table_id_start) || ...
            ~isfinite(opts.nastran_table_id_start) || opts.nastran_table_id_start < 1 || ...
            opts.nastran_table_id_start ~= round(opts.nastran_table_id_start)
        error('opts.nastran_table_id_start must be a positive integer.');
    end
    validate_positive_integer_option(opts.nastran_darea_sid_start, 'opts.nastran_darea_sid_start');
    validate_positive_integer_option(opts.nastran_rload_sid_start, 'opts.nastran_rload_sid_start');
    validate_positive_integer_option(opts.nastran_dload_sid, 'opts.nastran_dload_sid');
    validate_positive_integer_option(opts.nastran_subcase_id, 'opts.nastran_subcase_id');
    validate_positive_integer_option(opts.nastran_spc_sid, 'opts.nastran_spc_sid');
    validate_positive_integer_option(opts.nastran_method_sid, 'opts.nastran_method_sid');
    validate_positive_integer_option(opts.nastran_freq_sid, 'opts.nastran_freq_sid');
    if ~isempty(opts.nastran_sdamping_sid)
        validate_positive_integer_option(opts.nastran_sdamping_sid, 'opts.nastran_sdamping_sid');
    end
    if ~(ischar(opts.nastran_title) || (isstring(opts.nastran_title) && isscalar(opts.nastran_title)))
        error('opts.nastran_title must be a character vector or scalar string.');
    end
    opts.nastran_title = char(opts.nastran_title);
    if ~(isempty(opts.nastran_model_include) || ischar(opts.nastran_model_include) || ...
            (isstring(opts.nastran_model_include) && isscalar(opts.nastran_model_include)))
        error('opts.nastran_model_include must be empty, a character vector, or a scalar string.');
    end
    if isstring(opts.nastran_model_include)
        opts.nastran_model_include = char(opts.nastran_model_include);
    end
end


function active_idx = resolve_active_channel_indices(value, n_channels)
    if isempty(value)
        active_idx = 1:n_channels;
        return;
    end

    if islogical(value)
        mask = value(:);
        if numel(mask) ~= n_channels
            error('opts.active_channel_idx logical mask must have length %d.', n_channels);
        end
        active_idx = find(mask).';
    else
        if ~isnumeric(value)
            error('opts.active_channel_idx must be empty, a logical mask, or a numeric index vector.');
        end
        active_idx = double(value(:).');
        if isempty(active_idx) || any(~isfinite(active_idx)) || any(active_idx < 1) || ...
                any(active_idx > n_channels) || any(active_idx ~= round(active_idx))
            error('opts.active_channel_idx must contain integer channel indices from 1 to %d.', n_channels);
        end
        active_idx = unique(active_idx, 'stable');
    end

    if isempty(active_idx)
        error('opts.active_channel_idx must select at least one channel.');
    end
end


function plot_channel_idx = resolve_plot_channel_option(value, active_channel_idx)
    n_active = numel(active_channel_idx);
    if ~isscalar(value) || ~isnumeric(value) || ~isfinite(value) || value < 1 || value ~= round(value)
        error('opts.plot_channel_idx must be a positive integer.');
    end

    if value > n_active
        mapped = find(active_channel_idx == value, 1, 'first');
        if isempty(mapped)
            error('opts.plot_channel_idx must reference an active channel position from 1 to %d, or an original active channel index.', n_active);
        end
        plot_channel_idx = mapped;
        return;
    end

    plot_channel_idx = value;
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


function [solve_freqs_hz, a_meas_solve_fft, A_solve, solve_grid_source] = prepare_reconstruction_grid( ...
        flight_fft_freqs_hz, a_meas_fft_flight, frf_freqs_hz, A_frf, opts)
    if opts.solve_on_frf_grid
        fft_max_hz = flight_fft_freqs_hz(end);
        tol_hz = max(1e-12, 1e-9 * max(abs([fft_max_hz; frf_freqs_hz(:); 1.0])));
        in_range_mask = frf_freqs_hz >= (flight_fft_freqs_hz(1) - tol_hz) & ...
            frf_freqs_hz <= (fft_max_hz + tol_hz);
        solve_freqs_hz = frf_freqs_hz(in_range_mask);
        if isempty(solve_freqs_hz)
            error('No FRF frequency lines overlap the flight FFT range.');
        end

        A_solve = A_frf(in_range_mask, :, :);
        a_meas_solve_fft = interp_complex_frequency(solve_freqs_hz, flight_fft_freqs_hz, ...
            a_meas_fft_flight, complex(NaN, NaN));
        solve_grid_source = 'frf';
        return;
    end

    solve_freqs_hz = flight_fft_freqs_hz;
    a_meas_solve_fft = a_meas_fft_flight;
    A_solve = interp_complex_frequency(solve_freqs_hz, frf_freqs_hz, A_frf, complex(NaN, NaN));
    solve_grid_source = 'flight_fft';
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
            Ak = reshape(A_fft(k, :, :), n_sensors, size(A_fft, 3));
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


function [mobility_paths, flightInput, opts, is_legacy_triplet] = parse_reconstruction_inputs(varargin)
    if nargin < 2
        error(['Usage: reconstruct_forces_from_flight_csv(pathFx, pathFy, pathFz, flightInput, opts) ' ...
            'or reconstruct_forces_from_flight_csv(mobilityPaths, flightInput, opts).']);
    end

    is_legacy_triplet = false;
    if (nargin == 2 || nargin == 3) && is_path_list_input(varargin{1})
        mobility_paths = normalize_path_list(varargin{1}, 'mobilityPaths');
        flightInput = varargin{2};
        if nargin >= 3
            opts = varargin{3};
        else
            opts = struct();
        end
        return;
    end

    if nargin == 4 || nargin == 5
        mobility_paths = {
            normalize_scalar_path(varargin{1}, 'pathFx')
            normalize_scalar_path(varargin{2}, 'pathFy')
            normalize_scalar_path(varargin{3}, 'pathFz')
        };
        flightInput = varargin{4};
        if nargin >= 5
            opts = varargin{5};
        else
            opts = struct();
        end
        is_legacy_triplet = true;
        return;
    end

    error(['Usage: reconstruct_forces_from_flight_csv(pathFx, pathFy, pathFz, flightInput, opts) ' ...
        'or reconstruct_forces_from_flight_csv(mobilityPaths, flightInput, opts).']);
end


function tf = is_path_list_input(value)
    tf = iscell(value) || (isstring(value) && ~isscalar(value));
end


function paths = normalize_path_list(value, argName)
    if isstring(value)
        value = cellstr(value(:));
    end
    if ~iscell(value) || isempty(value)
        error('%s must be a non-empty cell array or string array of file paths.', argName);
    end

    paths = cell(numel(value), 1);
    for idx = 1:numel(value)
        paths{idx} = normalize_scalar_path(value{idx}, sprintf('%s{%d}', argName, idx));
    end
end


function pathValue = normalize_scalar_path(value, argName)
    if isstring(value)
        if ~isscalar(value)
            error('%s must be a scalar string or character vector.', argName);
        end
        value = char(value);
    end
    if ~ischar(value) || isempty(strtrim(value))
        error('%s must be a non-empty file path.', argName);
    end
    pathValue = strtrim(value);
end


function assert_file_exists(pathValue, argName)
    if ~isfile(pathValue)
        error('%s does not exist: %s', argName, pathValue);
    end
end


function assert_path_list_exists(paths, argName)
    for idx = 1:numel(paths)
        assert_file_exists(paths{idx}, sprintf('%s{%d}', argName, idx));
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


function [freqs_hz, H] = load_mobility_csv_stack(paths)
    freqs_hz = [];
    stacks = cell(1, numel(paths));

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

    H = cat(3, stacks{:});
end


function [freqs_hz, H] = load_mobility_csv_triplet(pathFx, pathFy, pathFz)
    [freqs_hz, H] = load_mobility_csv_stack({pathFx, pathFy, pathFz});
end


function load_case_names = resolve_load_case_names(opts, mobility_paths, n_loads, is_legacy_triplet)
    provided_names = {};
    if isfield(opts, 'load_case_names')
        provided_names = normalize_optional_name_list(opts.load_case_names, 'opts.load_case_names');
    end

    if ~isempty(provided_names)
        if numel(provided_names) ~= n_loads
            error('opts.load_case_names must have exactly %d entries.', n_loads);
        end
        load_case_names = provided_names(:).';
        return;
    end

    if is_legacy_triplet && n_loads == 3
        load_case_names = {'Fx', 'Fy', 'Fz'};
        return;
    end

    load_case_names = derive_load_case_names_from_paths(mobility_paths);
end


function names = derive_load_case_names_from_paths(paths)
    n = numel(paths);
    names = cell(1, n);
    used = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for idx = 1:n
        [~, stem] = fileparts(paths{idx});
        if isempty(stem)
            stem = sprintf('load%d', idx);
        end
        base_name = stem;
        if isKey(used, base_name)
            used(base_name) = used(base_name) + 1;
            stem = sprintf('%s_%d', base_name, used(base_name));
        else
            used(base_name) = 1;
        end
        names{idx} = stem;
    end
end


function names = normalize_optional_name_list(value, argName)
    if isempty(value)
        names = {};
        return;
    end
    if isstring(value)
        value = cellstr(value(:));
    end
    if ~iscell(value)
        error('%s must be empty, a cell array, or a string array.', argName);
    end

    names = cell(1, numel(value));
    for idx = 1:numel(value)
        name = value{idx};
        if isstring(name)
            if ~isscalar(name)
                error('%s{%d} must be a scalar string or character vector.', argName, idx);
            end
            name = char(name);
        end
        if ~ischar(name) || isempty(strtrim(name))
            error('%s{%d} must be a non-empty string.', argName, idx);
        end
        names{idx} = strtrim(name);
    end
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


function M = column_mac_matrix(A)
    n_columns = size(A, 2);
    M = zeros(n_columns, n_columns);
    for i = 1:n_columns
        ci = A(:, i);
        nii = real(ci' * ci);
        if nii <= 0
            continue;
        end
        for j = 1:n_columns
            cj = A(:, j);
            njj = real(cj' * cj);
            if njj <= 0
                continue;
            end
            cross = ci' * cj;
            M(i, j) = abs(cross) .^ 2 / (nii * njj);
        end
    end
end


function value = max_offdiag_column_mac(M)
    if isempty(M) || size(M, 1) < 2
        value = NaN;
        return;
    end
    mask = ~eye(size(M, 1));
    values = M(mask);
    if isempty(values)
        value = NaN;
    else
        value = max(values);
    end
end


function [mac_xy, mac_xz, mac_yz] = legacy_column_mac_pairs(M)
    mac_xy = NaN;
    mac_xz = NaN;
    mac_yz = NaN;
    if size(M, 1) >= 2
        mac_xy = M(1, 2);
    end
    if size(M, 1) >= 3
        mac_xz = M(1, 3);
        mac_yz = M(2, 3);
    end
end


function F = solve_force_complex(A, a, lam)
    AH = A';
    ATA = AH * A;
    ATa = AH * a;

    if lam > 0
        ATA = ATA + (lam ^ 2) * eye(size(A, 2));
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
    if isfield(result, 'solve_grid_source')
        fprintf('Solve frequency grid         : %s\n', strrep(result.solve_grid_source, '_', ' '));
    end
    if isfield(result, 'active_channel_idx') && isfield(result, 'n_channels_total')
        fprintf('Active channels             : %d / %d\n', ...
            numel(result.active_channel_idx), result.n_channels_total);
    end
    if isfield(result, 'load_case_names')
        fprintf('Load cases                  : %d\n', numel(result.load_case_names));
    end
    if isfield(result, 'preprocessing') && result.preprocessing.highpass_enabled
        fprintf('High-pass filter            : %d-pole %.3f Hz (%s)\n', ...
            result.preprocessing.highpass_order, result.preprocessing.highpass_hz, result.preprocessing.method);
    end
    fprintf('Valid frequency bins         : %d / %d\n', nnz(valid), numel(valid));
    fprintf('Median response MAC          : %.6f\n', median_finite(result.response_mac(valid)));
    fprintf('Median relative residual     : %.6f\n', median_finite(result.relative_residual(valid)));
    fprintf('Median condition number      : %.6f\n', median_finite(result.cond_number(valid)));
    if isfield(result, 'max_column_mac')
        fprintf('Median max column MAC        : %.6f\n', median_finite(result.max_column_mac(valid)));
    end
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
    force_names = result.load_case_names;
    valid = result.valid_mask;
    freq = result.freqs_hz;
    n_loads = numel(force_names);
    n_panels = n_loads + 3;
    n_cols = min(4, max(2, ceil(sqrt(n_panels))));
    n_rows = ceil(n_panels / n_cols);

    figure('Name', 'Force Reconstruction from Flight Data', 'Color', 'w');

    for force_idx = 1:n_loads
        subplot(n_rows, n_cols, force_idx);
        if any(valid)
            plot(freq(valid), abs(result.F_hat(valid, force_idx)), 'LineWidth', 1.25);
        else
            plot(freq, abs(result.F_hat(:, force_idx)), 'LineWidth', 1.25);
        end
        xlabel('Frequency (Hz)');
        ylabel('|F| (N)');
        title(sprintf('%s magnitude', force_names{force_idx}), 'Interpreter', 'none');
        grid on;
    end

    subplot(n_rows, n_cols, n_loads + 1);
    plot(freq(valid), result.response_mac(valid), 'b-', 'LineWidth', 1.25);
    xlabel('Frequency (Hz)');
    ylabel('MAC');
    title('Response MAC');
    grid on;
    ylim([0, 1.05]);

    subplot(n_rows, n_cols, n_loads + 2);
    plot(freq(valid), result.relative_residual(valid), 'm-', 'LineWidth', 1.25);
    xlabel('Frequency (Hz)');
    ylabel('Relative residual');
    title('Prediction residual');
    grid on;

    subplot(n_rows, n_cols, n_loads + 3);
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
    rms_meas_g = compute_rms_value(acc_meas_g);
    rms_pred_g = compute_rms_value(acc_pred_g);

    [time_plot, acc_meas_plot_g] = downsample_series_for_plot(result.time_sel_s, acc_meas_g, 5000);
    [~, acc_pred_plot_g] = downsample_series_for_plot(result.time_sel_s, acc_pred_g, 5000);

    [nperseg, noverlap] = resolve_psd_welch_options(numel(result.time_sel_s), inputs.options);
    [freq_meas_psd, psd_meas] = welch_psd_1d(acc_meas_g, result.fs_hz, nperseg, noverlap, inputs.options.fft_window);
    [freq_pred_psd, psd_pred] = welch_psd_1d(acc_pred_g, result.fs_hz, nperseg, noverlap, inputs.options.fft_window);
    psd_meas_plus_3db = psd_meas * db_power_ratio(3.0);
    psd_meas_minus_3db = psd_meas / db_power_ratio(3.0);
    [freq_plot, psd_meas_plot] = prepare_psd_curve_for_plot(freq_meas_psd, psd_meas, inputs.options);
    [freq_pred_plot, psd_pred_plot] = prepare_psd_curve_for_plot(freq_pred_psd, psd_pred, inputs.options);
    [freq_plus_plot, psd_meas_plus_plot] = prepare_psd_curve_for_plot(freq_meas_psd, psd_meas_plus_3db, inputs.options);
    [freq_minus_plot, psd_meas_minus_plot] = prepare_psd_curve_for_plot(freq_meas_psd, psd_meas_minus_3db, inputs.options);

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
    hold on;
    plot(freq_plot, psd_meas_plot, 'b-', 'LineWidth', 1.0);
    plot(freq_pred_plot, psd_pred_plot, 'r--', 'LineWidth', 1.0);
    plot(freq_plus_plot, psd_meas_plus_plot, 'k:', 'LineWidth', 0.9);
    plot(freq_minus_plot, psd_meas_minus_plot, 'k:', 'LineWidth', 0.9);
    hold off;
    apply_psd_axes(gca, inputs.options);
    xlabel('Frequency (Hz)');
    ylabel('g^2/Hz');
    title(sprintf('Acceleration PSD — %s', channel_name));
    legend(build_psd_legend_labels(rms_meas_g, rms_pred_g), 'Location', 'best');
    grid on;
end


function plot_all_channel_comparison(result, inputs)
    n_channels = size(result.acc_meas_sel_g, 2);
    if n_channels == 0
        return;
    end

    [time_plot, acc_meas_plot_g] = downsample_series_for_plot(result.time_sel_s, result.acc_meas_sel_g, 3000);
    [~, acc_pred_plot_g] = downsample_series_for_plot(result.time_sel_s, result.acc_pred_sel_g, 3000);
    [freq_psd, meas_psd_g, pred_psd_g, db_diff] = compute_all_channel_psd_difference(result, inputs);
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
    axis off;
    text(0.5, 0.5, sprintf('Channel-by-channel PSD comparison is shown in a separate figure.\nX range: %.1f to %.1f Hz, X scale: %s, Y scale: %s', ...
        inputs.options.plot_psd_fmin_hz, inputs.options.plot_psd_fmax_hz, ...
        upper(inputs.options.plot_psd_xscale), upper(inputs.options.plot_psd_yscale)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 10);
    title('PSD comparison');

    plot_all_channel_psd_subplots(freq_psd, meas_psd_g, pred_psd_g, ...
        result.acc_meas_sel_g, result.acc_pred_sel_g, inputs);

    if inputs.options.plot_psd_error_map
        plot_all_channel_psd_error_map(freq_psd, db_diff, inputs.channel_names, inputs.options);
    end
end


function [freq_psd, meas_psd_g, pred_psd_g, db_diff] = compute_all_channel_psd_difference(result, inputs)
    n_channels = size(result.acc_meas_sel_g, 2);
    [nperseg, noverlap] = resolve_psd_welch_options(numel(result.time_sel_s), inputs.options);
    freq_psd = [];
    meas_psd_g = [];
    pred_psd_g = [];

    for channel_idx = 1:n_channels
        [freq_this, psd_meas] = welch_psd_1d(result.acc_meas_sel_g(:, channel_idx), result.fs_hz, nperseg, noverlap, inputs.options.fft_window);
        [freq_pred, psd_pred] = welch_psd_1d(result.acc_pred_sel_g(:, channel_idx), result.fs_hz, nperseg, noverlap, inputs.options.fft_window);

        if isempty(freq_psd)
            freq_psd = freq_this;
            meas_psd_g = zeros(numel(freq_psd), n_channels);
            pred_psd_g = zeros(numel(freq_psd), n_channels);
        elseif ~isequal(size(freq_psd), size(freq_this)) || any(abs(freq_psd - freq_this) > 1e-12) || ...
                ~isequal(size(freq_psd), size(freq_pred)) || any(abs(freq_psd - freq_pred) > 1e-12)
            error('PSD frequency grids do not match across channels.');
        end

        meas_psd_g(:, channel_idx) = psd_meas;
        pred_psd_g(:, channel_idx) = psd_pred;
    end

    db_diff = 10.0 * log10(max(pred_psd_g, 1e-30) ./ max(meas_psd_g, 1e-30));
end


function plot_all_channel_psd_subplots(freq_psd, meas_psd_g, pred_psd_g, acc_meas_sel_g, acc_pred_sel_g, inputs)
    if isempty(freq_psd) || isempty(meas_psd_g) || isempty(pred_psd_g)
        return;
    end

    n_channels = size(meas_psd_g, 2);
    n_cols = ceil(sqrt(n_channels));
    n_rows = ceil(n_channels / n_cols);
    guide_scale = db_power_ratio(3.0);

    figure('Name', 'All Channel PSD Comparison', 'Color', 'w');
    for channel_idx = 1:n_channels
        subplot(n_rows, n_cols, channel_idx);
        rms_meas_g = compute_rms_value(acc_meas_sel_g(:, channel_idx));
        rms_pred_g = compute_rms_value(acc_pred_sel_g(:, channel_idx));
        psd_meas = meas_psd_g(:, channel_idx);
        psd_pred = pred_psd_g(:, channel_idx);
        psd_meas_plus_3db = psd_meas * guide_scale;
        psd_meas_minus_3db = psd_meas / guide_scale;

        [freq_plot, psd_meas_plot] = prepare_psd_curve_for_plot(freq_psd, psd_meas, inputs.options);
        [freq_pred_plot, psd_pred_plot] = prepare_psd_curve_for_plot(freq_psd, psd_pred, inputs.options);
        [freq_plus_plot, psd_meas_plus_plot] = prepare_psd_curve_for_plot(freq_psd, psd_meas_plus_3db, inputs.options);
        [freq_minus_plot, psd_meas_minus_plot] = prepare_psd_curve_for_plot(freq_psd, psd_meas_minus_3db, inputs.options);

        hold on;
        plot(freq_plot, psd_meas_plot, 'b-', 'LineWidth', 1.0);
        plot(freq_pred_plot, psd_pred_plot, 'r--', 'LineWidth', 1.0);
        plot(freq_plus_plot, psd_meas_plus_plot, 'k:', 'LineWidth', 0.8);
        plot(freq_minus_plot, psd_meas_minus_plot, 'k:', 'LineWidth', 0.8);
        hold off;
        apply_psd_axes(gca, inputs.options);
        grid on;
        title(resolve_plot_channel_name(inputs.channel_names, channel_idx), 'Interpreter', 'none');
        xlabel('Frequency (Hz)');
        ylabel('g^2/Hz');
        legend(build_psd_legend_labels(rms_meas_g, rms_pred_g), 'Location', 'best', 'FontSize', 8);
    end
end


function plot_all_channel_psd_error_map(freq_psd, db_diff, channel_names, opts)
    if isempty(freq_psd) || isempty(db_diff)
        return;
    end

    n_channels = size(db_diff, 2);
    class_map = zeros(size(db_diff));
    class_map(db_diff < -6.0) = -2;
    class_map(db_diff >= -6.0 & db_diff < -3.0) = -1;
    class_map(abs(db_diff) <= 3.0) = 0;
    class_map(db_diff > 3.0 & db_diff <= 6.0) = 1;
    class_map(db_diff > 6.0) = 2;

    figure('Name', 'All Channel PSD Error Map', 'Color', 'w');

    subplot(2, 1, 1);
    imagesc(freq_psd, 1:n_channels, class_map.');
    axis xy;
    xlabel('Frequency (Hz)');
    ylabel('Channel');
    title('Predicted vs measured PSD difference (dB classes)');
    colormap(gca, [0.1294 0.4000 0.6745; 0.5922 0.7804 0.9020; 0.6510 0.8471 0.3294; 0.9922 0.7490 0.4353; 0.8431 0.1882 0.1529]);
    caxis([-2 2]);
    cb = colorbar;
    cb.Ticks = [-2, -1, 0, 1, 2];
    cb.TickLabels = {'< -6 dB', '-6 to -3 dB', '+/- 3 dB', '+3 to +6 dB', '> +6 dB'};
    set(gca, 'YTick', 1:n_channels, 'YTickLabel', build_channel_axis_labels(channel_names, n_channels));
    xlim([opts.plot_psd_fmin_hz, opts.plot_psd_fmax_hz]);

    subplot(2, 1, 2);
    imagesc(freq_psd, 1:n_channels, db_diff.');
    axis xy;
    xlabel('Frequency (Hz)');
    ylabel('Channel');
    title('Predicted minus measured PSD (dB)');
    colormap(gca, parula(256));
    caxis([-12 12]);
    colorbar;
    set(gca, 'YTick', 1:n_channels, 'YTickLabel', build_channel_axis_labels(channel_names, n_channels));
    xlim([opts.plot_psd_fmin_hz, opts.plot_psd_fmax_hz]);
end


function labels = build_channel_axis_labels(channel_names, n_channels)
    labels = cell(n_channels, 1);
    for idx = 1:n_channels
        labels{idx} = resolve_plot_channel_name(channel_names, idx);
    end
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


function [nperseg, noverlap] = resolve_psd_welch_options(n_time, opts)
    if isempty(opts.psd_nperseg)
        nperseg = default_psd_nperseg(n_time);
    else
        nperseg = min(round(opts.psd_nperseg), n_time);
    end
    nperseg = max(1, nperseg);

    if isempty(opts.psd_noverlap)
        noverlap = floor(nperseg / 2);
    else
        noverlap = min(round(opts.psd_noverlap), nperseg - 1);
    end
    noverlap = max(0, noverlap);
end


function labels = build_psd_legend_labels(rms_meas_g, rms_pred_g)
    labels = { ...
        sprintf('Measured RMS = %.4g g', rms_meas_g), ...
        sprintf('Predicted RMS = %.4g g', rms_pred_g), ...
        'Measured +3 dB', ...
        'Measured -3 dB'};
end


function rms_value = compute_rms_value(x)
    x = double(x(:));
    rms_value = sqrt(mean(x .^ 2));
end


function ratio = db_power_ratio(db_value)
    ratio = 10.0 ^ (db_value / 10.0);
end


function [freq_plot, pxx_plot] = prepare_psd_curve_for_plot(freq_psd, pxx, opts)
    freq_psd = freq_psd(:);
    pxx = pxx(:);
    mask = isfinite(freq_psd) & isfinite(pxx);
    mask = mask & freq_psd >= opts.plot_psd_fmin_hz & freq_psd <= opts.plot_psd_fmax_hz;
    if strcmp(opts.plot_psd_xscale, 'log')
        mask = mask & freq_psd > 0;
    end

    if ~any(mask)
        mask = isfinite(freq_psd) & isfinite(pxx);
        if strcmp(opts.plot_psd_xscale, 'log')
            mask = mask & freq_psd > 0;
        end
    end

    freq_plot = freq_psd(mask);
    pxx_plot = pxx(mask);
    if strcmp(opts.plot_psd_yscale, 'log')
        pxx_plot = max(pxx_plot, 1e-30);
    end
end


function apply_psd_axes(ax, opts)
    set(ax, 'XScale', opts.plot_psd_xscale, 'YScale', opts.plot_psd_yscale);
    xlim(ax, [opts.plot_psd_fmin_hz, opts.plot_psd_fmax_hz]);
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


function write_force_spectrum_csv(freqs_hz, F_hat, valid_mask, load_case_names, path_out)
    fid = fopen(path_out, 'w');
    if fid < 0
        error('Could not open %s for writing.', path_out);
    end
    cleanup = onCleanup(@() fclose(fid));

    header_parts = {'freq_hz'};
    for load_idx = 1:numel(load_case_names)
        token = sanitize_csv_token(load_case_names{load_idx});
        header_parts{end + 1} = sprintf('%s_re', token); %#ok<AGROW>
        header_parts{end + 1} = sprintf('%s_im', token); %#ok<AGROW>
    end
    header_parts{end + 1} = 'valid';
    fprintf(fid, '%s\n', strjoin(header_parts, ','));

    for k = 1:numel(freqs_hz)
        fprintf(fid, '%.16g', freqs_hz(k));
        for load_idx = 1:size(F_hat, 2)
            fprintf(fid, ',%.16g,%.16g', real(F_hat(k, load_idx)), imag(F_hat(k, load_idx)));
        end
        fprintf(fid, ',%d\n', logical_to_int(valid_mask(k)));
    end
end


function write_reconstruction_diagnostics_csv(result, path_out)
    fid = fopen(path_out, 'w');
    if fid < 0
        error('Could not open %s for writing.', path_out);
    end
    cleanup = onCleanup(@() fclose(fid));

    header_parts = {'freq_hz'};
    for load_idx = 1:numel(result.load_case_names)
        token = sanitize_csv_token(result.load_case_names{load_idx});
        header_parts{end + 1} = sprintf('%s_re', token); %#ok<AGROW>
        header_parts{end + 1} = sprintf('%s_im', token); %#ok<AGROW>
    end
    header_parts{end + 1} = 'cond_number';
    for sigma_idx = 1:size(result.singular_values, 2)
        header_parts{end + 1} = sprintf('sigma%d', sigma_idx); %#ok<AGROW>
    end
    header_parts{end + 1} = 'max_column_mac';
    if numel(result.load_case_names) == 3
        header_parts{end + 1} = 'mac_xy'; %#ok<AGROW>
        header_parts{end + 1} = 'mac_xz'; %#ok<AGROW>
        header_parts{end + 1} = 'mac_yz'; %#ok<AGROW>
    end
    header_parts{end + 1} = 'response_mac';
    header_parts{end + 1} = 'relative_residual';
    header_parts{end + 1} = 'valid';
    fprintf(fid, '%s\n', strjoin(header_parts, ','));

    for k = 1:numel(result.freqs_hz)
        fprintf(fid, '%.16g', result.freqs_hz(k));
        for load_idx = 1:size(result.F_hat, 2)
            fprintf(fid, ',%.16g,%.16g', real(result.F_hat(k, load_idx)), imag(result.F_hat(k, load_idx)));
        end
        fprintf(fid, ',%.16g', result.cond_number(k));
        for sigma_idx = 1:size(result.singular_values, 2)
            fprintf(fid, ',%.16g', result.singular_values(k, sigma_idx));
        end
        fprintf(fid, ',%.16g', result.max_column_mac(k));
        if numel(result.load_case_names) == 3
            fprintf(fid, ',%.16g,%.16g,%.16g', result.mac_xy(k), result.mac_xz(k), result.mac_yz(k));
        end
        fprintf(fid, ',%.16g,%.16g,%d\n', result.response_mac(k), result.relative_residual(k), ...
            logical_to_int(result.valid_mask(k)));
    end
end


function write_nastran_force_tabled1(freqs_hz, F_hat, valid_mask, load_case_names, path_out, force_unit, table_id_start)
    if numel(valid_mask) ~= numel(freqs_hz)
        error('valid_mask must match the frequency vector length for NASTRAN export.');
    end

    export_mask = logical(valid_mask(:));
    if ~any(export_mask)
        error('No valid frequency bins are available for NASTRAN export.');
    end

    freqs_export = freqs_hz(export_mask);
    F_export = convert_force_from_si(F_hat(export_mask, :), force_unit);

    fid = fopen(path_out, 'w');
    if fid < 0
        error('Could not open %s for writing.', path_out);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, '$ Reconstructed complex force spectrum for NASTRAN include\n');
    fprintf(fid, '$ Frequency units: Hz\n');
    fprintf(fid, '$ Force units: %s\n', upper(force_unit));
    fprintf(fid, '$ Each load case is exported as two TABLED1 entries: real and imaginary parts.\n');
    fprintf(fid, '$ Replay all reconstructed load cases together in the same SOL111 run to preserve relative phase.\n');
    fprintf(fid, '$ This file only writes the TABLED1 blocks; connect them to your deck using your preferred dynamic load cards.\n');
    fprintf(fid, '$ Load-case/table mapping:\n');

    next_tid = table_id_start;
    for load_idx = 1:numel(load_case_names)
        fprintf(fid, '$   %s : TABLED1 real=%d imag=%d\n', load_case_names{load_idx}, next_tid, next_tid + 1);
        next_tid = next_tid + 2;
    end
    fprintf(fid, '$\n');

    next_tid = table_id_start;
    for load_idx = 1:numel(load_case_names)
        fprintf(fid, '$ Load case: %s\n', load_case_names{load_idx});
        write_nastran_tabled1_block(fid, next_tid, freqs_export, real(F_export(:, load_idx)));
        write_nastran_tabled1_block(fid, next_tid + 1, freqs_export, imag(F_export(:, load_idx)));

        next_tid = next_tid + 2;
        if load_idx < numel(load_case_names)
            fprintf(fid, '$\n');
        end
    end
end


function write_nastran_tabled1_block(fid, tid, x_values, y_values)
    fprintf(fid, 'TABLED1,%d,LINEAR,LINEAR\n', tid);
    for idx = 1:numel(x_values)
        fprintf(fid, '+,%s,%s\n', format_nastran_real(x_values(idx)), format_nastran_real(y_values(idx)));
    end
    fprintf(fid, '+,ENDT\n');
end


function write_nastran_replay_bdf(path_out, tabled1_path, load_case_names, grid_ids, components, darea_scales, opts)
    fid = fopen(path_out, 'w');
    if fid < 0
        error('Could not open %s for writing.', path_out);
    end
    cleanup = onCleanup(@() fclose(fid));

    include_ref = nastran_include_reference(path_out, tabled1_path);

    fprintf(fid, 'SOL 111\n');
    fprintf(fid, 'CEND\n');
    fprintf(fid, 'TITLE = %s\n', opts.nastran_title);
    fprintf(fid, 'SUBCASE %d\n', opts.nastran_subcase_id);
    fprintf(fid, '  SPC = %d\n', opts.nastran_spc_sid);
    fprintf(fid, '  METHOD = %d\n', opts.nastran_method_sid);
    fprintf(fid, '  FREQ = %d\n', opts.nastran_freq_sid);
    fprintf(fid, '  DLOAD = %d\n', opts.nastran_dload_sid);
    if ~isempty(opts.nastran_sdamping_sid)
        fprintf(fid, '  SDAMPING = %d\n', opts.nastran_sdamping_sid);
    else
        fprintf(fid, '  $ SDAMPING = <set this if your model uses modal damping>\n');
    end
    fprintf(fid, '  DISPLACEMENT(PLOT) = ALL\n');
    fprintf(fid, '  VELOCITY(PLOT) = ALL\n');
    fprintf(fid, '  ACCELERATION(PLOT) = ALL\n');
    fprintf(fid, '\n');
    fprintf(fid, 'BEGIN BULK\n');
    fprintf(fid, '$ Reconstructed-force replay deck generated by reconstruct_forces_from_flight_csv.m\n');
    fprintf(fid, '$ Force units in %s. Replay all listed load cases together to preserve relative phase.\n', ...
        upper(opts.nastran_force_unit));
    if ~isempty(opts.nastran_model_include)
        fprintf(fid, 'INCLUDE ''%s''\n', opts.nastran_model_include);
    else
        fprintf(fid, '$ INCLUDE ''your_model.bdf''\n');
    end
    fprintf(fid, '\n');
        fprintf(fid, '$ DAREA: spatial definition of each reconstructed load case\n');
    for load_idx = 1:numel(load_case_names)
        fprintf(fid, '$   %s -> GRID %d COMP %d SCALE %s\n', ...
            load_case_names{load_idx}, grid_ids(load_idx), components(load_idx), ...
            format_nastran_real(darea_scales(load_idx)));
        fprintf(fid, 'DAREA,%d,%d,%d,%s\n', ...
            opts.nastran_darea_sid_start + load_idx - 1, ...
            grid_ids(load_idx), components(load_idx), format_nastran_real(darea_scales(load_idx)));
    end
    fprintf(fid, '\n');
    fprintf(fid, '$ RLOAD1: real/imaginary TABLED1 reference for each reconstructed load case\n');
    for load_idx = 1:numel(load_case_names)
        fprintf(fid, 'RLOAD1,%d,%d,,,%d,%d\n', ...
            opts.nastran_rload_sid_start + load_idx - 1, ...
            opts.nastran_darea_sid_start + load_idx - 1, ...
            opts.nastran_table_id_start + 2 * (load_idx - 1), ...
            opts.nastran_table_id_start + 2 * (load_idx - 1) + 1);
    end
    fprintf(fid, '\n');
    fprintf(fid, '$ DLOAD: combine all reconstructed loads in one SOL111 replay case\n');
    fprintf(fid, 'DLOAD,%d,%s', opts.nastran_dload_sid, format_nastran_real(1.0));
    for load_idx = 1:numel(load_case_names)
        fprintf(fid, ',%s,%d', format_nastran_real(1.0), opts.nastran_rload_sid_start + load_idx - 1);
    end
    fprintf(fid, '\n');
    fprintf(fid, '\n');
    fprintf(fid, 'INCLUDE ''%s''\n', include_ref);
    fprintf(fid, 'ENDDATA\n');
end


function tabled1_path = resolve_nastran_tabled1_output_path(tabled1_path_opt, replay_bdf_path)
    if ~isempty(tabled1_path_opt)
        tabled1_path = tabled1_path_opt;
        return;
    end

    [folder, stem, ~] = fileparts(replay_bdf_path);
    tabled1_path = fullfile(folder, [stem '_tables.inc']);
end


function include_ref = nastran_include_reference(deck_path, include_path)
    [deck_folder, ~, ~] = fileparts(deck_path);
    [include_folder, include_name, include_ext] = fileparts(include_path);
    if strcmp(deck_folder, include_folder)
        include_ref = [include_name include_ext];
    else
        include_ref = include_path;
    end
end


function [grid_ids, components, darea_scales] = resolve_nastran_replay_dofs(opts, result)
    n_loads = numel(result.load_case_names);

    grid_ids = normalize_nastran_numeric_option(opts.nastran_grid_ids, 'opts.nastran_grid_ids');
    if isempty(grid_ids)
        error(['opts.nastran_grid_ids is required when opts.save_nastran_replay_bdf is set. ' ...
            'Provide a scalar grid ID for the standard 3-load case or one grid ID per load case.']);
    end
    if isscalar(grid_ids) && n_loads == 3
        grid_ids = repmat(grid_ids, 1, n_loads);
    end
    if numel(grid_ids) ~= n_loads
        error('opts.nastran_grid_ids must have exactly %d entries, or one scalar for the standard 3-load case.', n_loads);
    end

    components = normalize_nastran_numeric_option(opts.nastran_components, 'opts.nastran_components');
    if isempty(components)
        if n_loads == 3
            components = [1 2 3];
        else
            error('opts.nastran_components must have exactly %d entries for replay export.', n_loads);
        end
    end
    if numel(components) ~= n_loads
        error('opts.nastran_components must have exactly %d entries.', n_loads);
    end

    darea_scales = normalize_nastran_numeric_option(opts.nastran_darea_scales, 'opts.nastran_darea_scales');
    if isempty(darea_scales)
        darea_scales = ones(1, n_loads);
    end
    if numel(darea_scales) ~= n_loads
        error('opts.nastran_darea_scales must have exactly %d entries.', n_loads);
    end

    grid_ids = validate_nastran_integer_vector(grid_ids, 'opts.nastran_grid_ids');
    components = validate_nastran_integer_vector(components, 'opts.nastran_components');
    if any(components < 0 | components > 6)
        error('opts.nastran_components entries must be integers from 0 to 6.');
    end
    if any(~isfinite(darea_scales))
        error('opts.nastran_darea_scales must contain only finite numeric values.');
    end
end


function values = normalize_nastran_numeric_option(value, arg_name)
    if isempty(value)
        values = [];
        return;
    end
    if ~isnumeric(value)
        error('%s must be numeric.', arg_name);
    end
    values = double(value(:)).';
end


function values = validate_nastran_integer_vector(values, arg_name)
    if any(~isfinite(values)) || any(values ~= round(values)) || any(values < 0)
        error('%s must contain non-negative integer values.', arg_name);
    end
end


function F_out = convert_force_from_si(F_in, force_unit)
    switch force_unit
        case 'n'
            scale = 1.0;
        case 'lbf'
            scale = 1.0 / 4.4482216152605;
        otherwise
            error('Unsupported force unit "%s".', force_unit);
    end

    F_out = F_in * scale;
end


function out = logical_to_int(value)
    if value
        out = 1;
    else
        out = 0;
    end
end


function token = sanitize_csv_token(text)
    token = regexprep(strtrim(text), '[^A-Za-z0-9_]+', '_');
    token = regexprep(token, '_+', '_');
    token = regexprep(token, '^_+', '');
    token = regexprep(token, '_+$', '');
    if isempty(token)
        token = 'load';
    end
    if ~isletter(token(1))
        token = ['load_' token];
    end
end


function value = validate_axis_scale_option(value, arg_name)
    if isstring(value)
        if ~isscalar(value)
            error('%s must be a scalar string or character vector.', arg_name);
        end
        value = char(value);
    end
    if ~ischar(value)
        error('%s must be ''log'' or ''linear''.', arg_name);
    end

    value = lower(strtrim(value));
    if ~strcmp(value, 'log') && ~strcmp(value, 'linear')
        error('%s must be ''log'' or ''linear''.', arg_name);
    end
end


function value = validate_force_unit_option(value, arg_name)
    if isstring(value)
        if ~isscalar(value)
            error('%s must be a scalar string or character vector.', arg_name);
        end
        value = char(value);
    end
    if ~ischar(value)
        error('%s must be ''N'' or ''lbf''.', arg_name);
    end

    value = lower(strtrim(value));
    if ~strcmp(value, 'n') && ~strcmp(value, 'lbf')
        error('%s must be ''N'' or ''lbf''.', arg_name);
    end
end


function validate_positive_integer_option(value, arg_name)
    if ~isscalar(value) || ~isnumeric(value) || ~isfinite(value) || value < 1 || value ~= round(value)
        error('%s must be a positive integer.', arg_name);
    end
end


function text = format_nastran_real(value)
    if ~isfinite(value)
        error('NASTRAN export only supports finite real values.');
    end
    text = sprintf('%.16E', value);
end
