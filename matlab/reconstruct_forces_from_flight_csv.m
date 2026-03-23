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
%   opts (optional struct)
%       t_start               analysis start time [s], default = first sample
%       t_end                 analysis end time [s], default = last sample
%       g0                    standard gravity, default = 9.80665
%       tikhonov_lambda       force regularization lambda, default = 0
%       mobility_is_si        true if H is already (m/s)/N, default = false
%       skip_zero_hz          skip the DC bin, default = true
%       f_min_hz              optional lower analysis limit
%       f_max_hz              optional upper analysis limit
%       fft_window            'hann' or 'boxcar', default = 'hann'
%       plot_results          plot reconstructed forces + diagnostics, default = false
%       verbose               print summary metrics, default = true
%       save_force_csv        optional output path for F_hat CSV
%       save_diagnostics_csv  optional output path for diagnostics CSV
%
% Outputs
%   result
%       Struct aligned with the Python pipeline fields:
%       freqs_hz, F_hat, a_meas_fft, a_pred_fft, cond_number,
%       singular_values, mac_xy, mac_xz, mac_yz, response_mac,
%       relative_residual, valid_mask, fs_hz, t_start, t_end
%   inputs
%       Loaded source data and resolved options.
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

    assert_file_exists(pathFx, 'pathFx');
    assert_file_exists(pathFy, 'pathFy');
    assert_file_exists(pathFz, 'pathFz');
    assert_flight_input_exists(flightInput, 'flightInput');

    [frf_freqs_hz, H_input] = load_mobility_csv_triplet(pathFx, pathFy, pathFz);
    [time_s, acc_g, channel_names, flight_input_type] = load_flight_input(flightInput);

    opts = resolve_options(opts, time_s);
    if ~isempty(opts.f_min_hz) && ~isempty(opts.f_max_hz) && opts.f_max_hz < opts.f_min_hz
        error('opts.f_max_hz must be >= opts.f_min_hz.');
    end
    if opts.t_end < opts.t_start
        error('opts.t_end must be >= opts.t_start.');
    end

    if size(H_input, 2) ~= size(acc_g, 2)
        error('H has %d sensors, but the flight input has %d channels.', size(H_input, 2), size(acc_g, 2));
    end

    time_mask = time_s >= opts.t_start & time_s <= opts.t_end;
    time_sel = time_s(time_mask);
    acc_sel_g = acc_g(time_mask, :);

    if numel(time_sel) < 8
        error('Time slice too short for FFT. Need at least 8 samples after windowing.');
    end

    fs_hz = infer_fs(time_sel);
    acc_sel_si = acc_sel_g * opts.g0;
    [freqs_hz, a_meas_fft] = complex_rfft_matrix(acc_sel_si, fs_hz, opts.fft_window);

    H_si = H_input;
    if ~opts.mobility_is_si
        H_si = mobility_imp_to_si(H_input);
    end

    A_frf = accelerance_from_mobility(H_si, 2.0 * pi * frf_freqs_hz);
    A_fft = interp_complex_frequency(freqs_hz, frf_freqs_hz, A_frf, complex(NaN, NaN));

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

    for k = 1:n_freq
        fk = freqs_hz(k);

        if opts.skip_zero_hz && fk <= 0
            continue;
        end
        if ~isempty(opts.f_min_hz) && fk < opts.f_min_hz
            continue;
        end
        if ~isempty(opts.f_max_hz) && fk > opts.f_max_hz
            continue;
        end

        Ak = squeeze(A_fft(k, :, :));
        if any(~isfinite(Ak(:)))
            continue;
        end

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

    inputs = struct();
    inputs.path_fx = pathFx;
    inputs.path_fy = pathFy;
    inputs.path_fz = pathFz;
    inputs.path_flight = flightInput;
    inputs.flight_input = flightInput;
    inputs.flight_input_type = flight_input_type;
    inputs.frf_freqs_hz = frf_freqs_hz;
    inputs.H_input = H_input;
    inputs.time_s = time_s;
    inputs.acc_g = acc_g;
    inputs.channel_names = channel_names;
    inputs.options = opts;

    if opts.verbose
        print_summary(result);
    end

    if ~isempty(opts.save_force_csv)
        write_force_spectrum_csv(result.freqs_hz, result.F_hat, result.valid_mask, opts.save_force_csv);
    end
    if ~isempty(opts.save_diagnostics_csv)
        write_reconstruction_diagnostics_csv(result, opts.save_diagnostics_csv);
    end
    if opts.plot_results
        plot_reconstruction_results(result);
    end
end


function opts = resolve_options(opts, time_s)
    defaults = struct();
    defaults.t_start = time_s(1);
    defaults.t_end = time_s(end);
    defaults.g0 = 9.80665;
    defaults.tikhonov_lambda = 0.0;
    defaults.mobility_is_si = false;
    defaults.skip_zero_hz = true;
    defaults.f_min_hz = [];
    defaults.f_max_hz = [];
    defaults.fft_window = 'hann';
    defaults.plot_results = false;
    defaults.verbose = true;
    defaults.save_force_csv = '';
    defaults.save_diagnostics_csv = '';

    names = fieldnames(defaults);
    for idx = 1:numel(names)
        name = names{idx};
        if ~isfield(opts, name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
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


function [time_s, acc_g, names, input_type] = load_flight_input(flightInput)
    if ischar(flightInput) || (isstring(flightInput) && isscalar(flightInput))
        pathValue = char(flightInput);
        [~, ~, ext] = fileparts(pathValue);
        if strcmpi(ext, '.mat')
            [time_s, acc_g, names] = load_flight_mat_files({pathValue});
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
        [time_s, acc_g, names] = load_flight_mat_files(flightInput);
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


function [time_s, acc_g, names] = load_flight_mat_files(matPaths)
    if isstring(matPaths)
        matPaths = cellstr(matPaths(:));
    end
    if isempty(matPaths)
        error('At least one .mat flight channel file is required.');
    end

    n_channels = numel(matPaths);
    names = cell(1, n_channels);
    acc_g = [];
    time_s = [];
    sr_ref = [];

    for idx = 1:n_channels
        pathValue = char(matPaths{idx});
        [~, ~, ext] = fileparts(pathValue);
        if ~strcmpi(ext, '.mat')
            error('Per-channel flight inputs must be .mat files. Received: %s', pathValue);
        end

        [amp_g, t_this, sr_this] = load_single_channel_mat(pathValue);
        [~, baseName, ~] = fileparts(pathValue);
        names{idx} = baseName;

        if isempty(time_s)
            time_s = t_this;
            sr_ref = sr_this;
            acc_g = zeros(numel(t_this), n_channels);
        else
            validate_matching_timebase(t_this, sr_this, time_s, sr_ref, pathValue);
        end

        acc_g(:, idx) = amp_g;
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


function validate_matching_timebase(time_s, sr_hz, time_ref, sr_ref, pathValue)
    if numel(time_s) ~= numel(time_ref)
        error('Channel file %s has %d samples, expected %d.', pathValue, numel(time_s), numel(time_ref));
    end

    sr_tol = max(1e-9, 1e-6 * max(abs(sr_ref), 1.0));
    if abs(sr_hz - sr_ref) > sr_tol
        error('Channel file %s has sample rate %.16g Hz, expected %.16g Hz.', pathValue, sr_hz, sr_ref);
    end

    time_tol = max(1e-9, 1e-6 / max(abs(sr_ref), 1.0));
    if any(abs(time_s - time_ref) > time_tol)
        error('Channel file %s does not share the same time vector as the other channels.', pathValue);
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
    med_dt = median(dt);
    if med_dt <= 0
        error('Time vector must be strictly increasing.');
    end
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
    fprintf('Valid frequency bins         : %d / %d\n', nnz(valid), numel(valid));
    fprintf('Median response MAC          : %.6f\n', median_finite(result.response_mac(valid)));
    fprintf('Median relative residual     : %.6f\n', median_finite(result.relative_residual(valid)));
    fprintf('Median condition number      : %.6f\n', median_finite(result.cond_number(valid)));
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


function plot_reconstruction_results(result)
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
