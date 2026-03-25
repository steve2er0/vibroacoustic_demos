function result = compute_psd_from_complex_response_csv(pathResponseCsv, opts)
%COMPUTE_PSD_FROM_COMPLEX_RESPONSE_CSV Build pseudo PSD / CPSD from complex harmonic response CSV.
%
%   result = compute_psd_from_complex_response_csv(pathResponseCsv)
%   result = compute_psd_from_complex_response_csv(pathResponseCsv, opts)
%
% Input CSV format:
%   freq_hz,re0,im0,re1,im1,...
% or named pairs:
%   freq_hz,node1_t3_re,node1_t3_im,node2_t3_re,node2_t3_im,...
%
% Required input:
%   pathResponseCsv
%       Complex harmonic response CSV path. The first column is frequency in
%       Hz. Remaining columns are real/imaginary response pairs.
%
% Optional opts fields:
%   df_hz
%       Frequency spacing in Hz. Default = inferred from the frequency vector.
%   phasor_convention
%       'peak' or 'rms'. Default = 'peak'. Use 'peak' for the usual SOL111
%       real/imaginary harmonic-response phasor interpretation.
%   input_accel_unit
%       Input acceleration unit: 'native', 'g', 'm/s^2', or 'in/s^2'.
%       Default = 'native' (no assumed conversion).
%   output_accel_unit
%       Output acceleration unit: 'native', 'g', 'm/s^2', or 'in/s^2'.
%       Default = 'native'. If either input or output is 'native', both must
%       be 'native'.
%   save_auto_psd_csv
%       Output CSV path for auto PSD channels. Default =
%       <input>_auto_psd.csv. Set to '' to skip writing.
%   save_cpsd_csv
%       Output CSV path for upper-triangle CPSD terms. Default =
%       <input>_cpsd.csv. Set to '' to skip writing.
%   verbose
%       Print summary/output paths, default = true.
%
% Output result fields:
%   freqs_hz
%       Frequency vector [Hz]
%   response_complex
%       Complex response matrix in output_accel_unit, shape (n_freq, n_ch)
%   response_complex_input
%       Raw complex response matrix exactly as loaded from the CSV
%   channel_names
%       Parsed channel labels
%   df_hz
%       Frequency spacing [Hz]
%   phasor_convention
%       Resolved convention string
%   input_accel_unit
%       Resolved input acceleration unit
%   output_accel_unit
%       Resolved output acceleration unit
%   response_scale
%       Scalar applied to the complex response before PSD/CPSD formation
%   auto_psd
%       Auto PSD matrix, shape (n_freq, n_ch)
%   cpsd
%       Complex spectral matrix, shape (n_freq, n_ch, n_ch)
%
% Notes
%   For complex harmonic response lines X(f_k), this function computes:
%
%       Sxx(f_k) = alpha * X(f_k) * X(f_k)^H / df
%
%   where alpha = 0.5 for peak phasors and alpha = 1.0 for RMS phasors.
%   The output units are the square of the response units per Hz.

    if nargin < 2 || isempty(opts)
        opts = struct();
    end
    if ~(ischar(pathResponseCsv) || (isstring(pathResponseCsv) && isscalar(pathResponseCsv)))
        error('pathResponseCsv must be a character vector or scalar string.');
    end
    pathResponseCsv = char(pathResponseCsv);
    if ~isfile(pathResponseCsv)
        error('Response CSV not found: %s', pathResponseCsv);
    end

    opts = resolve_options(opts, pathResponseCsv);

    [freqs_hz, response_complex_input, channel_names] = load_complex_response_csv(pathResponseCsv);
    if isempty(opts.df_hz)
        df_hz = infer_uniform_df_hz(freqs_hz);
    else
        df_hz = opts.df_hz;
    end

    response_scale = accel_unit_scale(opts.input_accel_unit, opts.output_accel_unit);
    response_complex = response_complex_input * response_scale;

    switch opts.phasor_convention
        case 'peak'
            power_scale = 0.5;
        case 'rms'
            power_scale = 1.0;
        otherwise
            error('Unsupported phasor_convention "%s".', opts.phasor_convention);
    end

    n_freq = numel(freqs_hz);
    n_ch = size(response_complex, 2);
    cpsd = complex(zeros(n_freq, n_ch, n_ch));
    auto_psd = zeros(n_freq, n_ch);
    for freq_idx = 1:n_freq
        xk = response_complex(freq_idx, :).';
        Sk = (power_scale / df_hz) * (xk * xk');
        cpsd(freq_idx, :, :) = Sk;
        auto_psd(freq_idx, :) = real(diag(Sk)).';
    end

    result = struct();
    result.freqs_hz = freqs_hz;
    result.response_complex_input = response_complex_input;
    result.response_complex = response_complex;
    result.channel_names = channel_names;
    result.df_hz = df_hz;
    result.phasor_convention = opts.phasor_convention;
    result.input_accel_unit = opts.input_accel_unit;
    result.output_accel_unit = opts.output_accel_unit;
    result.response_scale = response_scale;
    result.auto_psd = auto_psd;
    result.cpsd = cpsd;

    if ~isempty(opts.save_auto_psd_csv)
        write_auto_psd_csv(opts.save_auto_psd_csv, freqs_hz, auto_psd, channel_names);
        result.save_auto_psd_csv = opts.save_auto_psd_csv;
    else
        result.save_auto_psd_csv = '';
    end
    if ~isempty(opts.save_cpsd_csv)
        write_cpsd_csv(opts.save_cpsd_csv, freqs_hz, cpsd, channel_names);
        result.save_cpsd_csv = opts.save_cpsd_csv;
    else
        result.save_cpsd_csv = '';
    end

    if opts.verbose
        fprintf('Computed pseudo PSD / CPSD from %s\n', pathResponseCsv);
        fprintf('Channels                  : %d\n', n_ch);
        fprintf('Frequency lines           : %d\n', n_freq);
        fprintf('df_hz                     : %.16g\n', df_hz);
        fprintf('Phasor convention         : %s\n', opts.phasor_convention);
        fprintf('Input accel unit          : %s\n', opts.input_accel_unit);
        fprintf('Output accel unit         : %s\n', opts.output_accel_unit);
        fprintf('Response scale            : %.16g\n', response_scale);
        if ~isempty(result.save_auto_psd_csv)
            fprintf('Auto PSD CSV              : %s\n', result.save_auto_psd_csv);
        end
        if ~isempty(result.save_cpsd_csv)
            fprintf('CPSD CSV                  : %s\n', result.save_cpsd_csv);
        end
    end
end


function opts = resolve_options(opts, pathResponseCsv)
    if ~isstruct(opts)
        error('opts must be a struct when provided.');
    end

    default_auto = default_output_path(pathResponseCsv, '_auto_psd.csv');
    default_cpsd = default_output_path(pathResponseCsv, '_cpsd.csv');

    if ~isfield(opts, 'df_hz')
        opts.df_hz = [];
    end
    if ~isfield(opts, 'phasor_convention') || isempty(opts.phasor_convention)
        opts.phasor_convention = 'peak';
    end
    if ~isfield(opts, 'input_accel_unit') || isempty(opts.input_accel_unit)
        opts.input_accel_unit = 'native';
    end
    if ~isfield(opts, 'output_accel_unit') || isempty(opts.output_accel_unit)
        opts.output_accel_unit = 'native';
    end
    if ~isfield(opts, 'save_auto_psd_csv')
        opts.save_auto_psd_csv = default_auto;
    end
    if ~isfield(opts, 'save_cpsd_csv')
        opts.save_cpsd_csv = default_cpsd;
    end
    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = true;
    end

    opts.phasor_convention = validate_phasor_convention(opts.phasor_convention);
    opts.input_accel_unit = validate_accel_unit(opts.input_accel_unit, 'opts.input_accel_unit');
    opts.output_accel_unit = validate_accel_unit(opts.output_accel_unit, 'opts.output_accel_unit');
    opts.verbose = logical(opts.verbose);

    if ~isempty(opts.df_hz)
        if ~isscalar(opts.df_hz) || ~isnumeric(opts.df_hz) || ~isfinite(opts.df_hz) || opts.df_hz <= 0
            error('opts.df_hz must be empty or a positive finite scalar.');
        end
    end

    opts.save_auto_psd_csv = normalize_optional_path(opts.save_auto_psd_csv, 'opts.save_auto_psd_csv');
    opts.save_cpsd_csv = normalize_optional_path(opts.save_cpsd_csv, 'opts.save_cpsd_csv');
end


function scale = accel_unit_scale(input_unit, output_unit)
    if strcmp(input_unit, 'native') || strcmp(output_unit, 'native')
        if strcmp(input_unit, output_unit)
            scale = 1.0;
            return;
        end
        error(['When using acceleration-unit conversion, set both opts.input_accel_unit ' ...
            'and opts.output_accel_unit explicitly.']);
    end

    scale = accel_unit_to_mps2(input_unit) / accel_unit_to_mps2(output_unit);
end


function value = validate_accel_unit(value, arg_name)
    if isstring(value)
        if ~isscalar(value)
            error('%s must be a scalar string or character vector.', arg_name);
        end
        value = char(value);
    end
    if ~ischar(value)
        error('%s must be ''native'', ''g'', ''m/s^2'', or ''in/s^2''.', arg_name);
    end

    value = lower(strtrim(value));
    aliases = containers.Map( ...
        {'native', 'g', 'm/s^2', 'm/s2', 'mps2', 'in/s^2', 'in/s2', 'ips2'}, ...
        {'native', 'g', 'm/s^2', 'm/s^2', 'm/s^2', 'in/s^2', 'in/s^2', 'in/s^2'});
    if ~isKey(aliases, value)
        error('%s must be ''native'', ''g'', ''m/s^2'', or ''in/s^2''.', arg_name);
    end
    value = aliases(value);
end


function scale = accel_unit_to_mps2(unit_name)
    switch unit_name
        case 'g'
            scale = 9.80665;
        case 'm/s^2'
            scale = 1.0;
        case 'in/s^2'
            scale = 0.0254;
        otherwise
            error('Unsupported acceleration unit "%s".', unit_name);
    end
end


function value = normalize_optional_path(value, arg_name)
    if isempty(value)
        value = '';
        return;
    end
    if isstring(value)
        if ~isscalar(value)
            error('%s must be a scalar string or character vector.', arg_name);
        end
        value = char(value);
    end
    if ~ischar(value)
        error('%s must be empty, a character vector, or a scalar string.', arg_name);
    end
end


function path_out = default_output_path(path_in, suffix)
    [folder, stem, ~] = fileparts(path_in);
    path_out = fullfile(folder, [stem suffix]);
end


function value = validate_phasor_convention(value)
    if isstring(value)
        if ~isscalar(value)
            error('opts.phasor_convention must be a scalar string or character vector.');
        end
        value = char(value);
    end
    if ~ischar(value)
        error('opts.phasor_convention must be ''peak'' or ''rms''.');
    end
    value = lower(strtrim(value));
    if ~strcmp(value, 'peak') && ~strcmp(value, 'rms')
        error('opts.phasor_convention must be ''peak'' or ''rms''.');
    end
end


function [freqs_hz, response_complex, channel_names] = load_complex_response_csv(pathValue)
    header = read_csv_header(pathValue);
    data = read_numeric_csv(pathValue, 1);

    if size(data, 2) < 3
        error('Expected frequency column followed by at least one re/im pair in %s.', pathValue);
    end

    freqs_hz = data(:, 1);
    rest = data(:, 2:end);
    if mod(size(rest, 2), 2) ~= 0
        error('Expected an even number of re/im columns after frequency in %s.', pathValue);
    end

    response_complex = complex(rest(:, 1:2:end), rest(:, 2:2:end));
    channel_names = derive_channel_names(header, size(response_complex, 2));
end


function header = read_csv_header(pathValue)
    fid = fopen(pathValue, 'r');
    if fid < 0
        error('Could not open %s for reading.', pathValue);
    end
    cleanup = onCleanup(@() fclose(fid));
    line = fgetl(fid);
    if ~ischar(line)
        error('Could not read CSV header from %s.', pathValue);
    end
    header = strsplit(strtrim(line), ',');
    for idx = 1:numel(header)
        header{idx} = strtrim(regexprep(header{idx}, '^#+', ''));
    end
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


function names = derive_channel_names(header, n_ch)
    names = cell(1, n_ch);
    if numel(header) < (1 + 2 * n_ch)
        for idx = 1:n_ch
            names{idx} = sprintf('ch%d', idx - 1);
        end
        return;
    end

    for idx = 1:n_ch
        re_col = header{2 * idx};
        im_col = header{2 * idx + 1};
        names{idx} = pair_name_from_columns(re_col, im_col, idx - 1);
    end
end


function name = pair_name_from_columns(re_col, im_col, idx_zero_based)
    re_match = regexp(re_col, '^(.*)_re$', 'tokens', 'once', 'ignorecase');
    im_match = regexp(im_col, '^(.*)_im$', 'tokens', 'once', 'ignorecase');
    if ~isempty(re_match) && ~isempty(im_match) && strcmp(re_match{1}, im_match{1}) && ~isempty(re_match{1})
        name = re_match{1};
        return;
    end

    if ~isempty(regexp(re_col, '^re\d+$', 'once', 'ignorecase')) && ...
            ~isempty(regexp(im_col, '^im\d+$', 'once', 'ignorecase'))
        name = sprintf('ch%d', idx_zero_based);
        return;
    end

    name = sprintf('ch%d', idx_zero_based);
end


function df_hz = infer_uniform_df_hz(freqs_hz)
    freqs_hz = freqs_hz(:);
    if numel(freqs_hz) < 2
        error('Need at least two frequency samples to infer df.');
    end

    df = diff(freqs_hz);
    df_hz = df(1);
    if df_hz <= 0
        error('Frequency vector must be strictly increasing.');
    end

    tol = max(1e-12, 1e-9 * max(max(abs(freqs_hz)), 1.0));
    if any(abs(df - df_hz) > tol)
        error('Frequency vector is not uniformly spaced; supply opts.df_hz explicitly.');
    end
end


function write_auto_psd_csv(path_out, freqs_hz, auto_psd, channel_names)
    fid = fopen(path_out, 'w');
    if fid < 0
        error('Could not open %s for writing.', path_out);
    end
    cleanup = onCleanup(@() fclose(fid));

    header_parts = {'freq_hz'};
    for idx = 1:numel(channel_names)
        header_parts{end + 1} = sprintf('%s_psd', sanitize_csv_token(channel_names{idx})); %#ok<AGROW>
    end
    fprintf(fid, '%s\n', strjoin(header_parts, ','));

    for k = 1:numel(freqs_hz)
        fprintf(fid, '%.16g', freqs_hz(k));
        for idx = 1:size(auto_psd, 2)
            fprintf(fid, ',%.16g', auto_psd(k, idx));
        end
        fprintf(fid, '\n');
    end
end


function write_cpsd_csv(path_out, freqs_hz, cpsd, channel_names)
    fid = fopen(path_out, 'w');
    if fid < 0
        error('Could not open %s for writing.', path_out);
    end
    cleanup = onCleanup(@() fclose(fid));

    header_parts = {'freq_hz'};
    n_ch = numel(channel_names);
    for i = 1:n_ch
        name_i = sanitize_csv_token(channel_names{i});
        for j = i:n_ch
            name_j = sanitize_csv_token(channel_names{j});
            token = sprintf('%s__%s', name_i, name_j);
            header_parts{end + 1} = sprintf('%s_re', token); %#ok<AGROW>
            header_parts{end + 1} = sprintf('%s_im', token); %#ok<AGROW>
        end
    end
    fprintf(fid, '%s\n', strjoin(header_parts, ','));

    for k = 1:numel(freqs_hz)
        fprintf(fid, '%.16g', freqs_hz(k));
        for i = 1:n_ch
            for j = i:n_ch
                value = cpsd(k, i, j);
                fprintf(fid, ',%.16g,%.16g', real(value), imag(value));
            end
        end
        fprintf(fid, '\n');
    end
end


function token = sanitize_csv_token(text)
    token = regexprep(strtrim(text), '[^A-Za-z0-9_]+', '_');
    token = regexprep(token, '_+', '_');
    token = regexprep(token, '^_+', '');
    token = regexprep(token, '_+$', '');
    if isempty(token)
        token = 'ch';
    end
    if ~isletter(token(1))
        token = ['ch_' token];
    end
end
