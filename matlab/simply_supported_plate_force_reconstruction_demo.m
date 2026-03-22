% simply_supported_plate_force_reconstruction_demo.m
%
% Self-contained MATLAB translation of the Python simply-supported plate
% demo in this repository. The goal is math verification, not UI parity.
%
% The script does four things:
% 1) Build a modal mobility FRF for a simply supported plate.
% 2) Create synthetic accelerometer time data from known complex force tones.
% 3) Reconstruct the forces with a frequency-by-frequency least-squares solve.
% 4) Print a few verification metrics and plot the result.
%
% Important MATLAB note:
% All local helper functions are placed at the end of the file.

clear;
clc;

% -----------------------------
% User-adjustable demo settings
% -----------------------------
params = struct();

% Plate model
params.length_m = 0.8;
params.width_m = 0.6;
params.thickness_m = 0.004;
params.youngs_modulus_pa = 70.0e9;
params.poisson_ratio = 0.33;
params.density_kg_m3 = 2700.0;
params.damping_ratio = 0.012;
params.m_max = 6;
params.n_max = 6;

% Time / frequency grids
params.fs_hz = 1024.0;
params.duration_s = 4.0;
params.frf_df_hz = 1.0;

% Noise and regularization
params.noise_std_g = 0.0;
params.seed = 7;
params.tikhonov_lambda = 1e-7;

% Reconstruction controls
params.t_start = 0.0;
params.t_end = [];
params.f_min_hz = 5.0;
params.f_max_hz = 220.0;
params.fft_window = 'boxcar';

% Use standard gravity and the same imperial/SI conversions as Python.
params.g0 = 9.80665;
params.lbf_to_n = 4.4482216152605;
params.in_to_m = 0.0254;

% -----------------------------
% Generate synthetic plate case
% -----------------------------
caseData = generate_simply_supported_plate_case(params);

if isempty(params.t_end)
    params.t_end = caseData.time_s(end);
end

% -----------------------------
% Reconstruct the force spectrum
% -----------------------------
result = reconstruct_force_spectrum(caseData, params);

% ---------------------------------------------------------
% Compare estimated forces against the known synthetic truth
% ---------------------------------------------------------
activeMask = caseData.force_active_mask & result.valid_mask;

if nnz(activeMask) == 0
    error('No active frequency bins survived the reconstruction filters.');
end

fTrue = caseData.true_force_fft(activeMask, :);
fHat = result.F_hat(activeMask, :);
forceDiff = fHat - fTrue;

relativeForceError = norm(forceDiff(:)) / (norm(fTrue(:)) + 1e-14);
medianResponseMac = median_finite(result.response_mac(activeMask));
medianRelativeResidual = median_finite(result.relative_residual(activeMask));

fprintf('\n');
fprintf('Simply supported plate force reconstruction demo\n');
fprintf('------------------------------------------------\n');
fprintf('Active force bins used          : %d\n', nnz(activeMask));
fprintf('Relative force error            : %.6f\n', relativeForceError);
fprintf('Median response MAC             : %.6f\n', medianResponseMac);
fprintf('Median relative residual        : %.6f\n', medianRelativeResidual);
fprintf('\n');

% These are the same acceptance targets used by the Python plate test.
if relativeForceError >= 0.05
    error('Relative force error exceeded the expected threshold of 0.05.');
end
if medianResponseMac <= 0.99
    error('Median response MAC fell below the expected threshold of 0.99.');
end
if medianRelativeResidual >= 0.03
    error('Median relative residual exceeded the expected threshold of 0.03.');
end

fprintf('Verification checks passed.\n');

plot_reconstruction_results(caseData, result, activeMask);


function caseData = generate_simply_supported_plate_case(params)
% Build the same synthetic plate demo used in the Python test.
%
% Three point loads are treated as the three unknown force channels.
% Eight acceleration channels are generated from the plate FRF.

    nTime = round(params.duration_s * params.fs_hz);
    if nTime < 32
        error('Need at least 32 time samples.');
    end

    time_s = (0:nTime-1).' / params.fs_hz;
    fft_freqs_hz = (0:floor(nTime / 2)).' * (params.fs_hz / nTime);

    sensorFrac = [
        0.14 0.18
        0.31 0.24
        0.49 0.30
        0.68 0.22
        0.19 0.58
        0.38 0.66
        0.57 0.62
        0.76 0.74
    ];

    loadFrac = [
        0.24 0.36
        0.53 0.46
        0.71 0.28
    ];

    sensor_xy_m = fraction_points(sensorFrac, params.length_m, params.width_m);
    load_xy_m = fraction_points(loadFrac, params.length_m, params.width_m);

    nSensors = size(sensor_xy_m, 1);
    nLoads = size(load_xy_m, 1);

    flexuralRigidity = params.youngs_modulus_pa * params.thickness_m^3 / ...
        (12.0 * (1.0 - params.poisson_ratio^2));
    arealMass = params.density_kg_m3 * params.thickness_m;

    [modeIndices, coupling] = modal_coupling( ...
        sensor_xy_m, ...
        load_xy_m, ...
        params.length_m, ...
        params.width_m, ...
        arealMass, ...
        params.m_max, ...
        params.n_max);

    mIdx = modeIndices(:, 1);
    nIdx = modeIndices(:, 2);
    omegaModes = sqrt(flexuralRigidity / arealMass) .* ...
        ((mIdx * pi / params.length_m).^2 + (nIdx * pi / params.width_m).^2);

    nyquist_hz = params.fs_hz / 2.0;
    frf_freqs_hz = (0:params.frf_df_hz:nyquist_hz).';

    H_si = complex(zeros(numel(frf_freqs_hz), nSensors, nLoads));

    for k = 1:numel(frf_freqs_hz)
        omega = 2.0 * pi * frf_freqs_hz(k);
        den = omegaModes.^2 - omega.^2 + 1i * 2.0 * params.damping_ratio .* omegaModes .* omega;

        receptance = complex(zeros(nSensors, nLoads));
        for modeIdx = 1:numel(den)
            receptance = receptance + squeeze(coupling(modeIdx, :, :)) / den(modeIdx);
        end

        H_si(k, :, :) = reshape(1i * omega * receptance, [1, nSensors, nLoads]);
    end

    % Mirror the Python path exactly:
    % generate in SI, convert to imperial, then reconstruction converts back.
    H_imperial = H_si * (params.lbf_to_n / params.in_to_m);

    toneFreqsHz = [31.0, 59.0, 97.0, 143.0];
    toneVectors = [
         8.0 + 2.5i, -1.0 + 0.6i,  0.7 - 0.4i
        -2.4 + 1.0i,  6.5 - 2.0i,  0.8 + 0.2i
         1.2 - 0.8i, -1.0 + 0.3i,  7.2 + 1.6i
         0.0 - 1.1i,  1.8 - 0.7i, -4.8 + 1.9i
    ];

    true_force_fft = complex(zeros(numel(fft_freqs_hz), 3));
    for i = 1:numel(toneFreqsHz)
        [~, idx] = min(abs(fft_freqs_hz - toneFreqsHz(i)));
        true_force_fft(idx, :) = true_force_fft(idx, :) + toneVectors(i, :);
    end

    omegaFrf = 2.0 * pi * frf_freqs_hz;
    A_frf = accelerance_from_mobility(H_si, omegaFrf);
    A_fft = interp_complex_frequency(fft_freqs_hz, frf_freqs_hz, A_frf, 0.0 + 0.0i);

    a_fft = complex(zeros(numel(fft_freqs_hz), nSensors));
    for k = 1:numel(fft_freqs_hz)
        Ak = squeeze(A_fft(k, :, :));
        fk = true_force_fft(k, :).';
        a_fft(k, :) = (Ak * fk).';
    end

    acc_m_s2 = irfft_matrix(a_fft, nTime);
    acc_g = acc_m_s2 / params.g0;

    if params.noise_std_g > 0
        rng(params.seed);
        acc_g = acc_g + params.noise_std_g * randn(size(acc_g));
    end

    force_active_mask = sqrt(sum(abs(true_force_fft).^2, 2)) > 0;

    caseData = struct();
    caseData.time_s = time_s;
    caseData.acc_g = acc_g;
    caseData.channel_names = arrayfun(@(k) sprintf('plate_s%d', k), 1:nSensors, 'UniformOutput', false);
    caseData.fft_freqs_hz = fft_freqs_hz;
    caseData.frf_freqs_hz = frf_freqs_hz;
    caseData.H_si = H_si;
    caseData.H_imperial = H_imperial;
    caseData.true_force_fft = true_force_fft;
    caseData.force_active_mask = force_active_mask;
end


function result = reconstruct_force_spectrum(caseData, params)
% Match the Python reconstruction pipeline as directly as possible.
%
% Steps:
% 1) Slice the requested time window.
% 2) Convert acceleration from g to m/s^2.
% 3) Compute a one-sided complex FFT for each sensor channel.
% 4) Convert mobility to accelerance.
% 5) Interpolate the FRF onto the FFT frequency grid.
% 6) Solve A * F ~= a at each frequency bin.

    timeMask = caseData.time_s >= params.t_start & caseData.time_s <= params.t_end;
    time_s = caseData.time_s(timeMask);
    acc_g = caseData.acc_g(timeMask, :);

    if numel(time_s) < 8
        error('Time slice too short for FFT.');
    end

    fs_hz = infer_fs(time_s);
    acc_m_s2 = acc_g * params.g0;
    [freqs_hz, a_meas_fft] = complex_rfft_matrix(acc_m_s2, fs_hz, params.fft_window);

    H_si = mobility_imp_to_si(caseData.H_imperial, params.in_to_m, params.lbf_to_n);
    A_frf = accelerance_from_mobility(H_si, 2.0 * pi * caseData.frf_freqs_hz);
    A_fft = interp_complex_frequency(freqs_hz, caseData.frf_freqs_hz, A_frf, complex(NaN, NaN));

    nFreq = numel(freqs_hz);
    nSensors = size(a_meas_fft, 2);

    F_hat = complex(zeros(nFreq, 3));
    a_pred_fft = complex(zeros(nFreq, nSensors));
    cond_number = NaN(nFreq, 1);
    singular_values = NaN(nFreq, 3);
    response_mac_values = NaN(nFreq, 1);
    relative_residual = NaN(nFreq, 1);
    valid_mask = false(nFreq, 1);

    for k = 1:nFreq
        fk = freqs_hz(k);

        if fk <= 0
            continue;
        end
        if ~isempty(params.f_min_hz) && fk < params.f_min_hz
            continue;
        end
        if ~isempty(params.f_max_hz) && fk > params.f_max_hz
            continue;
        end

        Ak = squeeze(A_fft(k, :, :));
        if any(~isfinite(Ak(:)))
            continue;
        end

        ak = a_meas_fft(k, :).';
        Fk = solve_force_complex(Ak, ak, params.tikhonov_lambda);
        aPred = Ak * Fk;

        s = svd(Ak, 'econ');
        singular_values(k, 1:numel(s)) = s(:).';
        if s(end) > 0 && isfinite(s(end))
            cond_number(k) = s(1) / s(end);
        else
            cond_number(k) = Inf;
        end

        F_hat(k, :) = Fk.';
        a_pred_fft(k, :) = aPred.';

        denom = norm(ak);
        if denom > 0
            relative_residual(k) = norm(aPred - ak) / denom;
        else
            relative_residual(k) = 0.0;
        end

        response_mac_values(k) = response_mac(ak, aPred);
        valid_mask(k) = true;
    end

    result = struct();
    result.freqs_hz = freqs_hz;
    result.F_hat = F_hat;
    result.a_meas_fft = a_meas_fft;
    result.a_pred_fft = a_pred_fft;
    result.cond_number = cond_number;
    result.singular_values = singular_values;
    result.response_mac = response_mac_values;
    result.relative_residual = relative_residual;
    result.valid_mask = valid_mask;
end


function plot_reconstruction_results(caseData, result, activeMask)
% Keep the plotting very simple so the math is easy to inspect.

    figure('Name', 'Simply Supported Plate Reconstruction', 'Color', 'w');

    for forceIdx = 1:3
        subplot(2, 2, forceIdx);
        hold on;
        plot(caseData.fft_freqs_hz, abs(caseData.true_force_fft(:, forceIdx)), 'k-', 'LineWidth', 1.5);
        plot(result.freqs_hz, abs(result.F_hat(:, forceIdx)), 'r--', 'LineWidth', 1.2);
        xlabel('Frequency (Hz)');
        ylabel('|F|');
        title(sprintf('Force channel %d', forceIdx));
        legend('True', 'Reconstructed', 'Location', 'northoutside');
        grid on;
        xlim([0, 220]);
        hold off;
    end

    subplot(2, 2, 4);
    hold on;
    plot(result.freqs_hz, result.response_mac, 'b-', 'LineWidth', 1.2);
    plot(result.freqs_hz, result.relative_residual, 'm-', 'LineWidth', 1.2);
    plot(result.freqs_hz(activeMask), result.response_mac(activeMask), 'bo', 'MarkerSize', 5);
    plot(result.freqs_hz(activeMask), result.relative_residual(activeMask), 'mo', 'MarkerSize', 5);
    xlabel('Frequency (Hz)');
    ylabel('Metric value');
    title('Reconstruction quality');
    legend('Response MAC', 'Relative residual', 'Active MAC', 'Active residual', ...
        'Location', 'southoutside');
    grid on;
    xlim([0, 220]);
    ylim([0, 1.05]);
    hold off;
end


function xy_m = fraction_points(fracXY, lx, ly)
% Convert fractional plate coordinates into meters.

    xy_m = fracXY;
    xy_m(:, 1) = xy_m(:, 1) * lx;
    xy_m(:, 2) = xy_m(:, 2) * ly;
end


function [modeIndices, coupling] = modal_coupling(sensor_xy_m, load_xy_m, length_m, width_m, arealMass, mMax, nMax)
% Return:
% modeIndices : [nModes x 2] table of (m, n) plate mode numbers
% coupling    : [nModes x nSensors x nLoads] modal participation tensor

    modalMass = arealMass * length_m * width_m / 4.0;
    nSensors = size(sensor_xy_m, 1);
    nLoads = size(load_xy_m, 1);
    nModes = mMax * nMax;

    modeIndices = zeros(nModes, 2);
    coupling = complex(zeros(nModes, nSensors, nLoads));

    modeCounter = 0;
    for m = 1:mMax
        for n = 1:nMax
            modeCounter = modeCounter + 1;

            phiSensor = sin(m * pi * sensor_xy_m(:, 1) / length_m) .* ...
                sin(n * pi * sensor_xy_m(:, 2) / width_m);
            phiLoad = sin(m * pi * load_xy_m(:, 1) / length_m) .* ...
                sin(n * pi * load_xy_m(:, 2) / width_m);

            modeIndices(modeCounter, :) = [m, n];
            coupling(modeCounter, :, :) = reshape((phiSensor * phiLoad.') / modalMass, [1, nSensors, nLoads]);
        end
    end
end


function H_si = mobility_imp_to_si(H_imperial, in_to_m, lbf_to_n)
% Convert mobility from (in/s)/lbf to (m/s)/N.

    scale = in_to_m / lbf_to_n;
    H_si = H_imperial * scale;
end


function A = accelerance_from_mobility(H, omega_rad_s)
% A = j * omega * H
%
% H is [nFreq x nSensors x 3]
% omega_rad_s is [nFreq x 1]

    A = H;
    for k = 1:numel(omega_rad_s)
        A(k, :, :) = reshape(1i * omega_rad_s(k) * squeeze(H(k, :, :)), [1, size(H, 2), size(H, 3)]);
    end
end


function out = interp_complex_frequency(fTargetHz, fSourceHz, values, fillValue)
% Linear interpolation for complex FRF data.
% Real and imaginary parts are interpolated separately.

    originalSize = size(values);
    nSource = originalSize(1);
    nColumns = prod(originalSize(2:end));

    values2d = reshape(values, nSource, nColumns);
    out2d = complex(zeros(numel(fTargetHz), nColumns));

    for j = 1:nColumns
        realPart = interp1(fSourceHz, real(values2d(:, j)), fTargetHz, 'linear', NaN);
        imagPart = interp1(fSourceHz, imag(values2d(:, j)), fTargetHz, 'linear', NaN);
        out2d(:, j) = complex(realPart, imagPart);
    end

    fillIsNaN = isnan(real(fillValue)) || isnan(imag(fillValue));
    if ~fillIsNaN
        badMask = isnan(real(out2d)) | isnan(imag(out2d));
        out2d(badMask) = fillValue;
    end

    out = reshape(out2d, [numel(fTargetHz), originalSize(2:end)]);
end


function [freqs_hz, spectrum] = complex_rfft_matrix(x, fs_hz, windowName)
% One-sided complex FFT per column.
%
% The scaling matches the Python code:
% u = x .* window .* sqrt(n / sum(window.^2))
% spectrum = rfft(u)

    if isvector(x)
        x = x(:);
    end

    n = size(x, 1);
    nCh = size(x, 2);
    win = get_window_vector(windowName, n);

    windowEnergy = sum(win.^2);
    if windowEnergy > 0
        scale = sqrt(n / windowEnergy);
    else
        scale = 1.0;
    end

    freqs_hz = (0:floor(n / 2)).' * (fs_hz / n);
    spectrum = complex(zeros(numel(freqs_hz), nCh));

    for j = 1:nCh
        u = x(:, j) .* win * scale;
        X = fft(u);
        spectrum(:, j) = X(1:numel(freqs_hz));
    end
end


function win = get_window_vector(windowName, n)
% Minimal window helper. Only the two windows used by the Python app are kept.

    switch lower(windowName)
        case {'boxcar', 'rect', 'rectangular'}
            win = ones(n, 1);
        case 'hann'
            if n == 1
                win = 1;
            else
                % Periodic Hann to match SciPy get_window(..., fftbins=True).
                idx = (0:n-1).';
                win = 0.5 - 0.5 * cos(2.0 * pi * idx / n);
            end
        otherwise
            error('Unsupported window "%s". Use "boxcar" or "hann".', windowName);
    end
end


function x = irfft_matrix(oneSidedSpectrum, nTime)
% Inverse of a one-sided FFT for real-valued signals.

    if mod(nTime, 2) == 0
        mirrored = conj(oneSidedSpectrum(end-1:-1:2, :));
    else
        mirrored = conj(oneSidedSpectrum(end:-1:2, :));
    end

    fullSpectrum = [oneSidedSpectrum; mirrored];
    x = real(ifft(fullSpectrum, nTime, 1));
end


function fs_hz = infer_fs(time_s)
% Sampling rate from the median time step.

    dt = diff(time_s);
    medianDt = median(dt);
    if medianDt <= 0
        error('Time vector must be strictly increasing.');
    end
    fs_hz = 1.0 / medianDt;
end


function F = solve_force_complex(A, a, lambda)
% Solve min ||A*F - a||^2 + lambda^2 ||F||^2.
%
% This is written in the explicit normal-equation form so the math is easy
% to verify by inspection:
%   (A' * A + lambda^2 * I) * F = A' * a
%
% In MATLAB, A' is the Hermitian (conjugate) transpose for complex data.

    normalMatrix = A' * A;
    rhs = A' * a;

    if lambda > 0
        normalMatrix = normalMatrix + (lambda^2) * eye(3);
    end

    F = normalMatrix \ rhs;
end


function mac = response_mac(aMeasured, aPredicted)
% Modal assurance criterion between two complex sensor vectors.

    numerator = abs(aMeasured' * aPredicted)^2;
    denominator = real(aMeasured' * aMeasured) * real(aPredicted' * aPredicted);

    if denominator > 0
        mac = numerator / denominator;
    else
        mac = 0.0;
    end
end


function value = median_finite(x)
% Median that ignores NaN/Inf without needing toolbox-specific options.

    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = median(x);
    end
end
