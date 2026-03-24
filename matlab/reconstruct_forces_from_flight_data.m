function varargout = reconstruct_forces_from_flight_data(varargin)
%RECONSTRUCT_FORCES_FROM_FLIGHT_DATA Preferred MATLAB entrypoint for flight CSV or .mat inputs.
%
% This is a thin wrapper around reconstruct_forces_from_flight_csv for a
% less CSV-specific API name. The mobility input may be either:
% 1) Three legacy CSV paths for Fx, Fy, Fz
% 2) An ordered list of one or more mobility CSVs defining arbitrary load cases
%
% The flight input may be either:
% 1) A flight CSV path: time_s,ch0,ch1,...
% 2) An ordered list of per-channel .mat files, each containing one
%    structure with fields amp, t, and sr.

    [varargout{1:nargout}] = reconstruct_forces_from_flight_csv(varargin{:});
end
