function S = decodeNextWaveBasis(basisCsv, windowId)
%DECODENEXTWAVEBASIS Read solve_basis.csv back into the wave basis and A.
%
%   S = decodeNextWaveBasis(basisCsv)            % first window in the file
%   S = decodeNextWaveBasis(basisCsv, windowId)  % a specific window
%   ids = decodeNextWaveBasis(basisCsv, 'list')  % available window ids
%
%   solve_basis.csv holds one row per wave component per solve window. This
%   pulls one window out, ordered by component_idx, and returns:
%
%     S.windowId, S.windowStartTime, S.windowEndTime
%     S.componentIdx  (N x 1)  position, also the index into A_cos / A_sin
%     S.kx, S.ky      (N x 1)  wavenumber components [rad/m]
%     S.omega         (N x 1)  angular frequency [rad/s]
%     S.amp           (N x 1)  per-component amplitude scale [m]
%     S.bound         (N x 1)  solver box bound = amp/1.4142 [m]
%     S.Acos, S.Asin  (N x 1)  solved quadrature amplitudes [m]
%     S.A             (2N x 1) = [Acos; Asin], the vector P multiplies
%
%   To rebuild the propagator and evaluate the model at any points:
%
%     S = decodeNextWaveBasis('solve_basis.csv', 0);
%     P = nextWaveBuildP(S.kx, S.ky, S.omega, x, y, t);
%     fields = P * S.A;            % [eta; u; v] stacked, 3M x 1
%
%   Passing the measurement x/y/time from <case>_training.csv reproduces that
%   window's reconstruction; any other points give the model elsewhere.
%
%   Note: the Python builds P in single precision and solves in double, so
%   expect agreement to roughly 1e-6 relative, not to machine precision.
%
%   See also NEXTWAVEBUILDP

arguments
    basisCsv (1,:) char
    windowId = []
end

T = readtable(basisCsv);

required = {'window_id', 'component_idx', 'kx_rad_per_m', 'ky_rad_per_m', ...
            'omega_rad_per_s', 'amp_m', 'bound_m', 'A_cos_m', 'A_sin_m'};
missing = setdiff(required, T.Properties.VariableNames);
if ~isempty(missing)
    error('decodeNextWaveBasis:columns', ...
          'Missing column(s) in %s: %s', basisCsv, strjoin(missing, ', '));
end

ids = unique(T.window_id);

if ischar(windowId) || isstring(windowId)
    if ~strcmpi(windowId, 'list')
        error('decodeNextWaveBasis:badArg', ...
              'Second argument must be a window id or ''list''.');
    end
    S = ids;
    return
end

if isempty(windowId)
    windowId = ids(1);
end

rows = T(T.window_id == windowId, :);
if isempty(rows)
    error('decodeNextWaveBasis:noWindow', ...
          'window_id %g not in %s (available: %g to %g, %d windows).', ...
          windowId, basisCsv, min(ids), max(ids), numel(ids));
end

% Components are written in order, but sort so the result never depends on
% file ordering; component_idx is the index into A_cos / A_sin.
rows = sortrows(rows, 'component_idx');

S = struct();
S.windowId        = windowId;
S.componentIdx    = rows.component_idx;
S.kx              = rows.kx_rad_per_m;
S.ky              = rows.ky_rad_per_m;
S.omega           = rows.omega_rad_per_s;
S.amp             = rows.amp_m;
S.bound           = rows.bound_m;
S.Acos            = rows.A_cos_m;
S.Asin            = rows.A_sin_m;
S.A               = [S.Acos; S.Asin];

if ismember('window_start_time', T.Properties.VariableNames)
    S.windowStartTime = rows.window_start_time(1);
end
if ismember('window_end_time', T.Properties.VariableNames)
    S.windowEndTime = rows.window_end_time(1);
end

% The file records how many components the solve used; flag any disagreement
% rather than silently returning a short basis.
if ismember('n_components', T.Properties.VariableNames)
    nExpected = rows.n_components(1);
    if height(rows) ~= nExpected
        warning('decodeNextWaveBasis:componentCount', ...
                'window %g: found %d rows but n_components says %d.', ...
                windowId, height(rows), nExpected);
    end
end
end
