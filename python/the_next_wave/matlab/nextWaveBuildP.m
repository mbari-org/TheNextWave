function P = nextWaveBuildP(kx, ky, omega, x, y, t, useVel)
%NEXTWAVEBUILDP Build a TheNextWave propagator matrix at arbitrary points.
%
%   P = nextWaveBuildP(kx, ky, omega, x, y, t)
%   P = nextWaveBuildP(kx, ky, omega, x, y, t, useVel)
%
%   Inputs (all column vectors):
%     kx, ky, omega : (N x 1) wave basis, from solve_basis.csv
%     x, y, t       : (M x 1) evaluation points (metres, metres, seconds)
%     useVel        : logical, default true. When true P includes the u and v
%                     row blocks; when false only the elevation block.
%
%   Output:
%     P : (3M x 2N) when useVel, else (M x 2N)
%
%   The model is
%
%       phi_j  = x*kx_j + y*ky_j - t*omega_j
%       eta    = sum_j A_cos_j*cos(phi_j) + A_sin_j*sin(phi_j)
%       u      = (kx_j/|k_j|)*omega_j * (same quadrature)
%       v      = (ky_j/|k_j|)*omega_j * (same quadrature)
%
%   so with A = [A_cos; A_sin] (2N x 1), P*A gives [eta; u; v] stacked.
%
%   The same routine builds P1 (pass the measurement x/y/t) and P2 (pass the
%   prediction x/y/t) -- they are the same formulation at different points.
%
%   Example -- reproduce one window's reconstruction and check it:
%
%     S  = decodeNextWaveBasis('solve/solve_basis.csv', 0);
%     T  = readtable('spread_training.csv');
%     T  = sortrows(T(T.window_id == 0, :), {'buoy_idx', 'sample_idx'});
%
%     P1 = nextWaveBuildP(S.kx, S.ky, S.omega, T.x_m, T.y_m, T.time_s);
%     r  = P1 * S.A;                       % [eta; u; v] stacked, 3M x 1
%
%     M    = height(T);
%     want = [T.z_recon_m; T.u_recon_mps; T.v_recon_mps];
%     fprintf('max abs diff: %.3e\n', max(abs(r - want)));
%
%     eta = r(1:M);                        % per block, if you want them split
%     u   = r(M+1:2*M);
%     v   = r(2*M+1:end);
%
%   Example -- forecast at a new location 30 s past the window:
%
%     n  = 64;
%     xq = repmat(-150, n, 1);
%     yq = repmat( -80, n, 1);
%     tq = S.windowEndTime + linspace(0, 30, n).';
%     eta = nextWaveBuildP(S.kx, S.ky, S.omega, xq, yq, tq, false) * S.A;
%
%   Row order matters: x/y/t must be ordered the way the solver stacked its
%   measurements -- column-major over (n_samples, n_buoys), i.e. every sample
%   for buoy 0, then buoy 1, and so on. Sorting the training table by
%   buoy_idx then sample_idx, as above, reproduces it.
%
%   See also DECODENEXTWAVEBASIS

if nargin < 7 || isempty(useVel)
    useVel = true;
end

kx = kx(:);  ky = ky(:);  omega = omega(:);
x  = x(:);   y  = y(:);   t     = t(:);

if ~isequal(numel(kx), numel(ky), numel(omega))
    error('nextWaveBuildP:basisSize', 'kx, ky and omega must be the same length.');
end
if ~isequal(numel(x), numel(y), numel(t))
    error('nextWaveBuildP:pointSize', 'x, y and t must be the same length.');
end

% Phase at every (point, component) pair: (M x N)
phi = x*kx.' + y*ky.' - t*omega.';
C = cos(phi);
S = sin(phi);

if ~useVel
    P = [C, S];
    return
end

% Deep-water Eulerian surface velocity factors, as row vectors (1 x N) so they
% broadcast down the columns of C and S.
kNorm = sqrt(kx.^2 + ky.^2);
kNorm(kNorm == 0) = 1;              % guard; matches the Python implementation
velX = ((kx ./ kNorm) .* omega).';
velY = ((ky ./ kNorm) .* omega).';

P = [           C,            S; ...
     velX .*    C, velX .*    S; ...
     velY .*    C, velY .*    S];
end
