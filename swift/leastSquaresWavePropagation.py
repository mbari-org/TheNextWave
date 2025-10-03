import numpy as np
import scipy.stats as sps
import scipy.optimize as spo

from .swift import LSQWavePropParams


def leastSquaresWavePropagation(z1, u1, v1, t1, x1, y1, t2, x2, y2, wavespec):
    #-> z2, zc, params, t:

    """
    %This function uses time series of vertical displacement and horizontal velocities to
    %generate a phase-resolved prediction of sea surface elevation at a specified time & location
    %using an inverse linear model.

    %input
    %z1:       vertical displacement time series where M is the length of record and 
    %          P is the number of point measurements.
    % u1,v1:   measurements of east, north velocities at sea surface (same size
    %          as z1) [m/s]
    %t1:       time stamp of measurements [seconds]
    %x1,y1:    easting, northing of measurement locations [meters]
    %wavespec: data structure containing the following fields
    %          Etheta - measured directional wave spectrum (Nautical convention)
    %          f - vector of wave frequencies [Hz]
    %          theta - vector of wave directions [degrees]

    %t2:       time stamp for prediction [seconds]
    %x2,y2:    easting, northing of target location for prediction [meters]
    """

    if len(u1) > 0 and len(v1) > 0:
        use_vel = True
    else:
        use_vel = False

    #tic;

    # convert wave spectrum to Cartesian coordinates (e.g. direction waves are moving TOWARDS)
    if wavespec.Etheta.shape[0] == len(wavespec.theta):
        wavespec.Etheta = wavespec.Etheta.T

    wavespec.theta, I = np.unique(wavespec.theta[::-1], return_index=True)  # return last indices of unique values
    wavespec.Etheta = wavespec.Etheta[:, I]
    t = wavespec.theta + 180.
    t[t > 360.] -= 360.
    I = np.argsort(t)
    wavespec.Etheta = wavespec.Etheta[:, I]

    idx = np.argmax(wavespec.Etheta.flatten(order='F'))
    c, r = np.unravel_index(idx, wavespec.Etheta.shape, order='F')
    DTp = np.radians(wavespec.theta(c))
    df = np.gradient(wavespec.f)

    # %Limit solution space to frequencies that statisfy S(f)/max(S(f))>5% &
    # %directions that statisfy DTp-pi/2 < DTp < DTp+pi/2

    wavespec.E = np.trapz(wavespec.Etheta.T, wavespec.theta)
    frange = np.where((df * wavespec.E.flatten(order='F')) / np.max(df * wavespec.E.flatten(order='F')) >= 0.05)[0]
    omega = np.logspace(np.log10(wavespec.f[frange[0]]), np.log10(wavespec.f[frange[-1]]), 40) * 2. * np.pi
    k = omega**2. / 9.81

    theta = np.linspace(DTp - np.pi/2., DTp + np.pi/2., 25)
    theta[theta > 2. * np.pi] -= 2. * np.pi
    theta[theta < 0.] += 2. * np.pi
    theta = np.sort(theta)

    # %Reshape input & check for consistency
    k = k.flatten()
    theta = theta.flatten()
    kx = np.sin(theta.T)
    ky = k * np.cos(theta.T)
    omega = np.sqrt(9.81 * k) * np.ones_like(theta.T)
    kx = kx.flatten()
    ky = ky.flatten()
    omega = omega.flatten()
    x1 = x1.flatten()
    y1 = y1.flatten()
    t1 = t1.flatten()
    z1 = z1.flatten()
    u1 = u1.flatten()
    v1 = v1.flatten()
    x2 = x2.flatten()
    y2 = y2.flatten()
    t2 = t2.flatten()

    N_input_pts = len(z1)
    if len(x1) != N_input_pts or len(y1) != N_input_pts or len(t1) != N_input_pts:
        print('Error: All input vectors must be equal length')

    N_output_pts = len(t2)
    if len(x2) != N_output_pts or len(y2) != N_output_pts:
        print('Error: All output vectors must be equal length')

    # %Interpolate Observed Spectrum to Solution Space
    F, T = np.meshgrid(wavespec.f, wavespec.theta)
    f2, thet2 = np.meshgrid(np.sqrt(k * 9.8), theta)
    Ei = 10. ** griddata(F, T, np.log10(wavespec.Etheta.T), f2 / (2. * np.pi), np.degrees(thet2))
    Ei[np.isnan(Ei)] = 0.

    Ei *= np.trapz(wavespec.f, np.trapz(wavespec.theta, wavespec.Etheta.T)) / np.trapz(f2[0,:] / (2. * np.pi), np.trapz(np.degrees(thet2[:,0]), Ei))
    amps = np.sqrt(Ei * np.diff([0, f2[0, :] / (2. * np.pi)]) * sps.mode(np.diff(np.degrees(thet2[:, 0])), axis=None, keepdims=False).mode.item())
    amps = amps.T
    amps = np.concatenate((amps.flatten(order='F'), amps.flatten(order='F')), axis=0)
    amps[isnan(amps)] = 0.

    # %Construct Propagator Matrices
    phi1 = x1 * kx.T + y1 * ky.T - t1 * omega.T
    phi2 = x2 * kx.T + y2 * ky.T - t2 * omega.T

    # %P1: Used to invert measured wave data (M1 x N)
    # %P2: Used to predict at target location/time (M2 x N)
    # %Note: P1 and P2 are consistent formulations, but M1 may be different than M2.

    if use_vel:
        P1 = [[np.cos(phi1), np.sin(phi1)],
              [(kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi1), (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi1)],
              [(ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi1), (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi1)]]

        P2 = [[np.cos(phi2), np.sin(phi2)],
              [(kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi2), (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi2)],
              [(ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi2), (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi2)]]
    else:
        P1 = [np.cos(phi1), np.sin(phi1)]
        P2 = [np.cos(phi2), np.sin(phi2)]

    good = np.nonzero(amps[:np.size(Ei)] != 0)[0]
    zero_mask = (amps == 0)
    if zero_mask.any():
        keep_mask = ~zero_mask
        P1 = P1[:, keep_mask]
        P2 = P2[:, keep_mask]
        amps = amps[keep_mask]

    # %Invert linear model to solve for unknown wave amplitudes using a
    # %bounded least-squares approach.
    # options = optimoptions('lsqlin', 'Algorithm', 'trust-region-reflective')
    if use_vel:
        b = np.concatenate((np.asarray(z1).ravel(order='F'),
                            np.asarray(u1).ravel(order='F'),
                            np.asarray(v1).ravel(order='F')), axis=0)
    else:
        b = np.asarray(z1).ravel(order='F')
    res = spo.lsq_linear(P1, b, bounds=(-amps / 1.4142, amps / 1.4142), method='trf')
    A = res.x
    zc = P1 @ A
    z2 = P2 @ A

    # %% bookkeeping
    params = LSQWavePropParams()
    params.A = A;
    params.Etheta = np.zeros_like(Ei.flatten(order='F')).T
    params.Etheta[good] = (A[:(len(A) // 2)]**2. + A[(len(A) // 2 + 1):]**2.) / 2.
    params.Etheta = params.Etheta.reshape((len(k), len(theta)), order='F').T
    params.Etheta /= np.diff(np.concatenate(([0.], f2[0,:] / (2. * np.pi)))) * sps.mode(np.diff(np.degrees(thet2[:, 0])), axis=None, keepdims=False).mode.item()
    params.f = f2[0, :].T / (2. * np.pi)
    params.theta = np.degrees(thet2[:, 0].T)
    params.theta += 180.
    params.theta[params.theta > 360.] -= 360.
    I = np.argsort(params.theta)
    params.theta = params.theta[I]
    params.Etheta = params.Etheta[I,:].T
    params.kx = kx[good]
    params.ky = ky[good]
    params.omega = omega[good]
    params.use_vel = use_vel

    return z2, zc, params, t
