import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import scipy.interpolate as spint

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
    DTp = np.radians(wavespec.theta[c])
    df = np.gradient(wavespec.f)

    # %Limit solution space to frequencies that statisfy S(f)/max(S(f))>5% &
    # %directions that statisfy DTp-pi/2 < DTp < DTp+pi/2

    wavespec.E = np.trapz(wavespec.Etheta.T, x=wavespec.theta, axis=0)
    frange = np.where((df * wavespec.E.flatten(order='F')) / np.max(df * wavespec.E.flatten(order='F')) >= 0.05)[0]
    omega = np.logspace(np.log10(wavespec.f[frange[0]]), np.log10(wavespec.f[frange[-1]]), 40) * 2. * np.pi
    k = omega**2. / 9.81

    theta = np.linspace(DTp - np.pi/2., DTp + np.pi/2., 25)
    theta[theta > 2. * np.pi] -= 2. * np.pi
    theta[theta < 0.] += 2. * np.pi
    theta = np.sort(theta)

    # print(f'{k.shape=} {theta.T.shape=}')

    # %Reshape input & check for consistency
    k = k.flatten()
    theta = theta.flatten()
    kx = np.outer(k, np.sin(theta))
    ky = np.outer(k, np.cos(theta))
    omega = np.outer(np.sqrt(9.81 * k), np.ones_like(theta))
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
    F, T = np.meshgrid(wavespec.f, wavespec.theta)    # shape (M, N)
    f2, thet2 = np.meshgrid(np.sqrt(k * 9.8), theta)  # shape of target grid
    points = np.column_stack((F.ravel(), T.ravel()))                       # (M*N, 2)
    values = np.log10(wavespec.Etheta.T).ravel()                           # (M*N,)
    xi = (f2 / (2. * np.pi), np.degrees(thet2))  # both are 2-D arrays
    zi_log = spint.griddata(points, values, xi, method='linear')   # shape == xi[0].shape
    Ei = 10.0 ** zi_log
    Ei[np.isnan(Ei)] = 0.

    Ei *= np.trapz(wavespec.E, x=wavespec.f, axis=0) / np.trapz(np.trapz(Ei, x=np.degrees(thet2[:,0]), axis=0), x=f2[0,:] / (2. * np.pi), axis=0)
    amps = np.sqrt(
        Ei * np.diff(
            f2[0, :] / (2. * np.pi),
            prepend=0.0
        ) * sps.mode(
            np.diff(np.degrees(thet2[:, 0])),
            axis=None,
            keepdims=False
        ).mode.item()
    )
    # print(f'{amps.shape=}')
    amps = amps.T
    amps = np.concatenate((amps.flatten(order='F'), amps.flatten(order='F')), axis=0)
    amps[np.isnan(amps)] = 0.
    # print(f'{amps.shape=}')

    x1 = x1.reshape((-1, 1))
    x2 = x2.reshape((-1, 1))
    y1 = y1.reshape((-1, 1))
    y2 = y2.reshape((-1, 1))
    kx = kx.reshape((-1, 1))
    ky = ky.reshape((-1, 1))
    t1 = t1.reshape((-1, 1))
    t2 = t2.reshape((-1, 1))
    omega = omega.reshape((-1, 1))

    # print(f'{x1.shape=}')
    # print(f'{x2.shape=}')
    # print(f'{y1.shape=}')
    # print(f'{y2.shape=}')
    # print(f'{kx.T.shape=}')
    # print(f'{ky.T.shape=}')
    # print(f'{t1.shape=}')
    # print(f'{t2.shape=}')
    # print(f'{omega.T.shape=}')

    # %Construct Propagator Matrices
    phi1 = x1 @ kx.T + y1 @ ky.T - t1 @ omega.T
    phi2 = x2 @ kx.T + y2 @ ky.T - t2 @ omega.T

    # %P1: Used to invert measured wave data (M1 x N)
    # %P2: Used to predict at target location/time (M2 x N)
    # %Note: P1 and P2 are consistent formulations, but M1 may be different than M2.

    if use_vel:
        P1_11 = np.cos(phi1)
        P1_21 = (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi1)
        P1_31 = (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi1)
        P1_12 = np.sin(phi1)
        P1_22 = (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi1)
        P1_32 = (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi1)

        row0 = np.hstack((P1_11, P1_12))   # (1044, 2000)
        row1 = np.hstack((P1_21, P1_22))   # (1044, 2000)
        row2 = np.hstack((P1_31, P1_32))   # (1044, 2000)

        P1 = np.vstack((row0, row1, row2))  # (3132, 2000)
        # P1 = np.array([[np.cos(phi1), np.sin(phi1)],
        #       [(kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi1), (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi1)],
        #       [(ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi1), (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi1)]])

        P2_11 = np.cos(phi2)
        P2_21 = (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi2)
        P2_31 = (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi2)
        P2_12 = np.sin(phi2)
        P2_22 = (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi2)
        P2_32 = (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi2)

        row0 = np.hstack((P2_11, P2_12))
        row1 = np.hstack((P2_21, P2_22))
        row2 = np.hstack((P2_31, P2_32))

        P2 = np.vstack((row0, row1, row2))
        # P2 = np.array([[np.cos(phi2), np.sin(phi2)],
        #       [(kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi2), (kx / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi2)],
        #       [(ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.cos(phi2), (ky / np.sqrt(kx**2. + ky**2.)).T * omega.T * np.sin(phi2)]])
    else:
        P1 = np.hstack([np.cos(phi1), np.sin(phi1)])
        P2 = np.hstack([np.cos(phi2), np.sin(phi2)])

    good = np.nonzero(amps[:np.size(Ei)] != 0)[0]
    zero_mask = (amps == 0)
    if zero_mask.any():
        keep_mask = ~zero_mask
        P1 = P1[:, keep_mask]
        P2 = P2[:, keep_mask]
        amps = amps[keep_mask]
        # print(f'{amps.shape=}')

    # %Invert linear model to solve for unknown wave amplitudes using a
    # %bounded least-squares approach.
    # options = optimoptions('lsqlin', 'Algorithm', 'trust-region-reflective')
    if use_vel:
        b = np.concatenate((np.asarray(z1).ravel(order='F'),
                            np.asarray(u1).ravel(order='F'),
                            np.asarray(v1).ravel(order='F')), axis=0)
    else:
        b = np.asarray(z1).ravel(order='F')

    # print(f'{P1.shape=} {b.shape=} {amps.shape=}')
    import time
    t_0 = time.time()
    #A_unb, *_ = np.linalg.lstsq(P1, b, rcond=None)
    #res = spo.lsq_linear(P1, b, bounds=(-amps / 1.4142, amps / 1.4142), method='bvls')
    #A = res.x

    # turn off warnings about galahad SSIDS
    import os
    os.environ["OMP_CANCELLATION"] = "FALSE"
    os.environ["OMP_PROC_BIND"] = "FALSE"
    os.environ["OMP_DISPLAY_ENV"] = "FALSE"

    # Hard-disable SSIDS if your build honors these:
    os.environ["GALAHAD_USE_SSIDS"] = "0"
    os.environ["HSL_USE_SSIDS"] = "0"

    # Silence gfortran output units:
    os.environ["GFORTRAN_STDERR_UNIT"] = "-1"
    os.environ["GFORTRAN_STDOUT_UNIT"] = "-1"

    # OPTIONAL: Avoid any thread binding at all
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    from galahad import bllsb

    o, n = P1.shape  # o = 3132, n = 2000
    x_l = -amps / 1.4142
    x_u =  amps / 1.4142
    w = np.ones(o)
    sigma = 1.0
    Ao_row, Ao_col = np.indices((o, n))
    Ao_row = Ao_row.ravel(order="F")     # match MATLAB column-major order
    Ao_col = Ao_col.ravel(order="F")
    Ao_val = P1.ravel(order="F")         # values
    Ao_ne  = Ao_val.size                 # total number of entries
    Ao_type = "coordinate"
    Ao_ptr  = None   # not used for COO
    options = bllsb.initialize()
    #print(options['sls_options'].keys())
    # options["use_ssids"] = False
    # options["ssids_aware"] = False        # some versions expect this too
    # options["use_linalg"] = True          # fallback dense LAPACK solver
    options["print_level"] = 0     # quiet model
    bllsb.load(n, o, Ao_type, Ao_ne, Ao_row, Ao_col, 0, Ao_ptr, options)
    x0 = np.zeros(n)
    z0 = np.zeros(n)
    A, r, z, x_stat = bllsb.solve_bllsb(
        n, o,
        Ao_ne, Ao_val,
        b,
        sigma,
        x_l, x_u,
        x0, z0,
        w
    )
    inform = bllsb.information()
    # print("Objective:", inform["obj"])
    bllsb.terminate()
    # print("A shape:", A.shape)
    # print("Residual norm:", np.linalg.norm(r))
    t = time.time() - t_0
    print(f'solve time: {t=}')
    # print(f'{A.shape=}')

    zc = P1 @ A
    z2 = P2 @ A

    # print(f'{zc.shape=} {z2.shape=}')

    # %% bookkeeping
    params = LSQWavePropParams()
    params.A = A
    params.Etheta = np.zeros_like(Ei.flatten(order='F')).T
    params.Etheta[good] = (A[:(len(A) // 2)]**2. + A[(len(A) // 2):]**2.) / 2.
    params.Etheta = params.Etheta.reshape((len(k), len(theta)), order='F').T
    params.Etheta /= np.diff(f2[0,:] / (2. * np.pi), prepend=0.0) * sps.mode(np.diff(np.degrees(thet2[:, 0])), axis=None, keepdims=False).mode.item()
    # print(f'{params.Etheta.shape=}')
    # print(f'{params.Etheta.sum()=}')
    params.f = (f2[0, :] / (2. * np.pi)).flatten()
    params.theta = np.degrees(thet2[:, 0])
    params.theta += 180.
    params.theta[params.theta > 360.] -= 360.
    I = np.argsort(params.theta)
    params.theta = params.theta[I].flatten()
    params.Etheta = params.Etheta[I,:].T
    # print(f'{params.f.shape=}')
    # print(f'{params.theta.shape=}')
    # print(f'{params.Etheta.shape=}')
    params.kx = kx[good].flatten()
    params.ky = ky[good].flatten()
    params.omega = omega[good].flatten()
    params.use_vel = use_vel

    return z2.reshape((-1, 1)), zc.reshape((-1, 1)), params, t
