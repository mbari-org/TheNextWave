import numpy as np


def MEM_directionalestimator(a1, a2, b1, b2, en, convert):
    """
    % function [NS,NE] = MEM_calc(a1,a2,b1,b2,en,convert)
    %
    % This function calculates the Maximum Entropy Method estimate of
    % the Directional Distribution of a wave field.
    %
    % NOTE: The normalized directional distribution array (NS) and the Energy
    % array (NE) have been converted to a geographic coordinate frame in which
    % direction is direction from.
    %
    %  Version: 1.1 - 5/2003,      Paul Jessen, NPS
    %  *** altered, 2/2005 ****    Jim Thomson, WHOI
    %
    % use convert to toggle between no rotation (0) and geographic convention (1)
    %
    %  First Version: 1.0 - 8/00
    %
    %  Latest Version: 1.1- 5/2003
    %
    %  Written by: Paul F. Jessen
    %              Department of Oceanography
    %              Naval Postgraduate School
    %              Monterey, CA
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % DESCRIPTION:
    %   Function calculates directional distribution of a wave field using the 
    %  Maximum Entropy Method of Lygre & Krogstad (JPO V16 1986). User passes the
    %  directional moments (a1,b1,a2,b2) and energy density (en) to the function.
    %  The directional moments are expected to be in a right hand coordinate 
    %  system (i.e. north, west) with direction being the direction towards.
    %  The returned energy and directional distributions have been converted to
    %  nautical convention with direction being the direction from.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % MODIFICATIONS
    %
    %  Version: 1.1 - Directional moments converted to a column vector at 
    %                 beginning of function (5/7/2003 - pfj)
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %

    %
    % calculate directional energy spectrum based on Maximum Entropy Method (MEM)
    % of Lygre & Krogstad, JPO V16 1986.
    """

    # switch to Krogstad notation
    d1 = np.asarray(a1)
    d2 = np.asarray(b1)
    d3 = np.asarray(a2)
    d4 = np.asarray(b2)
    en = np.asarray(en)
    c1 = d1 + d2 * 1j
    c2 = d3 + d4 * 1j
    p1 = (c1 - c2 * np.conj(c1)) / (1. - np.abs(c1)**2.)
    p2 = c2 - c1 * p1
    x1 = 1. - p1 * np.conj(c1) - p2 * np.conj(c2)

    # define directional domain, this is still in Datawell convention
    dtheta = 2.
    direc = np.arange(0., 360., dtheta)

    # get distribution with "dtheta" degree resolution (still in right hand system)
    dr = np.pi / 180.
    S = np.zeros((len(en), len(direc)), dtype=np.complex128)
    for n in range(len(direc)):
        alpha = direc[n] * dr
        e1 = np.cos(alpha) - np.sin(alpha) * 1j
        e2 = np.cos(2. * alpha) - np.sin(2. * alpha) * 1j
        y1 = np.abs(1. - p1 * e1 - p2 * e2)**2.

        #S(:, n) is the directional distribution across all frequencies (:) and directions (n).
        S[:, n] = (x1 / y1)
    S = np.real(S)

    # normalize each frequency band by the total across all directions so that the integral of
    # S(theta:f) is 1. Sn is the normalized directional distribution
    Sn = np.zeros_like(S)
    tot = np.sum(S, axis=1) * dtheta * dr
    for ii in range(len(en)):  # each frequency
        Sn[ii, :] = S[ii, :] / tot[ii]

    # calculate energy density by multiplying the energies at each frequency by the normalized
    # directional distribution at that frequency
    E = np.zeros_like(Sn)
    for ii in range(len(en)):
        E[ii, :] = Sn[ii, :] * en[ii]

    if not convert:
        # leave distribution in cartesian (observed) form
    	NE = E
    	NS = Sn

    else:
        # convert to a geographic coordinate frame
        ndirec = np.abs(direc - 360.)
        # convert from direction towards to direction from
        ndirec = ndirec + 180.
        ndirec[ndirec >= 360.] -= 360.

        # the Energy and distribution (s) arrays now don't go from 0-360. They now goes from 180-5 and
        # then from 360-185. Create new Energy and distribution matrices that go from 0-360.
        NE = np.zeros_like(E)
        NS = np.zeros_like(Sn)
        for ii in range(len(direc)):
            ia = np.where(ndirec == direc[ii])[0]
            if ia.size > 0:
                NE[:, ii] = E[:, ia]
                NS[:, ii] = Sn[:, ia]
            else:
                print('\n !!! Error converting to geographic coordinate frame !!!')

    return NS, NE
