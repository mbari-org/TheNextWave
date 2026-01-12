% script to run example data for "NextWave" algorithm
% example data are raw (5 Hz) motion data from four SWIFT buoys 
% the data are from a single "burst" (=512 seconds)
% 
% J. Thomson, 1/2026

clear all, close all


%% fixed variables
skipwarmup = 200; % number of samples to skip at the start of bursts (i.e., skipping AHRS initialization)
burstend = 2740;  % number of samples defining end of burst ... usually 2742, needs to be same for all buoys
nbuoys = 4;

latorigin = 41.6878; % origin of local coordinate system (everything needs to be within a few 100 m)
lonorigin = -9.0545; % origin of local coordinate system (everything needs to be within a few 100 m)
rotation = 180;  % rotation of local coordinate system ** THIS MUST BE CONSISTENT WITH USAGE OF VELOCITY COMPONENTS **
    % use rotation = 180 in 


%% load and format raw data from SBG Ellipse sesnor running at 5 Hz
% this has fragile indexing (assuming all data same size)
% this also also buoy system clocks are sync'd (and thus relative seconds since start of burst are consistent)

flist = dir('./ExampleData/SWIFT*_SBG_12Sep2022_07_01.mat'); % use '12-Sep-2022 07:00:00', which is burst index 92 from 'SWIFT22_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG.mat'

% preallocate 
zin = NaN( length(skipwarmup:burstend), nbuoys);
uin = NaN( length(skipwarmup:burstend), nbuoys);
vin = NaN( length(skipwarmup:burstend), nbuoys);
tin = NaN( length(skipwarmup:burstend), nbuoys);
xin = NaN( length(skipwarmup:burstend), nbuoys);
yin = NaN( length(skipwarmup:burstend), nbuoys);

for fi=1:nbuoys

    load(['./ExampleData/' flist(fi).name])
    zin(:,fi) = sbgData.ShipMotion.heave(skipwarmup:burstend)'; % vertical displacement used to invert for wave propagation
    ztime = sbgData.ShipMotion.time_stamp(skipwarmup:burstend)'./1e6; % time since burst started (microseconds --> seconds)
    uin(:,fi) = sbgData.GpsVel.vel_e(skipwarmup:burstend)'; % lateral velocity used to invert for wave propagation
    vin(:,fi) = sbgData.GpsVel.vel_n(skipwarmup:burstend)'; % lateral velocity used to invert for wave propagation
    uvtime = sbgData.GpsVel.time_stamp(skipwarmup:burstend)'./1e6; % time since burst started (microseconds --> seconds)
    lat = sbgData.GpsPos.lat(skipwarmup:burstend)'; % position of measurement
    lon = sbgData.GpsPos.long(skipwarmup:burstend)'; % position of measurement
    lltime = sbgData.GpsPos.time_stamp(skipwarmup:burstend)'./1e6; % time since burst started (microseconds --> seconds)

    % pick a time reference (could improve by interpolating everything to common timestamp first)
    tin(:,fi) = ztime;  

    % map everything to a local coordinate system (in meters)
    [ x, y ] = GenericCoordinateTransform(lat, lon, latorigin, lonorigin, rotation);  % function from SWIFT codes repo

    xin(:,fi) = x;
    yin(:,fi) = y;

end

% reverse the direction of the vertical displacement (becasue SBG sensor is mounted upside in SWIFT buoys)
zin = -zin;

%plot(xin, yin,'x')  % check positions


%% load directional spectra from previous burst (actual processing using SBGwaves.m from "SWIFTcodes" repo)
% can be determined from single buoy or [better] an average of multiple buoys

% load('SWIFT22_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG.mat')
% [Etheta theta E f dir spread spread2 spread2alt ] = SWIFTdirectionalspectra(SWIFT(92), true, true);
% wavespec.Etheta = Etheta; wavespec.theta = theta; wavespec.f = f;
% save wavespec wavespec

load ./ExampleData/wavespec.mat


%% run algorithm for a given output location

% step through temporal windows for prediction 
% (usually a few wave periods... see run_LS_prediction_SWIFTs.m)
twindow = 100:200; % index the 5 Hz samples, not the seconds

% output times and locations (should be within a few wave periods and wavelengths of input)
tout = twindow + 10;  % can be same as input twindow, or can extend abit into future or past
xout = 200 * ones(size(twindow));  % size is [times, locations] just like input
yout = 200 * ones(size(twindow));  % size is [times, locations] just like input

[zout, zc, params, t] = leastSquaresWavePropagation(zin(twindow,:), uin(twindow,:), vin(twindow,:), ... 
    tin(twindow,:), xin(twindow,:), yin(twindow,:), tout, xout, yout, wavespec);

%plot(tout, zout, tout, zc)

%% option to rerun for a different ouput location using same solution 
% (no need to re-solve if temporal window is same)

% z = reprocess_LS_predictions(xout,yout,tout,params)