% script to run example data for "NextWave" algorithm
% example data are raw (5 Hz) motion data from four SWIFT buoys
% the data are from a single "burst" (=512 seconds)
%
% the target location (x_out, y_out) for the prediction is arbitrary and
% can be changed by the user, but should be with in the range [0,500] for
% this example
%
% J. Thomson, 1/2026

clear

%% prepare a video of the results (fine to skip this)
makevideo = false;
if makevideo
    vidObj = VideoWriter('NextWaveExample','MPEG-4');
    open(vidObj);
end

%% local geometry
latorigin = 41.6878; % origin of local coordinate system (everything needs to be within a few 100 m)
lonorigin = -9.0545; % origin of local coordinate system (everything needs to be within a few 100 m)
rotation = 180;  % rotation of local coordinate system ** THIS MUST BE CONSISTENT WITH USAGE OF VELOCITY COMPONENTS **
% use rotation = 180 if using GenericCoordinateTransform.m from SWIFT codes repo

xtarget = 200; % user determined target location for prediction in local coordinate system [meters]
ytarget = 200; % user determined target location for prediction in local coordinate system [meters]


%% load example "burst" of raw data from SBG Ellipse sesnor running at 5 Hz on each buoy
skipwarmup = 200; % number of samples to skip at the start of bursts (i.e., skipping AHRS initialization)
burstend = 2740;  % number of samples defining end of burst ... usually 2742, needs to be same for all buoys
nbuoys = 4;

flist = dir('./ExampleData/SWIFT*_SBG_12Sep2022_07_01.mat'); % use '12-Sep-2022 07:00:00', which is burst index 92 from 'SWIFT22_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG.mat'

% preallocate input arrays
zin = NaN( length(skipwarmup:burstend), nbuoys);
uin = NaN( length(skipwarmup:burstend), nbuoys);
vin = NaN( length(skipwarmup:burstend), nbuoys);
tin = NaN( length(skipwarmup:burstend), nbuoys);
xin = NaN( length(skipwarmup:burstend), nbuoys);
yin = NaN( length(skipwarmup:burstend), nbuoys);

% loop through files (one for each buoy) and populate input array
% this has fragile indexing (assuming all data same size)
% this also assumes buoy system clocks are sync'd (and thus relative seconds since start of burst are consistent)
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

    % pick a time reference (could improve by interpolating everything to common timestamp first, but probably only a few ms)
    tin(:,fi) = ztime;

    % map everything to a local coordinate system (in meters)
    [ x, y ] = GenericCoordinateTransform(lat, lon, latorigin, lonorigin, rotation);  % function from SWIFT codes repo

    xin(:,fi) = x;
    yin(:,fi) = y;

end

% reverse the direction of the vertical displacement (becasue SBG sensor is mounted upside in SWIFT buoys)
zin = -zin;

fs = 1./mean(mean(diff(tin))); % raw data sampling rate (Hz)


%% load directional spectra for this example (actual processing using SBGwaves.m from "SWIFTcodes" repo)
% can be determined from single buoy or [better] an average of all buoys
%
% this example preprocessed from:
%load('SWIFT22_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG.mat')
%[Etheta theta E f dir spread spread2 spread2alt ] = SWIFTdirectionalspectra(SWIFT(92), true, true);
%wavespec.Etheta = Etheta; wavespec.theta = theta; wavespec.f = f;
%save wavespec wavespec

load ./ExampleData/wavespec.mat

Te = sum(wavespec.Etheta(:))./sum(sum(wavespec.Etheta,2) .* wavespec.f); % centroid wave period
ce = 9.8 * Te / (2 * 3.14); % phase speed at centroid wave period


%% run algorithm for a given output location
% step through temporal windows making predictions
NTe = 10; % number of wave periods to use in the input window (usually 10)

for ti = 1:fs:length(tin) % for smooth results, increment the windows slowly (every 1 s, thus increment indexing by data sampling rate fs)

    inputwindow = ti + (1:(NTe * Te * fs)); % indices for the samples to be used

    if ~any(inputwindow > length(tin))

        % output times and locations (should be within a few wave periods and wavelengths of input)
        maxtargetdistance = max( max( sqrt( (xin(inputwindow,:)-xtarget).^2 + (yin(inputwindow,:)-ytarget).^2 ) ) ); % how far is target location from buoy array
        leadtime = maxtargetdistance / ce; % how far to predict ih the future (recommend as max distance between target and buoys, divided by phase speed)
        % any more than this and the information from the farthest buoy has already propagated beyond the target
        tpred = max(max(tin(inputwindow,:))) + (1:leadtime);  % can be same as input inputwindow, or can extend abit into future or past
        xpred = xtarget * ones(size(tpred));  % size is [times, locations] just like input
        ypred = ytarget * ones(size(tpred));  % size is [times, locations] just like input

        [prediction, reconstruction, params, t] = leastSquaresWavePropagation(zin(inputwindow,:), uin(inputwindow,:), vin(inputwindow,:), ...
            tin(inputwindow,:), xin(inputwindow,:), yin(inputwindow,:), tpred, xpred, ypred, wavespec);

        prediction = reshape(prediction,length(tpred),[]);
        zout = prediction(:,1);
        uout = prediction(:,2);
        vout = prediction(:,3);

        reconstruction = reshape(reconstruction,length(inputwindow),[]);
        zr = reconstruction(:,1:nbuoys);
        ur = reconstruction(:,nbuoys+(1:nbuoys));
        vr = reconstruction(:,2*nbuoys+(1:nbuoys));

        % option to rerun for a different ouput location using same solution
        %prediction = reprocess_LS_predictions(xpred,ypred,tpred,params)

        % plot the results
        figure(1), clf

        % map
        subplot(2,2,2)
        plot(xin(inputwindow,:), yin(inputwindow,:),'x','linewidth',2), hold on  % input positions
        plot(xpred,ypred,'ko','linewidth',2,'markersize',8)  % output (target) positions
        axis([0 500 0 500]), xlabel('x [m]'), ylabel('y [m]'), grid, axis equal

        % input
        subplot(6,2,1),  plot(tin(inputwindow,:),zin(inputwindow,:)), ylabel('z in [m/s]')
        subplot(6,2,3),  plot(tin(inputwindow,:),uin(inputwindow,:)), ylabel('u in [m/s]')
        subplot(6,2,5), plot(tin(inputwindow,:),vin(inputwindow,:)), ylabel('v in [m/s]')

        % reconstruction
        subplot(6,2,7),  plot(tin(inputwindow,:),zr), ylabel('z out [m/s]')
        subplot(6,2,9),  plot(tin(inputwindow,:),ur), ylabel('u out [m/s]')
        subplot(6,2,11), plot(tin(inputwindow,:),vr), ylabel('v out [m/s]')
        xlabel('t [s]')

        % predictions
        subplot(6,2,8), plot(tpred,zout,'k'), ylabel('z_p [m]')
        subplot(6,2,10), plot(tpred,uout,'k'), ylabel('u_p [m/s]')
        subplot(6,2,12), plot(tpred,vout,'k'), ylabel('v_p [m/s]'),
        xlabel('t [s]')

        if makevideo
            currFrame = getframe(gcf);
            writeVideo(vidObj,currFrame);
        end

    end
end

if makevideo
    close(vidObj);
end




