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

%% set up the example
examplenum = 2;
fixedtarget = false;
buoytarget = true;

%% optional NetCDF output (single file for all prediction windows)
write_netcdf = true;
ncfile = 'NextWave_predictions.nc';

makevideo = true;


%% prepare a video of the results (fine to skip this)
if makevideo
    if fixedtarget
        vidObj = VideoWriter(['NextWaveExample' num2str(examplenum) '_fixedtarget'] ,'MPEG-4');
    elseif buoytarget
        vidObj = VideoWriter(['NextWaveExample' num2str(examplenum) '_buoytarget'] ,'MPEG-4');
    end
    open(vidObj);
end

%% local geometry (user defined)
latorigin = 41.6878; % origin of local coordinate system (everything needs to be within a few 100 m)
lonorigin = -9.0545; % origin of local coordinate system (everything needs to be within a few 100 m)
rotation = 180;  % rotation of local coordinate system ** THIS MUST BE CONSISTENT WITH USAGE OF VELOCITY COMPONENTS **
% use rotation = 180 if using GenericCoordinateTransform.m from SWIFT codes repo

%% target location(s) for prediction: can be user defined, or a buoy from the array itself (for testing)

if fixedtarget
    xtarget = 250; % user defined target location for prediction in local coordinate system [meters]
    ytarget = 150; % user defined target location for prediction in local coordinate system [meters]
elseif buoytarget
    targetbuoyindex = 4;  % which buoy to target (best if down-wave of others)
end

% Optional: show measured vs reprojected z/u/v for one buoy in extra panels.
% Set [] to disable, or 1..4 for SWIFT22..SWIFT25.
reproj_swift_idx = 4;  % [];

%% load example "burst" of raw data from SBG Ellipse sesnor running at 5 Hz on each buoy
skipwarmup = 200; % number of samples to skip at the start of bursts (i.e., skipping AHRS initialization)
burstend = 2700;  % number of samples defining end of burst ... usually 2742, needs to be same for all buoys
nbuoys = 4;  % examples have 4 buoys available.  Usually use 3 and test prediction against 4th one. 

flist = dir(['./ExampleData' num2str(examplenum) '/SWIFT*.mat']); % use '12-Sep-2022 07:00:00', which is burst index 92 from 'SWIFT22_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG.mat'

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

    load(['./ExampleData' num2str(examplenum)  '/' flist(fi).name])
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

if fixedtarget
    ttarget = tin(:,1);  % create timestamps for the wave predictions with same sampling rate as input (for convenience)
elseif buoytarget
    xtarget = xin(:, targetbuoyindex);
    ytarget = yin(:, targetbuoyindex);
    ztarget = zin(:, targetbuoyindex); % groundtruth measurements from target buoy location (for testing)
    utarget = uin(:, targetbuoyindex); % groundtruth measurements from target buoy location (for testing)
    vtarget = vin(:, targetbuoyindex); % groundtruth measurements from target buoy location (for testing)
    ttarget = tin(:, targetbuoyindex); % groundtruth measurements from target buoy location (for testing)
    zin(:,targetbuoyindex) = []; % remove target buoy from alorithm input
    uin(:,targetbuoyindex) = []; % remove target buoy from alorithm input
    vin(:,targetbuoyindex) = []; % remove target buoy from alorithm input
    tin(:,targetbuoyindex) = []; % remove target buoy from alorithm input
    xin(:,targetbuoyindex) = []; % remove target buoy from alorithm input
    yin(:,targetbuoyindex) = []; % remove target buoy from alorithm input
end

% reverse the direction of the vertical displacement (becasue SBG sensor is mounted upside in SWIFT buoys)
zin = -zin;

fs = 1./mean(mean(diff(tin))); % raw data sampling rate (Hz)


%% load directional spectra for this example (actual processing using SBGwaves.m from "SWIFTcodes" repo)
% can be determined from single buoy or [better] an average of all buoys
% note that this background spectra should be in the nautical direction (i.e., direction FROM which waves are coming, not towards)

%exampleindex = 92; % example 1
%exampleindex = 10; % example 2 '08-Sep-2022 21:00:00': 

%load('/Volumes/Data/DigiFloat/DIGIFLOAT_Portugal/SWIFT22_DIGIFLOAT_fall2022_part1/SWIFT22_DIGIFLOAT_07Sep2022-04Oct2022_reprocessedSBG.mat')
%[Etheta theta E f dir spread spread2 spread2alt ] = SWIFTdirectionalspectra(SWIFT(exampleindex), true, true);
%wavespec.Etheta = Etheta; wavespec.theta = theta; wavespec.f = f; Hs = SWIFT(exampleindex).sigwaveheight; Dp = SWIFT(exampleindex).peakwavedirT; Tp = SWIFT(exampleindex).peakwaveperiod;
%save wavespec wavespec Hs Tp Dp

load(['./ExampleData' num2str(examplenum) '/wavespec.mat']);

Te = sum(wavespec.Etheta(:))./sum(sum(wavespec.Etheta,2) .* wavespec.f); % centroid wave period
ce = 9.8 * Te / (2 * 3.14); % phase speed at centroid wave period
NTe = 10; % number of wave periods to use in the input window (usually 10)

if write_netcdf
    if exist(ncfile, 'file') == 2
        delete(ncfile);
    end

    % Per-window metadata (unlimited window dimension)
    nccreate(ncfile, 'ti', 'Dimensions', {'window', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'input_t_start', 'Dimensions', {'window', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'input_t_end', 'Dimensions', {'window', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'leadtime', 'Dimensions', {'window', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'solve_time_s', 'Dimensions', {'window', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'xtarget', 'Dimensions', {'window', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'ytarget', 'Dimensions', {'window', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'window_start_idx', 'Dimensions', {'window', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'window_count', 'Dimensions', {'window', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');

    % Flattened prediction arrays across all windows (unlimited sample dimension)
    nccreate(ncfile, 'tpred_flat', 'Dimensions', {'sample', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'zout_flat', 'Dimensions', {'sample', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'uout_flat', 'Dimensions', {'sample', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');
    nccreate(ncfile, 'vout_flat', 'Dimensions', {'sample', Inf}, 'Datatype', 'double', 'Format', 'netcdf4');

    % Run-level metadata
    ncwriteatt(ncfile, '/', 'latorigin', latorigin);
    ncwriteatt(ncfile, '/', 'lonorigin', lonorigin);
    ncwriteatt(ncfile, '/', 'rotation', rotation);
    if fixedtarget
        ncwriteatt(ncfile, '/', 'xtarget_nominal', xtarget);
        ncwriteatt(ncfile, '/', 'ytarget_nominal', ytarget);
    else
        ncwriteatt(ncfile, '/', 'xtarget_nominal', NaN);
        ncwriteatt(ncfile, '/', 'ytarget_nominal', NaN);
        ncwriteatt(ncfile, '/', 'targetbuoyindex', targetbuoyindex);
    end
    ncwriteatt(ncfile, '/', 'NTe', NTe);
    ncwriteatt(ncfile, '/', 'fs_hz', fs);
    ncwriteatt(ncfile, '/', 'Te_s', Te);
    ncwriteatt(ncfile, '/', 'ce_mps', ce);

    window_idx = 0;
    sample_idx = 0;
end


%% run algorithm for a given target location
% step through temporal windows making predictions
% not the timestamps for prediction are not required to be same sampling rate as input timeseries 
% (though are set up that way for these examples)
for ti = 1:round(fs):length(tin) % for smooth results, increment the windows slowly (every 1 s, thus increment indexing by data sampling rate fs)

    inputwindow = ti + (1:(NTe * Te * fs)); % indices for the samples to be used

    if ~any(inputwindow > length(tin))

        % output times and locations (should be within a few wave periods and wavelengths of input)
        maxtargetdistance = max( max( sqrt( (xin(inputwindow,:)-max(xtarget)).^2 + (yin(inputwindow,:)-max(ytarget)).^2 ) ) ); % how far is target location from buoy array
        leadtime = maxtargetdistance / ce; % how many seconds to predict ih the future (recommend as max distance between target and buoys, divided by phase speed)
        % any more than this and the information from the farthest buoy has already propagated beyond the target
        tlast = max( max(tin(inputwindow,:)) );  % last timestamp of the input window
        tpredindices = find( ttarget >= tlast & ttarget<=(tlast + leadtime ) );
        if fixedtarget && ~isempty(tpredindices)
            tpred = ttarget(tpredindices); % timestamps for the prediction ** these could be user defined instead **
            xpred = xtarget * ones(size(tpred));  % size is [times, locations] just like input
            ypred = ytarget * ones(size(tpred));  % size is [times, locations] just like input
        elseif buoytarget && ~isempty(tpredindices)
            tpred = ttarget( tpredindices ); % timestamps for the prediction ** these could be user defined instead **
            xpred = xtarget( tpredindices )';
            ypred = ytarget( tpredindices )';
        else
            tpred = NaN; % timestamps for the prediction ** these could be user defined instead **
            xpred = Nan(size(tpred));
            ypred = Nan(size(tpred));
        end

        [prediction, reconstruction, params, t] = leastSquaresWavePropagation(zin(inputwindow,:), uin(inputwindow,:), vin(inputwindow,:), ...
            tin(inputwindow,:), xin(inputwindow,:), yin(inputwindow,:), tpred, xpred, ypred, wavespec);

        prediction = reshape(prediction,length(tpred),[]);
        zout = prediction(:,1);
        uout = prediction(:,2);
        vout = prediction(:,3);

        reconstruction = reshape(reconstruction,length(inputwindow),[]);
        zr = reconstruction(:,1:size(zin,2));
        ur = reconstruction(:,size(zin,2)+(1:size(zin,2)));
        vr = reconstruction(:,2*size(zin,2)+(1:size(zin,2)));

        % option to rerun for a different ouput location using same solution
        %prediction = reprocess_LS_predictions(xpred,ypred,tpred,params)

        % plot the results
        figure(1), clf

        % map (top 3 rows, middle column)
        subplot(6,3,[2 5 8])
        plot(xin(inputwindow,:), yin(inputwindow,:),'x','linewidth',2), hold on  % input positions
        if buoytarget
            plot(xtarget(inputwindow), ytarget(inputwindow),'x','linewidth',2)
        end
        plot(xpred,ypred,'ko','linewidth',2,'markersize',8)  % output (target) positions
        axis([ (min(xin(:))-200) (max(xin(:))+200) (min(yin(:))-200) (max(yin(:))+200) ])
        xlabel('x [m]'), ylabel('y [m]'), grid, axis equal
        if exist('Dp', 'var')
            quiver(-150,50,-sind(Dp),-cosd(Dp),100,'filled','LineWidth',1,'color',[0 0 0])
        end

        % incident spectra (top-right column)
        if exist('Hs', 'var') && exist('Dp', 'var')
            subplot(6,3,[3 6 9])
            polarPcolor(wavespec.f', wavespec.theta, log10(wavespec.Etheta'))
            title(['Hs = ' num2str(Hs,2) ' m, Dp (FROM) = ' num2str(Dp,3) 'deg'])
        end

        % input (left column)
        subplot(6,3,1),  plot(tin(inputwindow,:),zin(inputwindow,:)), ylabel('z in [m]')
        subplot(6,3,4),  plot(tin(inputwindow,:),uin(inputwindow,:)), ylabel('u in [m/s]')
        subplot(6,3,7),  plot(tin(inputwindow,:),vin(inputwindow,:)), ylabel('v in [m/s]')

        % reconstruction (left column)
        subplot(6,3,10), plot(tin(inputwindow,:),zr), ylabel('z out [m]')
        subplot(6,3,13), plot(tin(inputwindow,:),ur), ylabel('u out [m/s]')
        subplot(6,3,16), plot(tin(inputwindow,:),vr), ylabel('v out [m/s]')
        xlabel('t [s]')

        % predictions (middle column)
        subplot(6,3,11), plot(tpred,zout,'k'), hold on, ylabel('z_p [m]')
        if buoytarget
            plot(tpred, ztarget(tpredindices),'k--','linewidth',2)
        end
        subplot(6,3,14), plot(tpred,uout,'k'), hold on, ylabel('u_p [m/s]')
        if buoytarget
            plot(tpred, utarget(tpredindices),'k--','linewidth',2)
        end
        subplot(6,3,17), plot(tpred,vout,'k'), hold on, ylabel('v_p [m/s]')
        if buoytarget
            plot(tpred, vtarget(tpredindices),'k--','linewidth',2)
        end
        xlabel('t [s]')

        % Optional measured vs reprojected (right column, bottom 3 rows)
        if ~isempty(reproj_swift_idx) && reproj_swift_idx >= 1 && reproj_swift_idx <= nbuoys
            j = reproj_swift_idx;
            subplot(6,3,12)
            plot(tin(inputwindow,j), zin(inputwindow,j), 'k'), hold on
            plot(tin(inputwindow,j), zr(:,j), '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2)
            ylabel('z [m]')
            title(sprintf('swift%d: measured vs reprojected', 21 + j))
            legend('meas','reproj','Location','best')

            subplot(6,3,15)
            plot(tin(inputwindow,j), uin(inputwindow,j), 'k'), hold on
            plot(tin(inputwindow,j), ur(:,j), '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2)
            ylabel('u [m/s]')

            subplot(6,3,18)
            plot(tin(inputwindow,j), vin(inputwindow,j), 'k'), hold on
            plot(tin(inputwindow,j), vr(:,j), '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2)
            ylabel('v [m/s]')
            xlabel('t [s]')
        end

        if write_netcdf
            window_idx = window_idx + 1;
            nsamp = numel(tpred);
            start_idx = sample_idx + 1;

            % Per-window metadata
            ncwrite(ncfile, 'ti', double(ti), window_idx);
            input_t_window = tin(inputwindow,:);
            ncwrite(ncfile, 'input_t_start', double(min(input_t_window(:))), window_idx);
            ncwrite(ncfile, 'input_t_end', double(max(input_t_window(:))), window_idx);
            ncwrite(ncfile, 'leadtime', double(leadtime), window_idx);
            ncwrite(ncfile, 'solve_time_s', double(t), window_idx);
            ncwrite(ncfile, 'xtarget', double(mean(xpred(:))), window_idx);
            ncwrite(ncfile, 'ytarget', double(mean(ypred(:))), window_idx);
            ncwrite(ncfile, 'window_start_idx', double(start_idx), window_idx);
            ncwrite(ncfile, 'window_count', double(nsamp), window_idx);

            % Flattened per-sample predictions for this window
            ncwrite(ncfile, 'tpred_flat', double(tpred(:)), start_idx);
            ncwrite(ncfile, 'zout_flat', double(zout(:)), start_idx);
            ncwrite(ncfile, 'uout_flat', double(uout(:)), start_idx);
            ncwrite(ncfile, 'vout_flat', double(vout(:)), start_idx);

            sample_idx = sample_idx + nsamp;
        end

        if makevideo
            currFrame = getframe(gcf);
            writeVideo(vidObj,currFrame);
        end

    end
end

if makevideo
    close(vidObj);
end

if write_netcdf
    fprintf('Wrote predictions to %s\n', ncfile);
end




