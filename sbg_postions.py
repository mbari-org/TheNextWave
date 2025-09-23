function SWIFT=sbg_postions(swift_no)
cd0=cd();
cd(['./SWIFT' num2str(swift_no) '_DIGIFLOAT_07Sep2022-04Oct2022/'])
file=dir('*reprocessedSBG_displacements.mat');
load(file.name)
for i=1:length(SWIFT)
load(['./SBG/Raw/' datestr(SWIFT(i).time,'yyyymmdd') '/SWIFT' num2str(swift_no) '_SBG_' datestr(SWIFT(i).time,'ddmmmyyyy') '_' datestr(SWIFT(i).time,'HH') '_' num2str(strread(datestr(SWIFT(i).time,'MM'))/12+1,'%02.f') '.mat'])
sbg = sbgData;
sbg_time=datenum(sbg.UtcTime.year,sbg.UtcTime.month,sbg.UtcTime.day,sbg.UtcTime.hour,sbg.UtcTime.min,sbg.UtcTime.sec+sbg.UtcTime.nanosec./1E9);
% Transform lat/lon into UTM x/y
medLat(i) = median(sbg.GpsPos.lat);
medLon(i) = median(sbg.GpsPos.long);
zone = utmzone([medLat(i),medLon(i)]);
[ellipsoid,estr] = utmgeoid(zone);
utmstruct = defaultm('utm');
utmstruct.zone = zone;
utmstruct.geoid = ellipsoid(1,:);
utmstruct = defaultm(utmstruct);
[SWIFT(i).sbg_x,SWIFT(i).sbg_y] = mfwdtran(utmstruct,sbg.GpsPos.lat,sbg.GpsPos.long);
good=find(sbg_time>datenum(2022,9,7));
sbg_time=sbg_time(good);
[sbg_time,id]=unique(sbg_time);
good=good(id);
[~,gps_good]=unique(sbg.GpsPos.time_stamp);
SWIFT(i).sbg_x=interp1(sbg.GpsPos.time_stamp(gps_good),SWIFT(i).sbg_x(gps_good),sbg.UtcTime.time_stamp(good));
SWIFT(i).sbg_y=interp1(sbg.GpsPos.time_stamp(gps_good),SWIFT(i).sbg_y(gps_good),sbg.UtcTime.time_stamp(good));
SWIFT(i).sbg_lat=interp1(sbg.GpsPos.time_stamp(gps_good),sbg.GpsPos.lat(gps_good),sbg.UtcTime.time_stamp(good));
SWIFT(i).sbg_lon=interp1(sbg.GpsPos.time_stamp(gps_good),sbg.GpsPos.long(gps_good),sbg.UtcTime.time_stamp(good));

[t,id]=unique(SWIFT(i).rawtime);
bad=find(diff(t).*24>1);
SWIFT(i).u=SWIFT(i).u(id);
SWIFT(i).v=SWIFT(i).v(id);
SWIFT(i).rawtime=SWIFT(i).rawtime(id);
SWIFT(i).z=SWIFT(i).z(id);
SWIFT(i).u(1:bad)=[];
SWIFT(i).v(1:bad)=[];
SWIFT(i).z(1:bad)=[];
SWIFT(i).rawtime(1:bad)=[];
SWIFT(i).sbg_x=interp1(sbg_time,SWIFT(i).sbg_x,SWIFT(i).rawtime);
SWIFT(i).sbg_y=interp1(sbg_time,SWIFT(i).sbg_y,SWIFT(i).rawtime);
SWIFT(i).sbg_lon=interp1(sbg_time,SWIFT(i).sbg_lon,SWIFT(i).rawtime);
SWIFT(i).sbg_lat=interp1(sbg_time,SWIFT(i).sbg_lat,SWIFT(i).rawtime);
end
save(file.name,'SWIFT')
cd(cd0)
end