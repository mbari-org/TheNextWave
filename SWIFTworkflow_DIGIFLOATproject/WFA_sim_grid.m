function [x,y,lon,lat,x0,y0,lon0,lat0]=WFA_sim_grid
% function to build the spatial grid used for prediction at Wind Floats
% (specific to Digifloat project)
lat0=41+(41+11.704/60)/60;
lon0=-(9+(3+0.777/60)/60);

[x0,y0,z]=deg2utm(lat0,lon0);

x=x0+(3:3:50)'.*cosd(0:5:360);
y=y0+(3:3:50)'.*sind(0:5:360);

[lat,lon]=utm2deg(x(:),y(:),repmat(z,numel(x),1));
lon=reshape(lon,size(x));
lat=reshape(lat,size(x));
end