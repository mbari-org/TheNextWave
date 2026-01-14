function [prediction, reconstruction, params, t] = leastSquaresWavePropagation(z_in,u_in,v_in,t_in,x_in,y_in,t_pred,x_pred,y_pred,wavespec)
% This function uses time series of vertical displacement (and horizontal velocities) to
% generate a phase-resolved prediction of sea surface elevation (and horizontal velocities) 
% at a specified time & location using an inverse linear model.  
% The solution is constrained by the directional spectra E(f,theta)
%
% INPUT
%   z_in:  vertical displacement time series where M is the length of record and 
%          P is the number of point measurements.
%   u_in,v_in:   measurements of east, north velocities at sea surface (same size as z_in) [m/s]
%           ** the velocities are optional and generally improve prediction accuracy **
%   t_in:  timestamps of input measurements [seconds]
%   x_in,y_in:   positions (easting, northing) of measurement locations [meters]
%   wavespec: data structure containing the following fields
%          Etheta - measured directional wave spectrum (Nautical convention)
%          f - vector of wave frequencies [Hz]
%          theta - vector of wave directions [degrees from North]
%        
%   t_pred:   timestamps for the prediction [seconds]
%   x_pred,y_pred:  position (easting, northing) of target location for prediction [meters]
%
% OUTPUT
%   prediction: predicted z,u,v at location(s) x_pred, y_pred during t_pred
%       if velocities are empty inputs, velocities will not be predicted
%   reconstruction: reconstructed z,u,v motions for the input locations x_in, y_in during t_in
%       if velocities are empty inputs, velocities will not be reconstructed
%   parameters: details of the solution (amplitudes and phases)
%   t: time for computation 
%
% A. Fisher and J. Thomson, 2019-2025

if ~isempty(u_in) & ~isempty(v_in)
    use_vel=true;
else
    use_vel=false;
end

tic;

%convert wave spectrum to Cartesian coordinates (e.g. direction waves are moving TOWARDS)
if size(wavespec.Etheta,1)==length(wavespec.theta)
wavespec.Etheta=wavespec.Etheta';
end

[wavespec.theta,I]=unique(wavespec.theta,'last');
wavespec.Etheta=wavespec.Etheta(:,I);
t=wavespec.theta+180;
t(t>360)=t(t>360)-360;
[~,I]=sort(t);
wavespec.Etheta=wavespec.Etheta(:,I);

[~,c]=find(wavespec.Etheta==max(wavespec.Etheta(:)),1,'first');
DTp=wavespec.theta(c).*pi./180;
df=gradient(wavespec.f(:));

% Limit solution space to frequencies that statisfy S(f)/max(S(f))>5 
% and directions that statisfy DTp-pi/2 < DTp < DTp+pi/2

wavespec.E=trapz(wavespec.theta,wavespec.Etheta');
frange=find((df.*wavespec.E(:))./max(df.*wavespec.E(:))>=0.05);
omega=logspace(log10(wavespec.f(frange(1))),log10(wavespec.f(frange(end))),40).*2.*pi;
k=omega.^2./9.81;

theta=linspace(DTp-pi/2,DTp+pi/2,25);
theta(theta>2*pi)=theta(theta>2*pi)-2*pi;
theta(theta<0)=theta(theta<0)+2*pi;
theta=sort(theta);

%Reshape input & check for consistency
k = k(:);
theta = theta(:);
kx = k*sin(theta');
ky = k*cos(theta');
omega = sqrt(9.81*k)*ones(size(theta'));
kx = kx(:);
ky = ky(:);
omega = omega(:);
x_in = x_in(:);
y_in = y_in(:);
t_in = t_in(:);
z_in = z_in(:);
u_in = u_in(:);
v_in = v_in(:);
x_pred = x_pred(:);
y_pred = y_pred(:);
t_pred = t_pred(:);

N_input_pts = length(z_in);
if length(x_in) ~= N_input_pts || length(y_in) ~= N_input_pts || length(t_in) ~= N_input_pts
    error('All input vectors must be equal length')
end

N_output_pts = length(t_pred);
if length(x_pred) ~= N_output_pts || length(y_pred) ~= N_output_pts
    error('All output vectors must be equal length')
end

%Interpolate Observed Spectrum to Solution Space
[F,T]=meshgrid(wavespec.f,wavespec.theta);
[f2,thet2]=meshgrid(sqrt(k.*9.8),theta);
Ei=10.^griddata(F,T,log10(wavespec.Etheta'),f2./(2*pi),thet2.*180./pi);
Ei(isnan(Ei))=0;

Ei=Ei.*trapz(wavespec.f,trapz(wavespec.theta,wavespec.Etheta'))./trapz(f2(1,:)./(2*pi),trapz(thet2(:,1).*180./pi,Ei));
amps=sqrt(Ei.*diff([0 f2(1,:)./(2*pi)]).*mode(diff(thet2(:,1).*180./pi)));
amps=amps';
amps=[amps(:);amps(:)];
amps(isnan(amps))=0;

%Construct Propagator Matrices
phi1=x_in*kx'+y_in*ky'-t_in*omega';
phi2=x_pred*kx'+y_pred*ky'-t_pred*omega';

%P1: Used to invert measured wave data (M1 x N)
%P2: Used to predict at target location/time (M2 x N)
%Note: P1 and P2 are consistent formulations, but M1 may be different than M2.

if use_vel
P1 = [[cos(phi1),sin(phi1)];...
    [(kx./sqrt(kx.^2+ky.^2))'.*omega'.*cos(phi1),(kx./sqrt(kx.^2+ky.^2))'.*omega'.*sin(phi1)];...
    [(ky./sqrt(kx.^2+ky.^2))'.*omega'.*cos(phi1),(ky./sqrt(kx.^2+ky.^2))'.*omega'.*sin(phi1)]];

P2 = [[cos(phi2),sin(phi2)];...
    [(kx./sqrt(kx.^2+ky.^2))'.*omega'.*cos(phi2),(kx./sqrt(kx.^2+ky.^2))'.*omega'.*sin(phi2)];...
    [(ky./sqrt(kx.^2+ky.^2))'.*omega'.*cos(phi2),(ky./sqrt(kx.^2+ky.^2))'.*omega'.*sin(phi2)]]; 
else
    P1 = [cos(phi1),sin(phi1)];
    P2 = [cos(phi2),sin(phi2)];
end

good=find(amps(1:numel(Ei))~=0);
P1(:,amps==0)=[];
P2(:,amps==0)=[];
amps(amps==0)=[];

%Invert linear model to solve for unknown wave amplitudes using a bounded least-squares approach.
options = optimoptions('lsqlin','Algorithm','trust-region-reflective'); % some matlab magic here
if use_vel
    A = lsqlin(P1,[z_in;u_in;v_in],[],[],[],[],-amps./1.4142,amps./1.4142,[]);
    reconstruction = P1*A;
    prediction = P2*A;
else
    A = lsqlin(P1,z_in,[],[],[],[],-amps./1.4142,amps./1.4142,[]);
    reconstruction = P1*A;
    prediction = P2*A;
end

t=toc;


%% bookkeeping
params.A=A;
params.Etheta=zeros(size(Ei(:)))';
params.Etheta(good)=(A(1:length(A)/2).^2+A(length(A)/2+1:end).^2)./2;
params.Etheta=reshape(params.Etheta,length(k),length(theta))'./(diff([0 f2(1,:)./(2*pi)]).*mode(diff(thet2(:,1).*180./pi)));
params.f=f2(1,:)'./(2.*pi);
params.theta=thet2(:,1)'.*180./pi;
params.theta=params.theta+180;
params.theta(params.theta>360)=params.theta(params.theta>360)-360;
[params.theta,I]=sort(params.theta);
params.Etheta=params.Etheta(I,:)';
params.kx=kx(good);
params.ky=ky(good);
params.omega=omega(good);
params.use_vel=use_vel;