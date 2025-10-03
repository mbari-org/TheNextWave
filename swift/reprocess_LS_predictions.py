function z=reprocess_LS_predictions(x,y,t,params)
z=nan(size(x));

for i=3:length(params)
if ~isempty(params(i).A)
%Construct Propagator Matrices
phi=x(i,:)'*params(i).kx'+y(i,:)'*params(i).ky'-t(i,:)'*params(i).omega';

%P1: Used to invert measured wave data (M1 x N)
%P2: Used to predict at target location/time (M2 x N)
%Note: P1 and P2 are consistent formulations, but M1 may be different than M2.

if params(i).use_vel
P = [[cos(phi),sin(phi)];...
    [(kx./sqrt(params(i).kx.^2+params(i).ky.^2))'.*params(i).omega'.*cos(phi),(params(i).kx./sqrt(params(i).kx.^2+params(i).ky.^2))'.*params(i).omega'.*sin(phi)];...
    [(ky./sqrt(params(i).kx.^2+params(i).ky.^2))'.*params(i).omega'.*cos(phi),(params(i).ky./sqrt(params(i).kx.^2+params(i).ky.^2))'.*params(i).omega'.*sin(phi)]]; 
else
P = [cos(phi),sin(phi)];
end

A=[params(i-2).A,params(i-1).A,params(i).A];
A=nanmean(A,2);
z(i,:) = P*A;
end
end