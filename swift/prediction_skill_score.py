function P=prediction_skill_score(array,prediction)
for i=1:50
[Etheta(:,:,1),wavespec.theta,E(:,1),wavespec.f,~,spread(:,1),spread2(:,1)]=SWIFTdirectionalspectra(array.swift22,false,true);
[Etheta(:,:,2),~,E(:,2),~,~,spread(:,2),spread2(:,2)]=SWIFTdirectionalspectra(array.swift23,false,true);
[Etheta(:,:,3),~,E(:,3),~,~,spread(:,3),spread2(:,3)]=SWIFTdirectionalspectra(array.swift24,false,true);
[Etheta(:,:,4),~,E(:,4),~,~,spread(:,4),spread2(:,4)]=SWIFTdirectionalspectra(array.swift25,false,true);
wavespec.Etheta=mean(Etheta,3);
wavespec.spread=mean(spread,2);
wavespec.spread2=mean(spread2,2);
E=mean(E,2);


theta=90-wavespec.theta;
theta(theta<360)=theta(theta<360)+360;
freq=wavespec.f;
EfthetaSwift=wavespec.Etheta;
theta(theta > 180) = theta(theta > 180) - 360;
[theta,I]=unique(theta);
EfthetaSwift=EfthetaSwift(:,I);
[theta, dsort] = sort(theta);
EfthetaSwift = EfthetaSwift(:,dsort);
indNan = sum(isnan(EfthetaSwift),2)>0;
        
        % Transform into WAFO spec struct format
        S = struct();
        S.date = prediction.tp;
        S.type = 'dir';
        S.S = EfthetaSwift(~indNan,:)'*180/pi;
        S.f = freq(~indNan);
        S.theta = theta'*pi/180;
        S.phi = 0;

%         P=spec2sdat(S,length(swift_array.t),0.2);
dx = 0.5;
dt = 0.2;
L = 1;
T = length(prediction.tp).*0.2;
Nx = 1;
Ny = 1;
Nt = round(T/dt);
x = linspace(0,L-dx,Nx); % m
y = linspace(0,L,Ny); % m
t = linspace(0,T-dt,Nt); % s
dy = dx;

% Call WAFO codes to simulate surface trace
% if fftdim = 2, 2D ifft at each time.  if fftdim = 1, 1D ifft at each space
fftdim =1;
[Y,~] = seasim(S,Nx,Ny,Nt,dx,dy,dt,fftdim,0);
P(:,i)=Y.Z;
% end
%         Gts = spec2sdat(S,[2047 50],0.2);
%         P(j).zp=P(j).zp(:);
%         P(j).zt=P(j).zt(:);
%         mse_gaussian=nanmean((Gts(P(j).zp~=0,2:end)-P(j).zt(P(j).zp~=0)).^2);
%         mse_lse=nanmean((P(j).zp(P(j).zp~=0)-P(j).zt(P(j).zp~=0)).^2);
%         stats(i).skillscore(cnt)=nanmedian(1-mse_lse./mse_gaussian);
%         stats(i).null(cnt).Gts=Gts(:,2:end);
%         stats(i).null(cnt).t=Gts(:,1);
%         stats(i).wave_series(cnt)=ws(I);
%         stats(i).num_buoys(cnt)=size(P(j).tk,2);
% end
end
