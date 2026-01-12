function [array,prediction]=run_LS_prediction_SWIFTS(array,data_deny)

N=9;
Nf=1;

[Etheta(:,:,1),wavespec.theta,E(:,1),wavespec.f,~,spread(:,1),spread2(:,1)]=SWIFTdirectionalspectra(array.swift22,false,true);
[Etheta(:,:,2),~,E(:,2),~,~,spread(:,2),spread2(:,2)]=SWIFTdirectionalspectra(array.swift23,false,true);
[Etheta(:,:,3),~,E(:,3),~,~,spread(:,3),spread2(:,3)]=SWIFTdirectionalspectra(array.swift24,false,true);
[Etheta(:,:,4),~,E(:,4),~,~,spread(:,4),spread2(:,4)]=SWIFTdirectionalspectra(array.swift25,false,true);
wavespec.Etheta=mean(Etheta,3);
wavespec.spread=mean(spread,2);
wavespec.spread2=mean(spread2,2);
E=mean(E,2);
TM0=trapz(wavespec.f,E)./trapz(wavespec.f,E.*wavespec.f);
[~,k,s]=disper(95,TM0);

t0=min([array.swift22.rawtime(1),array.swift23.rawtime(1),array.swift24.rawtime(1),array.swift25.rawtime(1)]);
x0=mean([array.swift22.sbg_x(1),array.swift23.sbg_x(1),array.swift24.sbg_x(1),array.swift25.sbg_x(1)]);
y0=mean([array.swift22.sbg_y(1),array.swift23.sbg_y(1),array.swift24.sbg_y(1),array.swift25.sbg_y(1)]);
a1=mean([array.swift22.wavespectra.a1,array.swift23.wavespectra.a1,array.swift24.wavespectra.a1,array.swift25.wavespectra.a1],2);
b1=mean([array.swift22.wavespectra.b1,array.swift23.wavespectra.b1,array.swift24.wavespectra.b1,array.swift25.wavespectra.b1],2);
a1=trapz(wavespec.f,E.*a1)./trapz(wavespec.f,E);
b1=trapz(wavespec.f,E.*b1)./trapz(wavespec.f,E);
dmo=atan2d(a1,b1);
dmo(dmo<0)=dmo(dmo<0)+360;

prediction.tp=(array.swift25.rawtime'-t0).*24.*3600;
if data_deny
prediction.tm=([array.swift22.rawtime',array.swift23.rawtime',array.swift24.rawtime']-t0).*24.*3600;
prediction.zm=nan(size(prediction.tm));
prediction.zc=nan(size(prediction.tm));
prediction.um=nan(size(prediction.tm));
prediction.vm=nan(size(prediction.tm));
prediction.uc=nan(size(prediction.tm));
prediction.vc=nan(size(prediction.tm));
prediction.zp=nan(size(prediction.tp));
prediction.up=nan(size(prediction.tp));
prediction.vp=nan(size(prediction.tp));
prediction.zt=nan(size(prediction.tp));
prediction.ut=nan(size(prediction.tp));
prediction.vt=nan(size(prediction.tp));

X=sqrt((nanmean(array.swift25.sbg_x)-nanmean(array.swift24.sbg_x)).^2+(nanmean(array.swift25.sbg_y)-nanmean(array.swift24.sbg_y)).^2);
Theta=289; %approximate bearing between SWIFT 24 & SWIFT 25
Cp=s./k;
% Nlead=round(0.5.*X.*cosd(Theta-dmo)./Cp);
Nlead=5;
else
[WFA1.x,WFA1.y,WFA1.lon,WFA1.lat,WFA1.x0,WFA1.y0]=WFA_sim_grid;

prediction.tm=([array.swift22.rawtime',array.swift23.rawtime',array.swift24.rawtime',array.swift25.rawtime']-t0).*24.*3600;
prediction.zm=nan(size(prediction.tm));
prediction.zc=nan(size(prediction.tm));
prediction.um=nan(size(prediction.tm));
prediction.vm=nan(size(prediction.tm));
prediction.uc=nan(size(prediction.tm));
prediction.vc=nan(size(prediction.tm));
prediction.zp=nan(length(prediction.tp),size(WFA1.x,1),size(WFA1.x,2));
Theta=335;
X=sqrt((nanmean(WFA1.x(:))-nanmean(array.swift25.sbg_x)).^2+(nanmean(WFA1.y(:))-nanmean(array.swift25.sbg_y)).^2);
Nlead=round(0.5.*X.*cosd(Theta-dmo)./Cp);
end


for n=round(N*TM0*5):Nf*5:length(prediction.tp)-Nf*5-Nlead*5 
subsample=n-round(N*TM0*5-1):n;
target_samp=(n+1:n+Nf*5)+Nlead*5;
if data_deny
zt=-array.swift25.z(target_samp);
ut=array.swift25.u(target_samp);
vt=array.swift25.v(target_samp);
xt=array.swift25.sbg_x(target_samp)-x0;
yt=array.swift25.sbg_y(target_samp)-y0;
tp=(array.swift25.rawtime(target_samp)'-t0).*24.*3600;

zk=-[array.swift22.z(subsample)',array.swift23.z(subsample)',array.swift24.z(subsample)'];
xk=[array.swift22.sbg_x(subsample)',array.swift23.sbg_x(subsample)',array.swift24.sbg_x(subsample)']-x0;
yk=[array.swift22.sbg_y(subsample)',array.swift23.sbg_y(subsample)',array.swift24.sbg_y(subsample)']-y0;
uk=[detrend(filtdat(array.swift22.u(subsample)',5,1/2,'low')),detrend(filtdat(array.swift23.u(subsample)',5,1/2,'low')),detrend(filtdat(array.swift24.u(subsample)',5,1/2,'low'))];
vk=[detrend(filtdat(array.swift22.v(subsample)',5,1/2,'low')),detrend(filtdat(array.swift23.v(subsample)',5,1/2,'low')),detrend(filtdat(array.swift24.v(subsample)',5,1/2,'low'))];
tk=([array.swift22.rawtime(subsample)',array.swift23.rawtime(subsample)',array.swift24.rawtime(subsample)']-t0).*24.*3600;

else
xt=repmat(WFA1.x(:)',length(target_samp),1)-x0;
yt=repmat(WFA1.y(:)',length(target_samp),1)-y0;
tp=(array.swift25.rawtime(target_samp)'-t0).*24.*3600;
tp=repmat(tp,1,numel(WFA1.x));
tp=tp(:);
xt=xt(:);
yt=yt(:);

zk=-[array.swift22.z(subsample)',array.swift23.z(subsample)',array.swift24.z(subsample)',array.swift25.z(subsample)'];
xk=[array.swift22.sbg_x(subsample)',array.swift23.sbg_x(subsample)',array.swift24.sbg_x(subsample)',array.swift25.sbg_x(subsample)']-x0;
yk=[array.swift22.sbg_y(subsample)',array.swift23.sbg_y(subsample)',array.swift24.sbg_y(subsample)',array.swift25.sbg_y(subsample)']-x0;
uk=[detrend(filtdat(array.swift22.u(subsample)',5,1/2,'low')),detrend(filtdat(array.swift23.u(subsample)',5,1/2,'low')),detrend(filtdat(array.swift24.u(subsample)',5,1/2,'low')),detrend(filtdat(array.swift25.u(subsample)',5,1/2,'low'))];
vk=[detrend(filtdat(array.swift22.v(subsample)',5,1/2,'low')),detrend(filtdat(array.swift23.v(subsample)',5,1/2,'low')),detrend(filtdat(array.swift24.v(subsample)',5,1/2,'low')),detrend(filtdat(array.swift25.v(subsample)',5,1/2,'low'))];

tk=([array.swift22.rawtime(subsample)',array.swift23.rawtime(subsample)',array.swift24.rawtime(subsample)',array.swift25.rawtime(subsample)']-t0).*24.*3600;

end

[zp,zc,params,t] = leastSquaresWavePropagation(zk,uk,vk,tk,xk,yk,tp,xt,yt,wavespec);

if data_deny
prediction.zp(target_samp,:)=zp(1:length(tp),:)';
% prediction.up(target_samp,:)=zp(length(tp)+1:2*length(tp),:)';
% prediction.vp(target_samp,:)=zp(2*length(tp)+1:end,:)';
prediction.zt(target_samp)=zt';
% prediction.ut(target_samp)=ut';
% prediction.vt(target_samp)=vt';
else
prediction.zp(target_samp,:,:)=reshape(zp(1:length(tp)),[length(target_samp) size(WFA1.x)]);
% prediction.up(target_samp,:)=zp(length(tp)+1:2*length(tp),:)';
% prediction.vp(target_samp,:)=zp(2*length(tp)+1:end,:)';
end
prediction.params(target_samp)=params;
prediction.comp_time(target_samp)=t;
prediction.zc(subsample,:)=reshape(zc(1:length(zk(:))),length(subsample),size(prediction.tm,2));
% prediction.uc(subsample,:)=reshape(zc(length(zk(:))+1:2*length(zk(:))),length(subsample),size(prediction.tm,2));
% prediction.vc(subsample,:)=reshape(zc(2*length(zk(:))+1:end),length(subsample),size(prediction.tm,2));
prediction.zm(subsample,:)=reshape(zk,length(subsample),size(prediction.tm,2));
% prediction.um(subsample,:)=reshape(uk,length(subsample),size(prediction.tm,2));
% prediction.vm(subsample,:)=reshape(vk,length(subsample),size(prediction.tm,2));

end
end




