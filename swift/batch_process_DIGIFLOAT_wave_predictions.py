function batch_process_DIGIFLOAT_wave_predictions(data_deny)

files=dir('./Sep7-Oct4/prediction_bursts/*.mat');


for i=1:333
load(['./Sep7-Oct4/prediction_bursts/' files(i).name])
[array,prediction]=run_LS_prediction_SWIFTS(array,data_deny);
if data_deny
prediction.zp(~isnan(prediction.zp))=movavg(prediction.zp(~isnan(prediction.zp)),5); %1 Hz smoothing filter
save(['./Sep7-Oct4/data_denial/' files(i).name],'prediction')
else
zp=reshape(prediction.zp,size(prediction.zp,1),16*73);
zp(~isnan(zp(:,1)),:)=movavg(zp(~isnan(zp(:,1)),:),5);%1 Hz smoothing filter
prediction.zp=reshape(zp,size(prediction.zp));
save(['./Sep7-Oct4/WFA1_reconstructions/' files(i).name],'prediction')
end
end
