function skill=calculate_skillscores
burst_files=dir('./Sep7-Oct4/prediction_bursts/*.mat');
files=dir('./Sep7-Oct4/data_denial/*.mat');
for i=1:length(files)
load(['./Sep7-Oct4/prediction_bursts/' burst_files(i).name])
load(['./Sep7-Oct4/data_denial/' files(i).name])
P=prediction_skill_score(array,prediction);
good=find(~isnan(prediction.zp));
skill(i)=1-sum((prediction.zp(good)-prediction.zt(good)).^2)./median(sum((P(good,:)-prediction.zt(good)).^2));
end