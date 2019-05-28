TR = 1.5
musicDur = 18; % how long the music lasts
silenceDur1 = 3;
storyTRs = 25:475;
nTRs_story = length(storyTRs);
nTRs_music = musicDur/TR;
stationTRs = zeros(nTRs_story,1); % actual files wer're going to use to test classifier
recordedTRs = zeros(nTRs_story,1);
st = load('stationsMat.mat');
nStations = size(st.stationsDict,2);
for s = 1:nStations
   this_station_TRs = st.stationsDict{s} + 1;
   stationTRs(this_station_TRs) = s;
   recordedTRs(this_station_TRs-3) = s;
end
% now make array specifying which station to look for
lookTRs = zeros(nTRs_story,1);
for t = 1:nTRs_story
   pastStation = max(stationTRs(1:t));
   lastTRForStation = find(stationTRs == pastStation);
   lastTRForStation = lastTRForStation(end);
   if pastStation > 0 && (t-lastTRForStation <=15) 
        lookTRs(t) = pastStation;
   end
end
% make all look TRs during recording parts 0
lookTRs(recordedTRs>0) = 0;
lookTRs(stationTRs>0) = 0;
%%


figure()
xlabel('TR')
ylabel('Station #')
title('Station Timing')
set(findall(gcf,'-property','FontSize'),'FontSize',18)
hold on;
plot(recordedTRs, 'r.', 'MarkerSize', 20)
plot(stationTRs+0.1, '.', 'MarkerSize', 20)
plot(lookTRs+0.25, 'k.', 'MarkerSize', 20)
legend('Station', 'Shifted TR', 'Look for data')
ylim([0.5,10])
stationInd = find(stationTRs>0);
all_diff = diff(stationInd) ;
all_diff(all_diff > 1)

(all_diff(all_diff > 1))*1.5


mean(all_diff(all_diff > 1))

mean((all_diff(all_diff > 1))*1.5)