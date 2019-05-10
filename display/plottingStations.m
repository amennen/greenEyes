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