% now plot results

%interp(101) = 'C';
%interp(102) = 'P';
interp(1) = 'C';
interp(2) = 'P';
interp(3) = 'P';
interp(4) = 'C';
interp(5) = 'C';
interp(6) = 'C';
interp(7) = 'C';
interp(8) = 'C';

subjects = [ 1 2 3 4 5 6 7 8];
nSub = length(subjects);
% plot in 2 different bars
figure(1);
hold on
for s = 1:nSub
    this_sub = subjects(s);
    bids_id = sprintf('%03d', this_sub);
    filename=['data/sub-' bids_id '/responses_scored.mat'];
    z = load(filename);
    context_score = z.mean_context_score;
    this_context = interp(this_sub);
    if this_context == 'C'
        context = 1;
    elseif this_context == 'P'
        context = 2;
    end
    plot(context,context_score, '.', 'MarkerSize', 20);
    hold on;
end
set(gca,'FontSize',15)
xlim([0 3])
ylim([-1 1])
%% plot empathy ratings
RATINGS{1} = 'Empathized with Arthur';
RATINGS{2} = 'Emphathized with Lee';
RATINGS{3} = 'Empathized with Joanie';
RATINGS{4} = 'Empathized with the girl';
RATINGS{5} = 'Enjoyed the story';
RATINGS{6} = 'Felt that you were engaged with the story';
RATINGS{7} = 'Felt that the neurofeedback helped you';
RATINGS{8} = 'Are certain that your interpretation is correct';
RATINGS{9} = 'Were sleepy in the scanner';
n_ratings = 9;

% make plot for each rating
for r = 1:n_ratings
    figure;
    for s = 1:nSub
        this_sub = subjects(s);
        bids_id = sprintf('%03d', this_sub);
        filename=['data/sub-' bids_id '/responses_scored.mat'];
        z = load(filename);
        rating = z.key_rating(r);
        this_context = interp(this_sub);
        if this_context == 'C'
            context = 1;
        elseif this_context == 'P'
            context = 2;
        end
        plot(context,rating, '.', 'MarkerSize', 20);
        hold on;
    end
    xlim([0 3])
    ratingStr = sprintf('%s',RATINGS{r})
    title(ratingStr);
    set(gca,'FontSize',15)
end