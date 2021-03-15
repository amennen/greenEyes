# classifyEntireRun_plotting 
# use this script after loading the data (so you don't need other packages and can use rtcloud)

# for seaborn use the myclone environment
import numpy as np
import pickle
import os
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import stats
from numpy.polynomial.polynomial import polyfit

from commonPlotting import *
offline_path = '/jukebox/norman/amennen/prettymouth_fmriprep2/code/saved_classifiers'
fmriprep_path = '/jukebox/norman/amennen/RT_prettymouth/data/bids/Norman/Mennen/5516_greenEyes/derivatives/fmriprep'
TOM_large = '/jukebox/norman/amennen/prettymouth_fmriprep2/ROI/TOM_large_resampled_maskedbybrain.nii.gz'
TOM_cluster = '/jukebox/norman/amennen/prettymouth_fmriprep2/ROI/TOM_cluster_resampled_maskedbybrain.nii.gz'
DMNmask='/jukebox/norman/amennen/MNI_things/Yeo_JNeurophysiol11_MNI152/Yeo_Network7mask_reoriented_resampledBOLD2.nii.gz'

subjects = np.array([25,26,28,29,30,31,32,33,35,36,37,38,39,41,40,42,43,44,45,46])
n_subs = len(subjects)
n_runs = 4
P2 = makeColorPalette(['#99d8c9','#fc9272'])
paranoid_c = '#99d8c9'
cheating_c = '#fc9272'
n_voxels = 2414
run_TRs = 450


# STATION INFORMATION
stationsDict = np.load('/jukebox/norman/amennen/prettymouth_fmriprep2/code/upper_right_winners_nofilter.npy',allow_pickle=True).item()
### CHANGING N STATIONS HERE ###
# For experiment 1 - there were 9 stations; For experiment 2 - there were 7 stations
nStations = 7 #len(stationsDict)
good_stations = np.arange(nStations) # because I specified all of the stations here!
good_subset = {key: stationsDict[key] for key in list(range(0,nStations))}
all_station_indices = sum(good_subset.values(), [])
non_station_indices = [i for i in list(range(0,run_TRs,1)) if i not in all_station_indices] 
opp_run_TRs = len(non_station_indices)

maskType = 1
removeAvg = 1
filterType = 0
k1 = 0
k2 = 25
filename_clf = offline_path + '/' + 'LOGISTIC_lbfgs_UPPERRIGHT_NOstations_' + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'
loaded_model = pickle.load(open(filename_clf, 'rb'))

filename_clf = offline_path + '/' + 'LOGISTIC_lbfgs_UPPERRIGHT_prepostInd_' + str(0) + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'
loaded_model_pre = pickle.load(open(filename_clf, 'rb'))

filename_clf = offline_path + '/' + 'LOGISTIC_lbfgs_UPPERRIGHT_prepostInd_' + str(1) + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'
loaded_model_post = pickle.load(open(filename_clf, 'rb'))

filename_clf_opposite = offline_path + '/' 'LOGISTIC_lbfgs_UPPERRIGHT_OPPOSITEstations_' + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'
loaded_model_opp = pickle.load(open(filename_clf_opposite, 'rb'))

allData_loaded = np.load('allSubjectsData_fmripreped.npy')
# now convert to p_cheating

## calculate pre/post stations
station_start = 28 # first TR in station 0
station_end = 354 # last Tr in station 6
n_clf = 2
prob_cheating_pre_post = np.zeros((n_subs,n_clf,n_runs))
prob_cheating_whole_run = np.zeros((n_subs,n_runs))
prob_cheating_opp_run = np.zeros((n_subs,n_runs))

for s in np.arange(n_subs):
	for r in np.arange(n_runs):

	# check if any nans
		if np.isnan(allData_loaded[:,:,r,s]).any():
			prob_cheating_pre_post[s,:,r] = np.nan
			prob_cheating_whole_run[s,r] = np.nan
			prob_cheating_opp_run[s,r] = np.nan
		else:
			testing_data_reshaped = np.reshape(allData_loaded[:,:,r,s],(1,n_voxels*run_TRs))
			prob_cheating_whole_run[s,r] = loaded_model.predict_proba(testing_data_reshaped)[0][1]
			testing_data_reshaped = np.reshape(allData_loaded[:,non_station_indices,r,s],(1,n_voxels*opp_run_TRs))
			prob_cheating_opp_run[s,r] = loaded_model_opp.predict_proba(testing_data_reshaped)[0][1]
			# go through pre/post
			for c in np.arange(n_clf):
				if c == 0:
					# we want to take everything BEFORE that first station
					TR_use = np.arange(0,station_start)
					n_TR_use = len(TR_use)	
					this_testing_data = allData_loaded[:,TR_use,r,s]
					testing_data_reshaped = np.reshape(this_testing_data,(1,n_voxels*n_TR_use))
					prob_cheating_pre_post[s,c,r] = loaded_model_pre.predict_proba(testing_data_reshaped)[0][1]
				elif c == 1:
					# we want to take everything AFTER that last station
					TR_use = np.arange(station_end+1,run_TRs)
					n_TR_use = len(TR_use)	
					this_testing_data = allData_loaded[:,TR_use,r,s]
					testing_data_reshaped = np.reshape(this_testing_data,(1,n_voxels*n_TR_use))
					prob_cheating_pre_post[s,c,r] = loaded_model_post.predict_proba(testing_data_reshaped)[0][1]

interpretations = {}
for s in np.arange(n_subs):
  interpretations[s] = getSubjectInterpretation(subjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']


# now get behavior
projectDir = '/jukebox/norman/amennen/RT_prettymouth/data/laptopData/'

all_context_scores = np.zeros((n_subs,))
all_story_scores = np.zeros((n_subs,))
nR = 9
all_rating_scores = np.zeros((n_subs,nR))
for s in np.arange(n_subs):  
    subject = subjects[s]
    context = getSubjectInterpretation(subject)
    bids_id = 'sub-{0:03d}'.format(subject)
    response_mat = projectDir + bids_id + '/' + 'responses_scored.mat'
    z = scipy.io.loadmat(response_mat)
    ratings =  z['key_rating'][0]
    all_rating_scores[s,:] = ratings
    context_score =  z['mean_context_score'][0][0]
    all_context_scores[s] = context_score
    story_score = z['story_score'][0][0]
    all_story_scores[s] = story_score

# change interpretation scores [context_score] to account for the 1 question that all subjects answered the same
z=all_context_scores*12
new_context_scores = (z.copy() + 1)/11
all_correct_context = new_context_scores.copy()
all_correct_context[P_ind] = -1*new_context_scores[P_ind]
all_context_scores = new_context_scores

# load top and bottom subjects
top_subjects = np.load('top_subj.npy')
bottom_subjects = np.load('bottom_subj.npy')

C_top = list(set.intersection(set(C_ind), set(list(top_subjects))))
P_top = list(set.intersection(set(P_ind), set(list(top_subjects))))
C_bottom = list(set.intersection(set(C_ind), set(list(bottom_subjects))))
P_bottom = list(set.intersection(set(P_ind), set(list(bottom_subjects))))

all_correct_prob_whole_run = prob_cheating_whole_run.copy()
all_correct_prob_whole_run[P_ind,:] = 1 - prob_cheating_whole_run[P_ind,:]
all_correct_prob_opp_run = prob_cheating_opp_run.copy()
all_correct_prob_opp_run[P_ind,:] = 1 - prob_cheating_opp_run[P_ind,:]

all_correct_prob_pre_post = prob_cheating_pre_post.copy()
all_correct_prob_pre_post[P_ind,:,:] = 1 - prob_cheating_pre_post[P_ind,:,:]
# all_correct_prob_z = stats.zscore(all_correct_prob,axis=2,ddof=1)
# prob_cheating_diff = np.diff(prob_cheating, axis=2)
# all_correct_diff = prob_cheating_diff.copy()
# all_correct_diff[P_ind,:,:] = -1*prob_cheating_diff[P_ind,:,:] #instead of subtracting should be multiplying


# first whole run, separated by groups
fig,ax = plt.subplots(figsize=(20,9))
sns.despine()
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(prob_cheating_whole_run[P_ind,:],axis=0),
	yerr=scipy.stats.sem(prob_cheating_whole_run[P_ind,:],
	axis=0,nan_policy='omit'
	),
	color=paranoid_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(prob_cheating_whole_run[C_ind,:],axis=0),
	yerr=scipy.stats.sem(prob_cheating_whole_run[C_ind,:],
	axis=0,nan_policy='omit'
	),
	color=cheating_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='--X',
	ms=10)
# why doesn't this agree with the other plot?
plt.xlabel('run',fontsize=25)
plt.ylabel('p(cheating)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(4),fontsize=20)
plt.savefig('savedPlots_checked/cprob_wholeRun.pdf')

# does initial scores determine final interpretation?
fig,ax = plt.subplots(figsize=(20,9))
for run in np.arange(4):
	plt.subplot(2,2,run+1)
	sns.despine()
	for s in np.arange(n_subs):
		if s in C_ind:
			color=cheating_c
		elif s in P_ind:
			color=paranoid_c
		plt.plot(prob_cheating_whole_run[s,run], all_context_scores[s], '.', color=color, ms=10)
	plt.xlim([0,1])
	plt.ylim([-1.05,1.05])
	if run > 1:
		plt.xlabel('p(cheating)')
	else:
		plt.xticks([])
	plt.ylabel('Context score')
	title = 'Run %i' %run
	plt.title(title)
	x,y=nonNan(prob_cheating_whole_run[:,run],all_context_scores)
	b, m = polyfit(x, y, 1)
	plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
	r,p=scipy.stats.pearsonr(x,y)
	text_f = 'r = %2.2f\np = %2.2f' % (r,p)
	# print(text_f)
	printStatsResults('corr', r, p)
	plot_text = 'r={0:2.2f}\np={1:2.2f}'.format(r,p)
	plt.text(0.9,-0.8, plot_text, fontsize=12)
plt.savefig('savedPlots_checked/cprob__context_wholeRun.pdf')

run=0
for s in np.arange(n_subs):
	if s in C_ind:
		color=cheating_c
	elif s in P_ind:
		color=paranoid_c
	plt.plot(prob_cheating_pre_post[s,1,run], all_context_scores[s], '.', color=color, ms=10)
plt.xlabel('p(cheating)')
plt.ylabel('Context score')
x,y=nonNan(prob_cheating_pre_post[:,1,run],all_context_scores)
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
print(text_f)

# next separate by top/bottom subjects
fig,ax = plt.subplots(figsize=(20,9))
sns.despine()
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(all_correct_prob_whole_run[top_subjects,:],axis=0),
	yerr=scipy.stats.sem(all_correct_prob_whole_run[top_subjects,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(all_correct_prob_whole_run[bottom_subjects,:],axis=0),
	yerr=scipy.stats.sem(all_correct_prob_whole_run[bottom_subjects,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='--X',
	ms=10)
# why doesn't this agree with the other plot?
plt.xlabel('run',fontsize=25)
plt.ylabel('p(correct)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(n_runs),fontsize=20)
plt.savefig('savedPlots_checked/cprob_correct_incor_wholeRun.pdf')
#########################OPPOSITE#################################
# first whole run, separated by groups
fig,ax = plt.subplots(figsize=(20,9))
sns.despine()
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(prob_cheating_opp_run[P_ind,:],axis=0),
	yerr=scipy.stats.sem(prob_cheating_opp_run[P_ind,:],
	axis=0,nan_policy='omit'
	),
	color=paranoid_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(prob_cheating_opp_run[C_ind,:],axis=0),
	yerr=scipy.stats.sem(prob_cheating_opp_run[C_ind,:],
	axis=0,nan_policy='omit'
	),
	color=cheating_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='--X',
	ms=10)
# why doesn't this agree with the other plot?
plt.xlabel('run',fontsize=25)
plt.ylabel('p(cheating)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(4),fontsize=20)
plt.savefig('savedPlots_checked/cprob_oppRun.pdf')

# does initial scores determine final interpretation?
fig,ax = plt.subplots(figsize=(20,9))
for run in np.arange(4):
	plt.subplot(2,2,run+1)
	sns.despine()
	for s in np.arange(n_subs):
		if s in C_ind:
			color=cheating_c
		elif s in P_ind:
			color=paranoid_c
		plt.plot(prob_cheating_whole_run[s,run], all_context_scores[s], '.', color=color, ms=10)
	plt.xlim([0,1])
	plt.ylim([-1.05,1.05])
	if run > 1:
		plt.xlabel('p(cheating)')
	else:
		plt.xticks([])
	plt.ylabel('Context score')
	title = 'Run %i' %run
	plt.title(title)
	x,y=nonNan(prob_cheating_whole_run[:,run],all_context_scores)
	b, m = polyfit(x, y, 1)
	plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
	r,p=scipy.stats.pearsonr(x,y)
	text_f = 'r = %2.2f\np = %2.2f' % (r,p)
	# print(text_f)
	printStatsResults('corr', r, p)
	plot_text = 'r={0:2.2f}\np={1:2.2f}'.format(r,p)
	plt.text(0.9,-0.8, plot_text, fontsize=12)
plt.savefig('savedPlots_checked/cprob__context_wholeRun.pdf')

run=0
for s in np.arange(n_subs):
	if s in C_ind:
		color=cheating_c
	elif s in P_ind:
		color=paranoid_c
	plt.plot(prob_cheating_pre_post[s,1,run], all_context_scores[s], '.', color=color, ms=10)
plt.xlabel('p(cheating)')
plt.ylabel('Context score')
x,y=nonNan(prob_cheating_pre_post[:,1,run],all_context_scores)
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
print(text_f)

# next separate by top/bottom subjects
fig,ax = plt.subplots(figsize=(20,9))
sns.despine()
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(all_correct_prob_whole_run[top_subjects,:],axis=0),
	yerr=scipy.stats.sem(all_correct_prob_whole_run[top_subjects,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(all_correct_prob_whole_run[bottom_subjects,:],axis=0),
	yerr=scipy.stats.sem(all_correct_prob_whole_run[bottom_subjects,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='--X',
	ms=10)
# why doesn't this agree with the other plot?
plt.xlabel('run',fontsize=25)
plt.ylabel('p(correct)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(n_runs),fontsize=20)
plt.savefig('savedPlots_checked/cprob_correct_incor_wholeRun.pdf')



#########################PRE/POST#################################

# next beginning/end separated by top/bottom
fig,ax = plt.subplots(figsize=(20,9))
plt.subplot(2,1,1)
sns.despine()
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(all_correct_prob_pre_post[top_subjects,0,:],axis=0),
	yerr=scipy.stats.sem(all_correct_prob_pre_post[top_subjects,0,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(all_correct_prob_pre_post[bottom_subjects,0,:],axis=0),
	yerr=scipy.stats.sem(all_correct_prob_pre_post[bottom_subjects,0,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='--X',
	ms=10)
plt.xticks([])
plt.ylabel('p(correct)')
plt.title('Pre stations')
plt.subplot(2,1,2)
sns.despine()
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(all_correct_prob_pre_post[top_subjects,1,:],axis=0),
	yerr=scipy.stats.sem(all_correct_prob_pre_post[top_subjects,1,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(all_correct_prob_pre_post[bottom_subjects,1,:],axis=0),
	yerr=scipy.stats.sem(all_correct_prob_pre_post[bottom_subjects,1,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='--X',
	ms=10)
# why doesn't this agree with the other plot?
plt.xlabel('run',fontsize=25)
plt.ylabel('p(correct)')
plt.title('Post stations')
# plt.ylim([0,1.15])
plt.xticks(np.arange(n_runs),fontsize=20)
plt.savefig('savedPlots_checked/cprob_correct_incor_prepostStations.pdf')

# average classification
p_cheating_avg = np.mean(prob_cheating_whole_run, axis=1)
for s in np.arange(n_subs):
	if s in C_ind:
		color=cheating_c
	elif s in P_ind:
		color=paranoid_c
	plt.plot(p_cheating_avg[s], all_context_scores[s], '.', color=color, ms=10)
plt.xlabel('p(cheating)')
plt.ylabel('Context score')
x,y=nonNan(p_cheating_avg,all_context_scores)
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
print(text_f)



fig = plotPosterStyle_multiplePTS(prob_cheating_diff,subjects)

# first just plot cheating scores in the end of nf by run
fig,ax = plt.subplots(figsize=(20,9))
sns.despine()
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(prob_cheating[P_ind,1,:],axis=0),
	yerr=scipy.stats.sem(prob_cheating[P_ind,1,:],
	axis=0,nan_policy='omit'
	),
	color=paranoid_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(prob_cheating[C_ind,1,:],axis=0),
	yerr=scipy.stats.sem(prob_cheating[C_ind,1,:],
	axis=0,nan_policy='omit'
	),
	color=cheating_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='--X',
	ms=10)
# why doesn't this agree with the other plot?
plt.xlabel('run',fontsize=25)
plt.ylabel('p(cheating)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(3),fontsize=20)

# run 3 - run 1
end_diff_run = prob_cheating_pre_post[:,1,3] - prob_cheating_pre_post[:,1,0]
plotPosterStyle_DF(end_diff_run,subjects)
end_diff_run_cheating = prob_cheating_whole_run[:,3] - prob_cheating_whole_run[:,0]
plotPosterStyle_DF(end_diff_run_cheating,subjects)



correct_diff_run = end_diff_run_cheating.copy()
correct_diff_run[P_ind] = -1*end_diff_run_cheating[P_ind]
data = {}
data_vector = correct_diff_run
subject_vector = np.arange(nSubs)
group_str = [''] * nSubs
for s in np.arange(nSubs):
	if s in top_subjects:
		group_str[s] = 'best'
	elif s in bottom_subjects:
		group_str[s] = 'worst'
data['comprehension'] = data_vector
data['subject'] = subject_vector
data['group'] = group_str
df = pd.DataFrame.from_dict(data)
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='group',y='comprehension',ci=68,linewidth=2.5,color='k', alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='group',y='comprehension',split=True,color='k',size=8)



data_df = convertMatToDF(end_diff_run, subjects)
fig,ax = plt.subplots(figsize=(20,9))
for s in np.arange(n_subs):
	if s in C_ind:
		color=cheating_c
	elif s in P_ind:
		color=paranoid_c
	plt.plot(s,end_diff_run[s],'.', color=color,ms=10)
#### next

fig,ax = plt.subplots(figsize=(20,9))
sns.despine()
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(all_correct_prob[top_subjects,1,:],axis=0),
	yerr=scipy.stats.sem(all_correct_prob_z[top_subjects,1,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(all_correct_prob[bottom_subjects,1,:],axis=0),
	yerr=scipy.stats.sem(all_correct_prob_z[bottom_subjects,1,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='--X',
	ms=10)
# why doesn't this agree with the other plot?
plt.xlabel('run',fontsize=25)
plt.ylabel('p(correct)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(3),fontsize=20)



# instead -- plot for each clf over all runs
fig,ax = plt.subplots(figsize=(20,9))
sns.despine()
plt.errorbar(
	x=np.arange(3),
	y=np.nanmean(all_correct_diff[top_subjects,1,:],axis=0),
	yerr=scipy.stats.sem(all_correct_diff[top_subjects,1,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(3),
	y=np.nanmean(all_correct_diff[bottom_subjects,1,:],axis=0),
	yerr=scipy.stats.sem(all_correct_diff[bottom_subjects,1,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='--X',
	ms=10)
plt.xlabel('run',fontsize=25)
plt.ylabel('p(correct)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(3),fontsize=20)



fig,ax = plt.subplots(figsize=(20,9))
sns.despine()
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(prob_cheating[P_ind,:],axis=0),
	yerr=scipy.stats.sem(prob_cheating[P_ind,:],
	axis=0,nan_policy='omit'
	),
	color=paranoid_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(prob_cheating[C_ind,:],axis=0),
	yerr=scipy.stats.sem(prob_cheating[C_ind,:],
	axis=0,nan_policy='omit'
	),
	color=cheating_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.xlabel('run',fontsize=25)
plt.ylabel('p(cheating)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(4),fontsize=20)

fig,ax = plt.subplots(figsize=(20,9))
sns.despine()
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(prob_cheating[P_bottom,:],axis=0),
	yerr=scipy.stats.sem(prob_cheating[P_bottom,:],
	axis=0,nan_policy='omit'
	),
	color=paranoid_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(4),
	y=np.nanmean(prob_cheating[C_bottom,:],axis=0),
	yerr=scipy.stats.sem(prob_cheating[C_bottom,:],
	axis=0,nan_policy='omit'
	),
	color=cheating_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.xlabel('run',fontsize=25)
plt.ylabel('p(cheating)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(4),fontsize=20)

# get changes from one run to the next
subj_diff = np.diff(prob_cheating,axis=1)
all_correct_prob = prob_cheating.copy()
all_correct_prob[P_ind,:] = 1 - prob_cheating[P_ind,:]
fig,ax = plt.subplots(figsize=(20,9))
plt.subplot(2,1,1)
sns.despine()
plt.errorbar(
	x=np.arange(3),
	y=np.nanmean(subj_diff[P_top,:],axis=0),
	yerr=scipy.stats.sem(subj_diff[P_top,:],
	axis=0,nan_policy='omit'
	),
	color=paranoid_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(3),
	y=np.nanmean(subj_diff[C_top,:],axis=0),
	yerr=scipy.stats.sem(subj_diff[C_top,:],
	axis=0,nan_policy='omit'
	),
	color=cheating_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.xlabel('run',fontsize=25)
plt.ylabel('p(cheating)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(4),fontsize=20)
plt.title('TOP')

plt.subplot(2,1,2)
sns.despine()
plt.errorbar(
	x=np.arange(3),
	y=np.nanmean(subj_diff[P_bottom,:],axis=0),
	yerr=scipy.stats.sem(subj_diff[P_bottom,:],
	axis=0,nan_policy='omit'
	),
	color=paranoid_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(3),
	y=np.nanmean(subj_diff[C_bottom,:],axis=0),
	yerr=scipy.stats.sem(subj_diff[C_bottom,:],
	axis=0,nan_policy='omit'
	),
	color=cheating_c,
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.xlabel('run',fontsize=25)
plt.ylabel('p(cheating)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(4),fontsize=20)
plt.title('BOTTOM')


## convert to correct direction
all_correct_prob = prob_cheating.copy()
all_correct_prob[P_ind,:] = 1 - prob_cheating[P_ind,:]
subj_diff_correct = np.diff(all_correct_prob,axis=1)
fig,ax = plt.subplots(figsize=(20,9))
plt.subplot(2,1,1)
sns.despine()
plt.errorbar(
	x=np.arange(3),
	y=np.nanmean(subj_diff_correct[top_subjects,:],axis=0),
	yerr=scipy.stats.sem(subj_diff_correct[top_subjects,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='-o',
	ms=10)
plt.errorbar(
	x=np.arange(3),
	y=np.nanmean(subj_diff_correct[bottom_subjects,:],axis=0),
	yerr=scipy.stats.sem(subj_diff_correct[bottom_subjects,:],
	axis=0,nan_policy='omit'
	),
	color='k',
	alpha=0.7,
	lw=3,
	label='top',
	fmt='--X',
	ms=10)
plt.xlabel('run',fontsize=25)
plt.ylabel('p(correct)')
# plt.ylim([0,1.15])
plt.xticks(np.arange(3),fontsize=20)
