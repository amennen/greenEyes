# Purpose: analyze data from EXP 2 (when probes are pressed)
# LEFT OFF - check if all the plots were used and comment out which ones weren't, add to notes
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json 
import datetime
from dateutil import parser
from subprocess import call
import time
import nilearn
from nilearn.masking import apply_mask
from scipy import stats
import scipy.io as sio
import pickle
import nibabel as nib
import argparse
import sys
import logging
import matplotlib.cm as cmx
import seaborn as sns
import scipy
from numpy.polynomial.polynomial import polyfit

sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
# when running not in file: sys.path.append(os.getcwd())
#WHEN TESTING
#sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
from rtCommon.utils import loadConfigFile, dateStr30, DebugLevels, writeFile, loadMatFile
from rtCommon.readDicom import readDicomFromBuffer
from rtCommon.fileClient import FileInterface
from rtCommon.structDict import StructDict
import rtCommon.dicomNiftiHandler as dnh
import greenEyes
from matplotlib.lines import Line2D
from commonPlotting import *

# params = {'legend.fontsize': 'large',
#           'figure.figsize': (5, 3),
#           'axes.labelsize': 'x-large',
#           'axes.titlesize': 'x-large',
#           'xtick.labelsize': 'x-large',
#           'ytick.labelsize': 'x-large'}
# font = {'weight': 'normal',
#         'size': 22}
# plt.rc('font', **font)
defaultConfig = os.path.join(os.getcwd(), 'conf/greenEyes_cluster.toml')
cfg = loadConfigFile(defaultConfig)
params = StructDict({'config':defaultConfig, 'runs': '1', 'scans': '9', 'webpipe': 'None', 'webfilesremote': False})
cfg = greenEyes.initializeGreenEyes(defaultConfig,params)
# date doesn't have to be right, but just make sure subject number, session number, computers are correct

def getPatternsData(subject_num,run_num):
    bids_id = 'sub-{0:03d}'.format(subject_num)
    ses_id = 'ses-{0:02d}'.format(2)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/patternsData_r{2}_*.mat'.format(bids_id,ses_id,run_num)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    cheating_prob = data['cheating_probability']
    cheating_prob_z = data['zTransferred']
    correct_score = data['correct_prob']
    return data, cheating_prob, cheating_prob_z, correct_score

def getBehavData(subject_num,run_num):
    bids_id = 'sub-{0:03d}'.format(subject_num)
    ses_id = 'ses-{0:02d}'.format(2)
    run_id = 'run-{0:03d}'.format(run_num)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/{2}/behavior_run{3}_*.mat'.format(bids_id,ses_id,run_id,run_num)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data

def getStationInformation(config='conf/greenEyes_cluster.toml'):
    allinfo = {}
    cfg = loadConfigFile(config)
    station_FN = cfg.cluster.classifierDir + '/' + cfg.stationDict
    station_dict = np.load(station_FN,allow_pickle=True).item()
    nStations = len(station_dict)
    last_tr_in_station = np.zeros((nStations,))
    allTR = list(station_dict.values())
    all_station_TRs = [item for sublist in allTR for item in sublist]
    for st in np.arange(nStations):
        last_tr_in_station[st] = station_dict[st][-1]
    return nStations, station_dict, last_tr_in_station, all_station_TRs

def getBehavData(subject_num,run_num):
    bids_id = 'sub-{0:03d}'.format(subject_num)
    ses_id = 'ses-{0:02d}'.format(2)
    run_id = 'run-{0:03d}'.format(run_num)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/{2}/behavior_run{3}_*.mat'.format(bids_id,ses_id,run_id,run_num)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data    

def createStationVector(station_dict):
    nStations = len(station_dict)
    allTRs = np.arange(25,475+1)
    nTRs_story = len(allTRs)
    recorded_TRs = np.zeros((nTRs_story,))
    for st in np.arange(nStations):
        this_station_TRs = np.array(station_dict[st]) # REMOVE + 1 # going from python --> matlab indexing
        recorded_TRs[this_station_TRs - 3] = st + 1
    return recorded_TRs

def getProbeKeyPresses(behav_data,recorded_TRs):
    run_data = behav_data['runData']
    left_key = run_data['LEFT_PRESS'][0]
    leftPress = run_data['leftPress']
    allLeftTRs = np.argwhere(leftPress[0,:] == 1)
    rightPress = run_data['rightPress']
    right_key = run_data['RIGHT_PRESS'][0]
    allRightTRs = np.argwhere(rightPress[0,:] == 1)
    nStations = 7
    probe_response = []
    for st in np.arange(nStations):
        LEFT = 0
        RIGHT = 0
        display_TR = np.argwhere(recorded_TRs == st+1)[0][0]

        if len(np.intersect1d(np.argwhere(display_TR - allLeftTRs[:,0] >= 0),np.argwhere(display_TR - allLeftTRs[:,0] < 3)) ) > 0:
            # then there was a left keypress for this station
            LEFT = 1
            probe_response_st = left_key
        if len(np.intersect1d(np.argwhere(display_TR - allRightTRs[:,0] >= 0),np.argwhere(display_TR - allRightTRs[:,0] < 3)) ) > 0:
            RIGHT = 1
            probe_response_st = right_key
        if LEFT and RIGHT:
            probe_response_st = 'BOTH'
        if LEFT == 0 and RIGHT == 0:
            probe_response_st = 'N/A'
        probe_response.append(probe_response_st)
    return probe_response

# get general experiment information
nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation()
recorded_TRs = createStationVector(stationDict)
nRuns = 4
subjects = np.array([25,26,28,29,30,31,32,33,35,36,37,38,39,41,40,42,43,44,45,46])
# efficacy_FB is the self-reported efficacy scores of how effective they thought they feedback was
# I tried relating this measure to others scores (see commented out parts of the script at the 
# bottom but I didn't find any relationships with these scores and peformance)
efficacy_FB = np.array([6,4,7,7,6,2,8,6,8,7,3,4,7,7,4,9,9,6,2,2])

# go through all subjects, get the cheating probability and probe response for every station
nSubs = len(subjects)
cheating_prob = np.empty((0,),int)
all_response = []
subj_means = np.zeros((nSubs,2)) * np.nan
subj_means_z = np.zeros((nSubs,2)) * np.nan
all_nf_score = np.zeros((nSubs,nStations,nRuns))*np.nan
all_cheating_prob = np.zeros((nSubs,nStations,nRuns))*np.nan
all_cheating_prob_z = np.zeros((nSubs,nStations,nRuns))*np.nan
cheating_prob_by_run = np.zeros((nStations,nRuns,nSubs)) * np.nan
subj_vector = []
station_vector =  np.empty((0,),int)
cheating_prob_cheating = np.zeros((nSubs,nStations*nRuns))* np.nan
cheating_prob_paranoid = np.zeros((nSubs,nStations*nRuns))* np.nan
for s in np.arange(nSubs):
    subjectNum = subjects[s]
    this_sub_cheating = np.empty((0,),int)
    this_sub_cheating_z = np.empty((0,),int)
    this_sub_response = []
    if subjectNum==41:
        nRuns = 2
    else:
        nRuns = 4
    for runNum in np.arange(nRuns):
        data, c_prob, c_prob_z, nf_score = getPatternsData(subjectNum,runNum + 1)
        all_nf_score[s,:,runNum] = nf_score
        all_cheating_prob[s,:,runNum] = c_prob
        all_cheating_prob_z[s,:,runNum] = c_prob_z
        this_sub_cheating = np.append(this_sub_cheating,c_prob[0,:])
        this_sub_cheating_z = np.append(this_sub_cheating_z,c_prob_z[0,:])
        # now get probe answers
        b = getBehavData(subjectNum,runNum+1)
        probe_response = getProbeKeyPresses(b,recorded_TRs)
        this_sub_response = this_sub_response + probe_response
        cheating_prob_by_run[:,runNum,s] = c_prob[0,:]
    sub_stations = np.tile(np.arange(nStations),nRuns)
    station_vector = np.append(station_vector,sub_stations)
    ind_cheating = [i for i, x in enumerate(this_sub_response) if x == "CHEATING"]  
    ind_paranoid = [i for i, x in enumerate(this_sub_response) if x == "PARANOID"]  
    subj_means[s,0] = np.nanmean(this_sub_cheating[ind_cheating])
    subj_means[s,1] = np.nanmean(this_sub_cheating[ind_paranoid])
    subj_means_z[s,0] = np.nanmean(this_sub_cheating_z[ind_cheating])
    subj_means_z[s,1] = np.nanmean(this_sub_cheating_z[ind_paranoid])
    cheating_prob = np.append(cheating_prob,this_sub_cheating)
    if len(ind_cheating) > 1:
        cheating_prob_cheating[s,0:len(ind_cheating)] = this_sub_cheating[ind_cheating]
    else:
        cheating_prob_cheating[s,0] = this_sub_cheating[ind_cheating]
    if len(ind_paranoid) > 1:
        cheating_prob_paranoid[s,0:len(ind_paranoid)] = this_sub_cheating[ind_paranoid]
    else:
        cheating_prob_paranoid[s,0] = this_sub_cheating[ind_paranoid]
    all_response = all_response + this_sub_response
    subj_vector = subj_vector + [str(subjectNum)] * len(this_sub_response)

# classifier separation of cheating probability based on probe response
classifier_separation = subj_means[:,0] - subj_means[:,1]
print('classifier separation')
print(classifier_separation)
print('n correct side')
all_correct_classification = np.argwhere(classifier_separation>0)[:,0]
print(len(all_correct_classification))
# specify plotting colors
P2 = makeColorPalette(['#99d8c9','#fc9272'])
paranoid_c = '#99d8c9'
cheating_c = '#fc9272'

# get interpretations group
interpretations = {}
for s in np.arange(nSubs):
  interpretations[s] = getSubjectInterpretation(subjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']

# get behavioral data
all_context_scores = np.zeros((nSubs,))
all_correct_context = np.zeros((nSubs,))
all_story_scores = np.zeros((nSubs,))
nR = 9
all_rating_scores = np.zeros((nSubs,nR))
for s in np.arange(nSubs):  
    subject = subjects[s]
    context = getSubjectInterpretation(subject)
    bids_id = 'sub-{0:03d}'.format(subject)
    response_mat = '/jukebox/norman/amennen/RT_prettymouth/data/laptopData/' + bids_id + '/' + 'responses_scored.mat'
    z = scipy.io.loadmat(response_mat)
    ratings =  z['key_rating'][0]
    all_rating_scores[s,:] = ratings
    context_score =  z['mean_context_score'][0][0]
    all_context_scores[s] = context_score
    story_score = z['story_score'][0][0]
    all_story_scores[s] = story_score

# correct interpretation scores for the bad question
z=all_context_scores*12
new_context_scores = (z.copy() + 1)/11
all_correct_context = new_context_scores.copy()
all_correct_context[P_ind] = -1*new_context_scores[P_ind]

# subtract arthur - lee empathy
arthur_minus_lee = all_rating_scores[:,0] - all_rating_scores[:,1]
lee = all_rating_scores[:,1]
lee_cor = lee.copy()
lee[C_ind] = -1*lee[C_ind]
# convert arthur - lee empathy to the correct direction
arthur_minus_lee_cor = arthur_minus_lee.copy()
arthur_minus_lee_cor[P_ind] = -1*arthur_minus_lee[P_ind]


# reformat data into a data frame for plotting
group_str = [''] * nSubs
for s in np.arange(nSubs):
  if s in C_ind:
    group_str[s] = 'C'
  elif s in P_ind:
    group_str[s] = 'P'
vec_subj_means = subj_means.flatten()
vec_subj_means_z = subj_means_z.flatten()
resp = ['CHEATING','PARANOID']*nSubs
subj = np.repeat(np.arange(nSubs),2)
group_vector = np.repeat(group_str,2)
DATA = {}
DATA['resp'] = resp
DATA['subj'] = subj
DATA['p(cheating)'] = vec_subj_means
DATA['z-scored'] = vec_subj_means_z
DATA['group'] = group_vector
df = pd.DataFrame.from_dict(DATA)

# (1) plot average classifier separation by subject
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='resp',y='p(cheating)',ci=68,order=['CHEATING', 'PARANOID'],color='k',alpha=0.5)
g = sns.swarmplot(data=df[df['group']=='C'],x='resp',y='p(cheating)',order=['CHEATING', 'PARANOID'],color=cheating_c,alpha=0.5,size=10)
g = sns.swarmplot(data=df[df['group']=='P'],x='resp',y='p(cheating)',order=['CHEATING', 'PARANOID'],color=paranoid_c,alpha=0.5,size=10)
for s in np.arange(nSubs):
    plt.plot([0,1],subj_means[s,:], '--', color='k', lw=1, alpha=0.3)
plt.title('Classification divided by probe response',fontsize=30)
plt.ylim([0,1])
plt.xlabel('probe response',fontsize=25)
plt.ylabel('p(cheating)',fontsize=25)
plt.xticks(fontsize=20)
plt.xticks(fontsize=20)
x,y=nonNan(subj_means[:,0],subj_means[:,1])
r,p=scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0,1,0.92,.05,0,text_above='C>P')
printStatsResults('avg classification based on probe responses 1-sided', r, p/2)
plt.ylim([0,1.1])
plt.savefig('savedPlots_checked/classification_averaged.pdf')
#plt.show()

# (2) plot the linear relationship between correct context responses and classifier accuracy
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
for s in np.arange(nSubs):
    if interpretations[s] == 'C':
        color=cheating_c
    elif interpretations[s] == 'P':
        color=paranoid_c
    plt.plot(classifier_separation[s],all_correct_context[s],'.',ms=20,color=color,alpha=1)
b, m = polyfit(classifier_separation, all_correct_context, 1)
plt.plot(classifier_separation, b + m * classifier_separation, '-',alpha=0.6,lw=3, color='k')
plt.xlabel('classification accuracy',fontsize=25)
plt.ylabel('correct interpretation score', fontsize=25)
plt.title('Interpretation and classification relationship',fontsize=30)
r,p=scipy.stats.pearsonr(classifier_separation,all_correct_context)
printStatsResults('interpretation and classifier linear relationship',r, p)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-0.2,0.8,text_f,fontsize=25)
plt.savefig('savedPlots_checked/classification_context.pdf')
#plt.show()


# (3) plot linear relationship between empathy difference and intepretation score
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
for s in np.arange(nSubs):
    if interpretations[s] == 'C':
        color=cheating_c
    elif interpretations[s] == 'P':
        color=paranoid_c
    plt.plot(all_correct_context[s],arthur_minus_lee_cor[s],'.',ms=20,color=color,alpha=1)
plt.xlabel('correct interpretation score',fontsize=25)
plt.ylabel('correct empathy difference',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Empathy and interpretation relationship',fontsize=30)
x = all_correct_context
y = arthur_minus_lee_cor
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
printStatsResults('interpretation and empathy linear relationship', r, p)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-1,3,text_f,fontsize=25)
plt.savefig('savedPlots_checked/empathy_context.pdf')
#plt.show()


# (4) - plot neurofeedback scores over all run, averaged by group
fig = plotPosterStyle_multiplePTS(all_nf_score,subjects)
plt.subplot(1,4,1)
plt.ylabel('NF score ($)',fontsize=25)
plt.ylim([0,1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('run 1',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.subplot(1,4,2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0,1])
plt.title('run 2',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.subplot(1,4,3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0,1])
plt.title('run 3',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.subplot(1,4,4)
plt.title('run 4',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0,1])
plt.xlabel('station',fontsize=25)
plt.savefig('savedPlots_checked/nf_score.pdf')
#plt.show()

# calculate variance for p(cheating)
cheating_prob_mean = np.nanmean(all_cheating_prob,axis=2)
# calculate within-group variance 
stationVariance = np.zeros((nStations,2)) # 1 for each group
for st in np.arange(nStations):
    C_data = cheating_prob_mean[C_ind,st]
    P_data = cheating_prob_mean[P_ind,st]
    stationVariance[st,0] = scipy.stats.sem(C_data)
    stationVariance[st,1] = scipy.stats.sem(P_data)
printStatsResults('Within group variance C correct_prob', np.mean(stationVariance),-1)

# calculate variance for p(cheating) z scored
cheating_prob_mean_z = np.nanmean(all_cheating_prob_z,axis=2)
# calculate within-group variance 
stationVariance = np.zeros((nStations,2)) # 1 for each group
for st in np.arange(nStations):
    C_data = cheating_prob_mean_z[C_ind,st]
    P_data = cheating_prob_mean_z[P_ind,st]
    stationVariance[st,0] = scipy.stats.sem(C_data)
    stationVariance[st,1] = scipy.stats.sem(P_data)
printStatsResults('Within group variance C correct_prob - z scored', np.mean(stationVariance),-1)

# calculate variance for neurofeedback scores
nf_score_mean = np.nanmean(all_nf_score,axis=2)
# calculate within-group variance 
stationVariance = np.zeros((nStations,2)) # 1 for each group
for st in np.arange(nStations):
    C_data = nf_score_mean[C_ind,st]
    P_data = nf_score_mean[P_ind,st]
    stationVariance[st,0] = scipy.stats.sem(C_data)
    stationVariance[st,1] = scipy.stats.sem(P_data)
printStatsResults('Within group variance NF scores', np.mean(stationVariance),-1)

# now calculate correlations across groups
C_group_average = np.nanmean(all_nf_score[C_ind,:,:],axis=0)
P_group_average = np.nanmean(all_nf_score[P_ind,:,:],axis=0)
# look at run 1 correlation across groups
r,p=scipy.stats.pearsonr(C_group_average[:,0],P_group_average[:,0])
printStatsResults('between group nf correlation - run 1', r, p)
# look at run 4 correlation across groups
r,p=scipy.stats.pearsonr(C_group_average[:,3],P_group_average[:,3])
printStatsResults('between group nf correlation - run 4', r, p)


# if we wanted to plot p(cheating) like with experiment 1, this is how we could do it
# fig = plotPosterStyle_multiplePTS(all_cheating_prob,subjects)
# plt.subplot(1,4,1)
# plt.ylabel('p(cheating)')
# plt.ylim([0,1])
# plt.title('run 1')
# plt.xlabel('station')
# plt.subplot(1,4,2)
# plt.ylim([0,1])
# plt.title('run 2')
# plt.xlabel('station')
# plt.subplot(1,4,3)
# plt.ylim([0,1])
# plt.title('run 3')
# plt.xlabel('station')
# plt.subplot(1,4,4)
# plt.title('run 4')
# plt.ylim([0,1])
# plt.xlabel('station')
# plt.savefig('savedPlots/p_cheating.pdf')
# plt.show()

# other analyses:
# DATA = {}
# DATA['resp'] = all_response
# DATA['subj'] = subj_vector
# DATA['p(cheating)'] = cheating_prob
# DATA['station'] = station_vector
# ind_cheating = [i for i, x in enumerate(all_response) if x == "CHEATING"]  
# ind_paranoid = [i for i, x in enumerate(all_response) if x == "PARANOID"]  
# n_cheating = len(ind_cheating)
# df = pd.DataFrame.from_dict(DATA)
# plt.figure()
# sns.barplot(data=df,x='station',y='p(cheating)',hue='resp',ci=68)
# plt.title('Classifier evidence across subjects')
# plt.ylim([0,1])
# plt.xlabel('Response to probe')
# plt.ylabel('p(cheating)')
# plt.show()

# df = pd.DataFrame.from_dict(DATA)
# plt.figure()
# sns.barplot(data=df,x='resp',y='p(cheating)',ci=68)
# plt.title('Classifier evidence across subjects')
# plt.ylim([0,1])
# plt.xlabel('Response to probe')
# plt.ylabel('p(cheating)')
# plt.show()

# looking at linear probability between classification and empathy ratings, broken down by 
# correct and incorrect group
# all_incorrect_classification = np.argwhere(classifier_separation<=0)[:,0]
# all_correct_classification = np.argwhere(classifier_separation>0)[:,0]
# fig,ax = plt.subplots(figsize=(17,9))
# sns.despine()
# for s in np.arange(nSubs):
#     if interpretations[s] == 'C':
#         color=cheating_c
#     elif interpretations[s] == 'P':
#         color=paranoid_c
#     plt.plot(classifier_separation[s],arthur_minus_lee_cor[s],'.',ms=20,color=color,alpha=0.5)
# x = classifier_separation[all_incorrect_classification]
# y = arthur_minus_lee_cor[all_incorrect_classification]
# b, m = polyfit(x, y, 1)
# plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
# x2 = classifier_separation[all_correct_classification]
# y2 = arthur_minus_lee_cor[all_correct_classification]
# b2, m2 = polyfit(x2, y2, 1)
# plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
# plt.plot(x2, b2 + m2 * x2, '-',alpha=0.6,lw=3, color='k')
# plt.xlabel('difference in p(cheating) based on probe response',fontsize=10)
# plt.ylabel('empathy diff',rotation='vertical')
# r,p=scipy.stats.pearsonr(x,y)
# text_f = 'r = %2.2f\np = %2.2f' % (r,p)
# plt.savefig('savedPlots_checked/arthur_minus_lee_by_classification.pdf')
# plt.text(-0.15,0.9,text_f)
# plt.show()

##################################################################################


# plt.figure()
# plt.plot(nf_score_change3,all_correct_context,'.',ms=20)
# plt.xlabel('nf score change r4 - r1')
# plt.ylabel('correct context score')
# plt.show()
# scipy.stats.pearsonr(nf_score_change3,all_correct_context)

# plt.figure()
# plt.plot(classifier_separation,all_correct_context,'.',ms=20)
# plt.xlabel('classifier separation')
# plt.ylabel('correct context score')
# plt.show()
# scipy.stats.pearsonr(classifier_separation,all_correct_context)

# plt.figure()
# plt.plot(nf_score_change3,arthur_minus_lee_cor, '.', ms=20)
# plt.show()
# x,y=nonNan(nf_score_change3,arthur_minus_lee_cor)
# scipy.stats.pearsonr(x,y)


## RUN FROM 0 --> 1 RESULTS IN HIGHER EMPATHY RATING DIFF (but probably not context diff)
# maybe some subset of questions?
# x1 = np.mean(nf_score_by_run,axis=0)
# x1 = nf_score_change1
# cheating_prob_run = np.nanmean(cheating_prob_by_run,axis=0)
# cheating_diff = cheating_prob_run[3,:] - cheating_prob_run[0,:]
# x1 =np.nanmean(cheating_prob_run,axis=0)
# y1 = arthur_minus_lee[0:]
# plt.figure()
# plt.plot(x1,y1, '.', ms=20)
# plt.show()
# x,y=nonNan(x1,y1)
# scipy.stats.pearsonr(x,y)

