# purpose collapse across groups, only top/bottom clf performance

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
from commonPlotting import *

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
    stationDict = np.load(station_FN,allow_pickle=True).item()
    n_stations = len(stationDict)
    last_tr_in_station = np.zeros((n_stations,))
    allTR = list(stationDict.values())
    all_station_TRs = [item for sublist in allTR for item in sublist]
    for st in np.arange(n_stations):
        last_tr_in_station[st] = stationDict[st][-1]
    return n_stations, stationDict, last_tr_in_station, all_station_TRs

def getBehavData(subject_num,run_num):
    bids_id = 'sub-{0:03d}'.format(subject_num)
    ses_id = 'ses-{0:02d}'.format(2)
    run_id = 'run-{0:03d}'.format(run_num)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/{2}/behavior_run{3}_*.mat'.format(bids_id,ses_id,run_id,run_num)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data    

def createStationVector(stationDict):
    n_stations = len(stationDict)
    allTRs = np.arange(25,475+1)
    nTRs_story = len(allTRs)
    recorded_TRs = np.zeros((nTRs_story,))
    for st in np.arange(n_stations):
        this_station_TRs = np.array(stationDict[st]) # remove + 1 # going from python --> matlab indexing
        recorded_TRs[this_station_TRs - 3] = st +1
    return recorded_TRs

def getProbeKeyPresses(behavData,recorded_TRs):
    runData = behavData['runData']
    left_key = runData['LEFT_PRESS'][0]
    leftPress = runData['leftPress']
    allLeftTRs = np.argwhere(leftPress[0,:] == 1)
    rightPress = runData['rightPress']
    right_key = runData['RIGHT_PRESS'][0]
    allRightTRs = np.argwhere(rightPress[0,:] == 1)
    n_stations = 7
    probe_response = []
    for st in np.arange(n_stations):
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

def makeCustomLegend(color1,color2,lw):
    custom_lines = [Line2D([0], [0], color=color1, lw=lw),
                Line2D([0], [0], color=color2, lw=lw)]
    return custom_lines

nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation()
recorded_TRs = createStationVector(stationDict)
nRuns = 4
subjects = np.array([25,26,28,29,30,31,32,33,35,36,37,38,39,41,40,42,43,44,45,46])
nSubs = len(subjects)
# get time course by run by subject
interpretations = {}
for s in np.arange(nSubs):
    interpretations[s] = getSubjectInterpretation(subjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']


all_nf_score = np.zeros((nSubs,nStations,nRuns))*np.nan
all_cheating_prob = np.zeros((nSubs,nStations,nRuns))*np.nan
all_cheating_prob_z = np.zeros((nSubs,nStations,nRuns))*np.nan
all_choices = np.zeros((nSubs,nStations,nRuns))*np.nan
behavioral_cheating = np.zeros((nSubs,))
behavioral_paranoid = np.zeros((nSubs,))
neural_cheating = np.zeros((nSubs,))
neural_paranoid = np.zeros((nSubs,))
subj_means = np.zeros((nSubs,2))
states = np.zeros((nSubs,nStations,nRuns))*np.nan

# first go through all subjects and calculate classifier cheating probability based on probe responses
for s in np.arange(nSubs):
    subjectNum = subjects[s]
    this_sub_cheating = np.empty((0,),int)
    this_sub_score = np.empty((0,),int)
    this_sub_response = []
    this_sub_interpretation = interpretations[s]
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
        this_sub_score = np.append(this_sub_score,nf_score[0,:])
        # now get prob answers
        b = getBehavData(subjectNum,runNum+1)
        probe_response = getProbeKeyPresses(b,recorded_TRs)
        C_press = [i for i in np.arange(len(probe_response))  if probe_response[i]  == 'CHEATING'] 
        P_press = [i for i in np.arange(len(probe_response))  if probe_response[i]  == 'PARANOID'] 
        all_choices[s,C_press,runNum] = 1
        all_choices[s,P_press,runNum] = 0
        this_sub_response = this_sub_response + probe_response
    ind_cheating = [i for i, x in enumerate(this_sub_response) if x == "CHEATING"]  
    ind_paranoid = [i for i, x in enumerate(this_sub_response) if x == "PARANOID"]  
    subj_means[s,0] = np.nanmean(this_sub_cheating[ind_cheating])
    subj_means[s,1] = np.nanmean(this_sub_cheating[ind_paranoid])

# classifier separation is the difference of p(cheating) for each response
classifier_separation = subj_means[:,0] - subj_means[:,1]
print('classifier separation')
print(classifier_separation)
print('n correct side')
all_correct_classification = np.argwhere(classifier_separation>0)[:,0]
print(len(all_correct_classification))
# specify plotting colors
# sort by "good" and "bad" subjects for each group
c_sep_P = classifier_separation[P_ind]
c_sep_C = classifier_separation[C_ind]
ord_P = np.argsort(c_sep_P)[::-1]  # this is best to worst
P_ind2 = np.array(P_ind)
ord_C = np.argsort(c_sep_C)[::-1]  # this is best to worst
C_ind2 = np.array(C_ind)
top_P_subj = P_ind2[ord_P[0:5]]
bottom_P_subj = P_ind2[ord_P[5:]]
top_C_subj = C_ind2[ord_C[0:5]]
bottom_C_subj = C_ind2[ord_C[5:]]
top_subj = np.concatenate((top_P_subj,top_C_subj))
bottom_subj = np.concatenate((bottom_P_subj,bottom_C_subj))
np.save('top_subj.npy', top_subj)
np.save('bottom_subj.npy', bottom_subj)

# now get the behavioral data 
projectDir = '/jukebox/norman/amennen/RT_prettymouth/data/laptopData/'
all_context_scores = np.zeros((nSubs,))
all_story_scores = np.zeros((nSubs,))
nR = 9
all_rating_scores = np.zeros((nSubs,nR))
for s in np.arange(nSubs):  
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

# change context scores to adjust for the one bad question
z=all_context_scores*12
new_context_scores = (z.copy() + 1)/11
all_correct_context = new_context_scores.copy()
all_correct_context[P_ind] = -1*new_context_scores[P_ind]
all_context_scores = new_context_scores
arthur_minus_lee = all_rating_scores[:,0] - all_rating_scores[:,1]
artur_minus_lee_cor = arthur_minus_lee.copy()
artur_minus_lee_cor[P_ind] = arthur_minus_lee[P_ind] * -1


# now get "correct" and "incorrect" probe responses based on group
all_choices_correct = np.zeros_like(all_choices)*np.nan # change into being correct and incorrect instead
for s in np.arange(nSubs):
    this_sub_interpretation = interpretations[s]
    for r in np.arange(nRuns):
        for st in np.arange(nStations):
            if all_choices[s,st,r] == 1: # if they chose the cheating response on that run
                if this_sub_interpretation == 'C':
                    correct = 1
                elif this_sub_interpretation == 'P':
                    correct = 0
            elif all_choices[s,st,r] == 0: # if they chose the paranoid response 
                if this_sub_interpretation == 'C':
                    correct = 0
                elif this_sub_interpretation == 'P':
                    correct = 1
            all_choices_correct[s,st,r] = correct


# (1) Plot probe responses by group first
fig = plotPosterStyle_multiplePTS(all_choices,subjects)
plt.subplot(1,4,1)
plt.ylim([0,1.15])
plt.xlim([-0.25,6.25])
cor = nStations*nRuns
for st in np.arange(nStations):
    x,y=nonNan(all_choices[C_ind,st,0],all_choices[P_ind,st,0],)
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (0, st)
        printStatsResults(text, t, p/2)
plt.yticks(np.array([0,0.5,1]), [ 'all paranoid','neutral','all cheating'],fontsize=20,rotation=0) 
plt.plot([-1,7],[0.5,0.5], '--', color='k')
plt.ylabel('p(probe response)', fontsize=25)
plt.title('run 1',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.xticks(fontsize=20)
plt.subplot(1,4,2)
plt.xlim([-0.25,6.25])
plt.ylim([0,1.15])
plt.yticks([])
plt.plot([-1,7],[0.5,0.5], '--', color='k')
for st in np.arange(nStations):
    x,y=nonNan(all_choices[C_ind,st,1],all_choices[P_ind,st,1],)
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (1, st)
        printStatsResults(text, t, p/2)
plt.title('run 2',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.xticks(fontsize=20)
plt.subplot(1,4,3)
plt.xlim([-0.25,6.25])
plt.ylim([0,1.15])
plt.plot([-1,7],[0.5,0.5], '--', color='k')
for st in np.arange(nStations):
    x,y=nonNan(all_choices[C_ind,st,2],all_choices[P_ind,st,2],)
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (2, st)
        printStatsResults(text, t, p/2)
plt.yticks([])
plt.title('run 3',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.xticks(fontsize=20)
plt.subplot(1,4,4)
plt.xlim([-0.25,6.25])
plt.plot([-1,7],[0.5,0.5], '--', color='k')
plt.ylim([0,1.15])
plt.title('run 4',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.xticks(fontsize=20)
for st in np.arange(nStations):
    x,y=nonNan(all_choices[C_ind,st,3],all_choices[P_ind,st,3],)
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (3, st)
        printStatsResults(text, t, p/2)
plt.yticks([])
plt.savefig('savedPlots_checked/choices_stations.pdf')
#plt.show()


# (2) plot probe responses, now divided by the top and bottom classifier performances (collapsing across interpretation groups)
fig,ax = plt.subplots(figsize=(20,9))
for d in np.arange(nRuns):
  plt.subplot(1,nRuns,d+1)
  sns.despine()
  nPoints = nStations
  plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_choices_correct[top_subj,:,d],axis=0),yerr=scipy.stats.sem(all_choices_correct[top_subj,:,d],axis=0),color='k',alpha=0.7,lw=3,label='top',fmt='-o',ms=10)
  plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_choices_correct[bottom_subj,:,d],axis=0),yerr=scipy.stats.sem(all_choices_correct[bottom_subj,:,d],axis=0),color='k',alpha=0.5,lw=3,label='bottom',fmt='--X',ms=10)
  plt.xlabel('station',fontsize=25)
  #plt.ylabel('area under -0.1')
  plt.xticks(np.arange(nPoints),fontsize=20)
plt.subplot(1,4,1)
plt.xlim([-0.25,6.25])
plt.plot([-1,7],[0.5,0.5], '--', color='k')
plt.ylim([0,1.15])
plt.ylabel('p(correct choice)',fontsize=25)
plt.yticks(np.array([0,0.5,1]),fontsize=20)
# test significance across all points and do Bonferroni correction
cor = nStations*nRuns
for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[top_subj,st,0],all_choices_correct[bottom_subj,st,0],)
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (0, st)
        printStatsResults(text, t, p/2)
plt.title('run 1',fontsize=30)
plt.subplot(1,4,2)
plt.ylim([0,1.15])
plt.xlim([-0.25,6.25])
plt.plot([-1,7],[0.5,0.5], '--', color='k')
plt.yticks([])
for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[top_subj,st,1],all_choices_correct[bottom_subj,st,1],)
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (1, st)
        printStatsResults(text, t, p/2)
plt.title('run 2',fontsize=30)
plt.subplot(1,4,3)
plt.yticks([])
plt.ylim([0,1.15])
plt.xlim([-0.25,6.25])
plt.plot([-1,7],[0.5,0.5], '--', color='k')
for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[top_subj,st,2],all_choices_correct[bottom_subj,st,2],)
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (2, st)
        printStatsResults(text, t, p/2)
plt.title('run 3',fontsize=30)
plt.subplot(1,4,4)
plt.title('run 4',fontsize=30)
plt.ylim([0,1.15])
plt.xlim([-0.25,6.25])
plt.plot([-1,7],[0.5,0.5], '--', color='k')
for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[top_subj,st,3],all_choices_correct[bottom_subj,st,3],)
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (3, st)
        printStatsResults(text, t, p/2)
plt.yticks([])
plt.savefig('savedPlots_checked/choices_stations_correct_incor.pdf')
#plt.show()

# (3) plot comprehension differences across top and bottom performing classified subjects, collapsing across groups
maxH = 1.1
# first make data frame
data = {}
data_vector = all_story_scores
subject_vector = np.arange(nSubs)
group_str = [''] * nSubs
for s in np.arange(nSubs):
	if s in top_subj:
		group_str[s] = 'best'
	elif s in bottom_subj:
		group_str[s] = 'worst'
data['comprehension'] = data_vector
data['subject'] = subject_vector
data['group'] = group_str
df = pd.DataFrame.from_dict(data)

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='group',y='comprehension',ci=68,linewidth=2.5,color='k', alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='group',y='comprehension',split=True,color='k',size=8)
maxH = 1.05
plt.ylim([0.5,1.05])
plt.ylabel('accuracy',fontsize=25)
plt.title('Comprehension scores',fontsize=30)
plt.xlabel('classifier accuracy group',fontsize=25)
x,y=nonNan(all_story_scores[top_subj],all_story_scores[bottom_subj])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,0,1,maxH,.05,0,text_above=r'B\neqW')
printStatsResults('comprehension diff', t, p)
plt.yticks(np.array([0.5,0.75,1]))
plt.savefig('savedPlots_checked/comprehension_score_cor_incor.pdf')
#plt.show()

# (4) plot interpretation differences across top and bottom performing classified subjects, collapsing across groups
data = {}
data_vector = all_correct_context
subject_vector = np.arange(nSubs)
group_str = [''] * nSubs
for s in np.arange(nSubs):
	if s in top_subj:
		group_str[s] = 'best'
	elif s in bottom_subj:
		group_str[s] = 'worst'
data['comprehension'] = data_vector
data['subject'] = subject_vector
data['group'] = group_str
df = pd.DataFrame.from_dict(data)
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='group',y='comprehension',ci=68,linewidth=2.5,color='k', alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='group',y='comprehension',split=True,color='k',size=8)
maxH = 1.05
plt.ylim([-1.1,1.5])
plt.ylabel('skewness to one interpretation',fontsize=25)
plt.title('Interpretation scores', fontsize=30)
plt.yticks(np.array([-1,0,1]), ['incorrect','neutral','correct'],fontsize=20,rotation=45) 
plt.xlabel('classifier accuracy group',fontsize=25)
x,y=nonNan(all_correct_context[top_subj],all_correct_context[bottom_subj])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='B>W')
printStatsResults('interpretation diff', t, p/2)
plt.plot([-2,2],[0,0], '--', color='k')
plt.yticks(np.array([-1, 0, 1]))
plt.savefig('savedPlots_checked/context_score_cor_incor.pdf')
#plt.show()


# (5) plot empathy differences for Arthur - Lee for the top and bottom groups, collapsing across interpretation groups
data = {}
data_vector = artur_minus_lee_cor
subject_vector = np.arange(nSubs)
group_str = [''] * nSubs
for s in np.arange(nSubs):
	if s in top_subj:
		group_str[s] = 'best'
	elif s in bottom_subj:
		group_str[s] = 'worst'
data['comprehension'] = data_vector
data['subject'] = subject_vector
data['group'] = group_str
df = pd.DataFrame.from_dict(data)

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='group',y='comprehension',ci=68,linewidth=2.5,color='k', alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='group',y='comprehension',split=True,color='k',size=8)
maxH = 5.1
#plt.ylim([-1.1,1.5])
plt.ylabel('correct empathy difference',fontsize=25)
plt.title('Arthus minus Lee', fontsize=30)
plt.xlabel('classifier accuracy group',fontsize=25)
plt.plot([-2,2],[0,0], '--', color='k')
x,y=nonNan(artur_minus_lee_cor[top_subj],artur_minus_lee_cor[bottom_subj])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='B>W')
printStatsResults('Arthur - Lee empathy diff ', t, p/2)
plt.yticks(np.array([-5, 0, 5]))
plt.savefig('savedPlots_checked/empathy_diff_cor_incor.pdf')
#plt.show()
