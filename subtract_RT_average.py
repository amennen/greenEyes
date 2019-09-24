# purpose: check over real-time classification
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
import scipy
sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
# when running not in file: sys.path.append(os.getcwd())
#WHEN TESTING
#sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
from rtCommon.utils import loadConfigFile, dateStr30, DebugLevels, writeFile, loadMatFile
from rtCommon.readDicom import readDicomFromBuffer
from rtCommon.fileClient import FileInterface
import rtCommon.webClientUtils as wcutils
from rtCommon.structDict import StructDict
import rtCommon.dicomNiftiHandler as dnh
import greenEyes
params = {'legend.fontsize': 'large',
          'figure.figsize': (5, 3),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
font = {'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
defaultConfig = os.path.join(os.getcwd(), 'conf/greenEyes_organized.toml')
cfg = loadConfigFile(defaultConfig)
params = StructDict({'config':defaultConfig, 'runs': '1', 'scans': '9', 'webpipe': 'None', 'webfilesremote': False})
cfg = greenEyes.initializeGreenEyes(defaultConfig,params)
# date doesn't have to be right, but just make sure subject number, session number, computers are correct


def getPatternsData(subjectNum,runNum):
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    ses_id = 'ses-{0:02d}'.format(2)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/patternsData_r{2}_*.mat'.format(bids_id,ses_id,runNum)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data

def getCorrectProbability(subjectNum,nRuns=4,nStations=9):
    all_correct_prob = np.zeros((nRuns,nStations))
    for r in np.arange(nRuns):
        run_data = getPatternsData(subjectNum,r+1) # for matlab indexing
        all_correct_prob[r,:] = run_data.correct_prob[0,:]
    return all_correct_prob

def getSubjectInterpretation(subjectNum):
    # load interpretation file and get it
    # will be saved in subject full day path
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    ses_id = 'ses-{0:02d}'.format(2)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/' + bids_id + '/' + ses_id + '/' + bids_id + '_' + ses_id + '_' + 'intepretation.txt'
    z = open(filename, "r")
    temp_interpretation = z.read()
    if 'C' in temp_interpretation:
        interpretation = 'C'
    elif 'P' in temp_interpretation:
        interpretation = 'P'
    return interpretation

def getStationInformation(config='conf/greenEyes_organized.toml'):
    allinfo = {}
    cfg = loadConfigFile(config)
    station_FN = cfg.cluster.classifierDir + '/' + cfg.stationDict
    stationDict = np.load(station_FN,allow_pickle=True).item()
    nStations = len(stationDict)
    last_tr_in_station = np.zeros((nStations,))
    allTR = list(stationDict.values())
    all_station_TRs = [item for sublist in allTR for item in sublist]
    for st in np.arange(nStations):
        last_tr_in_station[st] = stationDict[st][-1]
    return nStations, stationDict, last_tr_in_station, all_station_TRs

def getBehavData(subjectNum,runNum):
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    ses_id = 'ses-{0:02d}'.format(2)
    run_id = 'run-{0:03d}'.format(runNum)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/{2}/behavior_run{3}_*.mat'.format(bids_id,ses_id,run_id,runNum)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data

def loadClassifier(cfg,station):
    thisClassifierFileName = cfg.classifierDir + cfg.classifierNamePattern.format(station)
    loaded_model = pickle.load(open(thisClassifierFileName, 'rb'))
    return loaded_model

def getReward(all_correct_prob,cfg='conf/greenEyes_organized.toml',nRuns=4):
    run_avg = np.zeros((nRuns,))
    for r in np.arange(nRuns):
        this_run = all_correct_prob[r,:]
        this_run[this_run < 0.5] = 0
        run_avg[r] = np.nanmean(this_run)
    print('all runs were:')
    print(run_avg)
    total_money_reward = np.nansum(run_avg)*5
    rewardStr = 'TOTAL REWARD: %5.2f\n' % total_money_reward
    print(rewardStr)
    return total_money_reward

def getRunAvgInc(stationProb,avg_removed,nStations):
    low_ind = np.argwhere(stationProb < 0.5)[:,0] 
    n_low = len(low_ind)
    if n_low >= 1:
        if np.all(nStations-1 != low_ind):  
            pass
        else:
            n_low = n_low - 1 # don't include last station
        improvements = np.zeros((n_low,))
        for l in np.arange(n_low):
            this_index = low_ind[l]
            improvements[l] = avg_removed[this_index+1] - avg_removed[this_index]
        mean_improvement = np.mean(improvements)
    else:
        mean_improvement = np.nan
    return mean_improvement

def getRunAvgInc_all(stationProb,avg_removed,nStations):
    improvements = np.zeros((nStations-1,))
    for l in np.arange(nStations-1):
        improvements[l] = avg_removed[l+1] - avg_removed[l]
    mean_improvement = np.nanmean(improvements)
    return mean_improvement

def getRunAvgInc_correct(stationProb,avg_removed,nStations):
    low_ind = np.argwhere(stationProb > 0.5)[:,0] 
    n_low = len(low_ind)
    if n_low >= 1:
        if np.all(nStations-1 != low_ind):  
            pass
        else:
            n_low = n_low - 1 # don't include last station
        improvements = np.zeros((n_low,))
        for l in np.arange(n_low):
            this_index = low_ind[l]
            improvements[l] = avg_removed[this_index+1] - avg_removed[this_index]
        mean_improvement = np.mean(improvements)
    else:
        mean_improvement = np.nan
    return mean_improvement

def getNumberofTRs(stationsDict,st):
    this_station_TRs = np.array(stationsDict[st])
    n_station_TRs = len(this_station_TRs)
    return n_station_TRs


def getDifferenceByStation(avg_removed,st):
    # nruns x nstations x nsubs
    beginning_prob = avg_removed[0,st,:]
    last_prob = avg_removed[3,st,:]
    change_in_prob = last_prob - beginning_prob
    return change_in_prob


colors = ['r', 'g']
labels = ['cheating', 'paranoid']
allSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19]
nSubs = len(allSubjects)
nRuns = 4
nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation()
allStationProb = np.zeros((nRuns,nStations,nSubs))
allImprovements = np.zeros((nRuns,nSubs))
allImprovements_cor = np.zeros((nRuns,nSubs))
allImprovements_all = np.zeros((nRuns,nSubs))
allCheatingProb = np.zeros((nRuns,nStations,nSubs))

allRewards = np.zeros((nSubs,))
interpretations = {}
for s in np.arange(nSubs):
    interpretations[s] = getSubjectInterpretation(allSubjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
for s in np.arange(nSubs): 
    subjectNum = allSubjects[s]
    allStationProb[:,:,s] = getCorrectProbability(subjectNum)
    if interpretations[s] == 'C':
        allCheatingProb[:,:,s] = 1 - getCorrectProbability(subjectNum)
    else: 
        allCheatingProb[:,:,s] = getCorrectProbability(subjectNum)

all_average_cheating_prob = np.mean(allCheatingProb,axis=2)
# AVERAGE OVER ALL RUNS TOO!!!
final_all_average_cheating_prob = np.mean(all_average_cheating_prob,axis=0)
matrix_all_average_cheating_prob = np.repeat(final_all_average_cheating_prob[:,np.newaxis],nSubs,axis=1)
avg_removed = allCheatingProb - matrix_all_average_cheating_prob

# now first plot average subtracted by group
run_avg_cheating_prob = np.mean(avg_removed,axis=0)
#run = 0
#run_avg_cheating_prob = allCheatingProb[run,:,:]
C_mean = np.mean(run_avg_cheating_prob[:,C_ind],axis=1)
P_mean = np.mean(run_avg_cheating_prob[:,P_ind],axis=1)

plt.figure()
for s in np.arange(nSubs):
    interp = interpretations[s]
    if interp == 'C':
        index = 0 
    elif interp == 'P':
        index = 1
    plt.plot(np.arange(nStations),run_avg_cheating_prob[:,s], '-', color=colors[index],alpha=0.5)
plt.xlabel('Station number')
plt.ylabel('p(cheating)')
plt.plot(np.arange(nStations),P_mean, '-', color=colors[1],lw=5)
plt.plot(np.arange(nStations),C_mean, '-', color=colors[0],lw=5)
plt.show()

# plot for each run
plt.figure()
for r in np.arange(nRuns):
    plt.subplot(2,2,r+1)
    C_mean = np.mean(avg_removed[r,:,C_ind],axis=0)
    P_mean = np.mean(avg_removed[r,:,P_ind],axis=0)
    for s in np.arange(nSubs):
        interp = interpretations[s]
        if interp == 'C':
            index = 0 
        elif interp == 'P':
            index = 1
        plt.plot(np.arange(nStations),avg_removed[r,:,s], '-', color=colors[index],alpha=0.5)
    plt.xlabel('Station number',fontsize=10)
    plt.ylabel('p(cheating)')
    plt.title('Run % i' % r)
    plt.ylim([-0.2,0.6])
    plt.plot(np.arange(nStations),P_mean, '-', color=colors[1],lw=5)
    plt.plot(np.arange(nStations),C_mean, '-', color=colors[0],lw=5)
plt.show()



# doesn't make sense to subtract one because now it's the difference it's no longer probability!!
# avg_removed_correct = avg_removed.copy()
# for s in np.arange(nSubs):
#     if interpretations[s] == 'P':
#         avg_removed_correct[:,:,s] = 1 - avg_removed[:,:,s]
## NOW CALCULATE IMPROVEMENTS WITH AVERAGE REMOVED
for s in np.arange(nSubs):
    for r in np.arange(nRuns):
        stationProb = allStationProb[r,:,s]
        allImprovements[r,s] = getRunAvgInc(stationProb,avg_removed[r,:,s],nStations)
        allImprovements_cor[r,s] = getRunAvgInc_correct(stationProb,avg_removed[r,:,s],nStations)
        allImprovements_all[r,s] = getRunAvgInc_all(stationProb,avg_removed[r,:,s],nStations)

plt.figure()
plt.subplot(1,3,1)
mean_p = np.nanmean(allImprovements[:,P_ind],axis=1)
mean_c = np.nanmean(allImprovements[:,C_ind],axis=1)
for s in np.arange(nSubs):
    interp = interpretations[s]
    if interp == 'C':
        index = 0 
    elif interp == 'P':
        index = 1
    plt.plot(np.arange(nRuns),allImprovements[:,s], '.', color=colors[index],alpha=0.5)
    plt.plot(np.arange(nRuns),allImprovements[:,s], '-', color=colors[index],alpha=0.5)
plt.plot(np.arange(nRuns),mean_p, '-', color=colors[1],lw=5)
plt.plot(np.arange(nRuns),mean_c, '-', color=colors[0],lw=5)
plt.ylabel('p(cheating) station(i+1) - station(i)')
plt.xticks(np.arange(nRuns) ,fontsize=10) 
plt.ylim([-.25,.25])
plt.title('Changes after being wrong')
plt.subplot(1,3,2)
mean_p = np.nanmean(allImprovements_cor[:,P_ind],axis=1)
mean_c = np.nanmean(allImprovements_cor[:,C_ind],axis=1)
for s in np.arange(nSubs):
    interp = interpretations[s]
    if interp == 'C':
        index = 0 
    elif interp == 'P':
        index = 1
    plt.plot(np.arange(nRuns),allImprovements_cor[:,s], '.', color=colors[index],alpha=0.5)
    plt.plot(np.arange(nRuns),allImprovements_cor[:,s], '-', color=colors[index],alpha=0.5)
plt.plot(np.arange(nRuns),mean_p, '-', color=colors[1],lw=5)
plt.plot(np.arange(nRuns),mean_c, '-', color=colors[0],lw=5)
plt.ylabel('p(cheating) station(i+1) - station(i)')
plt.xticks(np.arange(nRuns) ,fontsize=10) 
plt.ylim([-.25,.25])
plt.title('Changes after being right')
plt.subplot(1,3,3)
mean_p = np.nanmean(allImprovements_all[:,P_ind],axis=1)
mean_c = np.nanmean(allImprovements_all[:,C_ind],axis=1)
for s in np.arange(nSubs):
    interp = interpretations[s]
    if interp == 'C':
        index = 0 
    elif interp == 'P':
        index = 1
    plt.plot(np.arange(nRuns),allImprovements_all[:,s], '.', color=colors[index],alpha=0.5)
    plt.plot(np.arange(nRuns),allImprovements_all[:,s], '-', color=colors[index],alpha=0.5)
plt.plot(np.arange(nRuns),mean_p, '-', color=colors[1],lw=5)
plt.plot(np.arange(nRuns),mean_c, '-', color=colors[0],lw=5)
plt.ylabel('p(cheating) station(i+1) - station(i)')
plt.xticks(np.arange(nRuns) ,fontsize=10) 
plt.ylim([-.25,.25])
plt.title('Changes after all stations')
plt.show()
#    subjectProb = allStationProb[:,:,s].copy()
#    allRewards[s] = getReward(subjectProb)
# now for time points where prob was less than
subjAvg = np.nanmean(allImprovements,axis=0)
subjAvg_cor = np.nanmean(allImprovements_cor,axis=0)
subjAvg_all = np.nanmean(allImprovements_all,axis=0)

subjAvg_run_prob = np.mean(allStationProb,axis=0)
subjAvg_prob = np.mean(subjAvg_run_prob,axis=0)


all_context_scores = np.zeros((nSubs,))
all_story_scores = np.zeros((nSubs,))
nR = 9
all_rating_scores = np.zeros((nSubs,nR))
for s in np.arange(nSubs):  
    subject = allSubjects[s]
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

# plot context score x and run diff on y
correct_context_scores = all_context_scores.copy()
for s in np.arange(nSubs):
    subjectNum = allSubjects[s]
    interpretation = getSubjectInterpretation(subjectNum)
    if interpretation == 'C':
        correct_context_scores[s] =  all_context_scores[s]
    elif interpretation == 'P':
        correct_context_scores[s] = -1* all_context_scores[s]


#### NEW: ADD STATION ACCURACY INFORMATION ###
all_station_correlations = np.zeros((nStations,))
station_indices = np.array([7, 6, 8, 3 ,1, 5, 4, 2, 0])
station_length = np.zeros((nStations,))
allstations = np.arange(nStations)
#[x for x if allstations[x] ==  station_indices]
station_accuracy = np.array([0.778,  0.6885, 0.662,  0.645 , 0.607 , 0.6055 ,0.6055, 0.5955, 0.5715])
plt.figure()
for st in np.arange(nStations):
    plt.subplot(3,3,st+1)
    for s in np.arange(nSubs):
        interp = interpretations[s]
        if interp == 'C':
            index = 0 
        elif interp == 'P':
            index = 1
        plt.plot(run_avg_cheating_prob[st,s],all_context_scores[s], '.', color=colors[index],ms=20,alpha=0.3)
    r,p = scipy.stats.pearsonr(run_avg_cheating_prob[st,:],all_context_scores)
    station_index = np.argwhere(station_indices == st)[0][0]
    all_station_correlations[station_index] = r
    station_length[station_index] = getNumberofTRs(stationDict,st)
    plt.title('STATION %i: Corr = %2.2f; p= %2.2f' % (st,r,p),fontsize=10)
    plt.xticks(fontsize=5)
    if st + 1 >= 7:
        plt.xlabel('Avg diff from mean',fontsize=10)
    
    plt.ylabel('Context score',fontsize=10)
    plt.xlim([-.1,.1]) 
plt.show()


## NEW 8/14: look up improvements for station 7
st = 7
all_changes_station = getDifferenceByStation(avg_removed,st)
plt.figure()
for s in np.arange(nSubs):
    interp = interpretations[s]
    if interp == 'C':
        index = 0 
    elif interp == 'P':
        index = 1
    plt.plot(all_changes_station[s],all_context_scores[s], '.', ms = 40,color=colors[index],alpha=0.3)
plt.xlabel('Changes run 4 - run 1 p(cheating) station 7')
plt.ylabel('Context intrepretation score')
#plt.title('Corr = %2.2f; p = %2.2f' % (r,p))
plt.show()

# now see for each if it matters
plt.figure()
plt.subplot(1,2,1)
for s in np.arange(nSubs):
    interp = interpretations[s]
    if interp == 'C':
        index = 0 
    elif interp == 'P':
        index = 1
    plt.plot(avg_removed[0,st,s],all_context_scores[s], '.', ms = 40,color=colors[index],alpha=0.3)
plt.xlabel('first run p(cheating) station 7')
plt.ylabel('Context intrepretation score')
plt.subplot(1,2,2)
for s in np.arange(nSubs):
    interp = interpretations[s]
    if interp == 'C':
        index = 0 
    elif interp == 'P':
        index = 1
    plt.plot(avg_removed[3,st,s],all_context_scores[s], '.', ms = 40,color=colors[index],alpha=0.3)
plt.xlabel('last run p(cheating) station 7')
plt.ylabel('Context intrepretation score')
#plt.title('Corr = %2.2f; p = %2.2f' % (r,p))
plt.show()

# now plot correlation by classifier accuracy
r,p = scipy.stats.pearsonr(station_accuracy,all_station_correlations)

plt.figure()
plt.plot(station_accuracy,all_station_correlations, '.', ms = 40)
plt.xlabel('Offline station accuracy')
plt.ylabel('Correlation: classification to context interpretation')
plt.title('Corr = %2.2f; p = %2.2f' % (r,p))
plt.show()

r,p = scipy.stats.pearsonr(station_length,station_accuracy)

plt.figure()
plt.plot(station_length,station_accuracy, '.', ms = 40)
plt.ylabel('Offline station accuracy')
plt.xlabel('Station length')
plt.title('Corr = %2.2f; p = %2.2f' % (r,p))

plt.show()

r,p = scipy.stats.pearsonr(station_length,all_station_correlations)

plt.figure()
plt.plot(station_length,all_station_correlations, '.', ms = 40)
plt.xlabel('Station length')
plt.ylabel('Correlation: classification to context interpretation')
plt.title('Corr = %2.2f; p = %2.2f' % (r,p))

plt.show()

scipy.stats.pearsonr(station_length,all_station_correlations)

## by subject see if there's a trend
run_mean = np.mean(avg_removed,axis=0)
station_mean = np.mean(run_mean,axis=0)
plt.figure()
for s in np.arange(nSubs):
    interp = interpretations[s]
    if interp == 'C':
        index = 0 
    elif interp == 'P':
        index = 1
    plt.plot(station_mean[s],all_context_scores[s], '.', color=colors[index],ms=20,alpha=0.3)
r,p = scipy.stats.pearsonr(station_mean,all_context_scores)
plt.title('Corr,pval = %2.2f, %2.2f' % (r,p),fontsize=10)

plt.xlabel('Avg diff from mean',fontsize=10)

plt.ylabel('Context score',fontsize=10)
#plt.xlim([-.1,.1]) 
plt.show()





## now check on the constants
station_prob_a = np.zeros((nStations,))
station_prob_b = np.zeros((nStations,))
station_intercept = np.zeros((nStations,))
for st in np.arange(nStations):
    clf = loadClassifier(cfg,st)
    station_prob_a[st] = clf.probA_
    station_prob_b[st] = clf.probB_
    station_intercept[st] = clf.intercept_[0]

plt.figure()
plt.subplot(1,3,1)
plt.title('A')
plt.plot(station_prob_a)
plt.xlabel('Station number')
plt.subplot(1,3,2)
plt.title('B')
plt.plot(station_prob_b)
plt.xlabel('Station number')
plt.subplot(1,3,3)
plt.plot(station_intercept)
plt.xlabel('Station number')
plt.title('Decision function intercept')
plt.show()

