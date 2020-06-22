# what Ken wanted--> when get no reward--if prob < 0.5, what's the average classifier change on the next station?
# calculate by run, calculate overall and see if that correlates with behavior


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

defaultConfig = os.path.join(os.getcwd(), 'conf/greenEyes_organized.toml')
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

def getReward(cfg,all_correct_prob):
    nRuns = int(cfg.totalNumberRuns)
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

def plotCorrectProbability(cfg,all_correct_prob):
    # now plot everything
    nRuns = int(cfg.totalNumberRuns)
    cmap = plt.get_cmap('Blues')
    color_idx = np.linspace(0, 1, nRuns)
    plt.figure()
    for r in np.arange(nRuns):
        label = 'Run %i' % r
        plt.plot(np.arange(cfg.nStations),all_correct_prob[r,:],color=plt.cm.cool(color_idx[r]), label = label )
    plt.plot([0,cfg.nStations], [0.5,0.5], '--', color='red', label='chance')
    plt.legend()
    plt.xlabel('Station')
    plt.ylabel('Correct prob')
    plt.ylim([0 ,1 ])
    plt.xlim([0,cfg.nStations-1])
    plt.show()

def getRunAvgInc(stationProb,nStations):
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
            improvements[l] = stationProb[this_index+1] - stationProb[this_index]
        mean_improvement = np.mean(improvements)
    else:
        mean_improvement = np.nan
    return mean_improvement

def getRunAvgInc_correct(stationProb,nStations):
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
            improvements[l] = stationProb[this_index+1] - stationProb[this_index]
        mean_improvement = np.mean(improvements)
    else:
        mean_improvement = np.nan
    return mean_improvement

colors = ['r', 'g']
labels = ['cheating', 'paranoid']
allSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,16]
nSubs = len(allSubjects)
nRuns = 4
nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation()
allStationProb = np.zeros((nRuns,nStations,nSubs))
allImprovements = np.zeros((nRuns,nSubs))
allImprovements_cor = np.zeros((nRuns,nSubs))
interpretations = {}
for s in np.arange(nSubs):
    interpretations[s] = getSubjectInterpretation(allSubjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
for s in np.arange(nSubs): 
    subjectNum = allSubjects[s]
    allStationProb[:,:,s] = getCorrectProbability(subjectNum)
    for r in np.arange(nRuns):
        stationProb = allStationProb[r,:,s]
        allImprovements[r,s] = getRunAvgInc(stationProb,nStations)
        allImprovements_cor[r,s] = getRunAvgInc_correct(stationProb,nStations)
# now for time points where prob was less than
subjAvg = np.nanmean(allImprovements,axis=0)
subjAvg_cor = np.nanmean(allImprovements_cor,axis=0)

# plot subject improvements over time to see if it changes by run

plt.figure()

plt.subplot(1,2,1)
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
plt.ylabel('p(cor) station(i+1) - station(i)')
plt.xticks(np.arange(nRuns) ,fontsize=10) 
plt.ylim([-.25,.25])
plt.xlabel('run #')

plt.title('Changes after being wrong')
plt.subplot(1,2,2)
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
plt.ylabel('p(cor) station(i+1) - station(i)')
plt.ylim([-.25,.25])
plt.xticks(np.arange(nRuns) ,fontsize=10) 
plt.xlabel('run #')
plt.title('Changes after being right')
plt.show()

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

plt.figure()
for s in np.arange(nSubs):
    subjectNum = allSubjects[s]
    interpretation = getSubjectInterpretation(subjectNum)
    if interpretation == 'C':
        index = 0 
        context_score =  all_context_scores[s]
    elif interpretation == 'P':
        index = 1
        context_score = -1* all_context_scores[s]
    plt.plot(subjAvg[s],context_score,'.',color=colors[index],ms=30,alpha=0.3)
#plt.title('Diff prob cheating R4 - R1')
#plt.xticks(np.array([0,1]), labels) 
plt.xlabel('Average station difference when wrong')
plt.ylabel('Correct interpretation score')
plt.show()

plt.figure()
for s in np.arange(nSubs):
    subjectNum = allSubjects[s]
    interpretation = getSubjectInterpretation(subjectNum)
    if interpretation == 'C':
        index = 0 
        context_score =  all_context_scores[s]
    elif interpretation == 'P':
        index = 1
        context_score = -1* all_context_scores[s]
    plt.plot(subjAvg_cor[s],context_score,'.',color=colors[index],ms=30,alpha=0.3)
#plt.title('Diff prob cheating R4 - R1')
#plt.xticks(np.array([0,1]), labels) 
plt.xlabel('Average station difference when right')
plt.ylabel('Correct interpretation score')
plt.show()

# create histogram of all probability
plt.figure()
plt.hist(allStationProb.flatten())
plt.xlabel('Probability of being correct')
plt.ylabel('Counts')
plt.show()

cheating_prob = allStationProb.copy()
cheating_prob[:,:,P_ind] = 1 - allStationProb[:,:,P_ind]
plt.figure()
plt.hist(cheating_prob.flatten())
plt.xlabel('Probability(cheating interpretation)')
plt.ylabel('Counts')
plt.show()