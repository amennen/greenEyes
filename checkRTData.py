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

def getRunAvgInc_all(stationProb,nStations):
    improvements = np.zeros((nStations-1,))
    for l in np.arange(nStations-1):
        improvements[l] = stationProb[l+1] - stationProb[l]
    mean_improvement = np.nanmean(improvements)
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
allSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,18]
nSubs = len(allSubjects)
nRuns = 4
nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation()
allStationProb = np.zeros((nRuns,nStations,nSubs))
allImprovements = np.zeros((nRuns,nSubs))
allImprovements_cor = np.zeros((nRuns,nSubs))
allImprovements_all = np.zeros((nRuns,nSubs))

allRewards = np.zeros((nSubs,))
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
        allImprovements_all[r,s] = getRunAvgInc_all(stationProb,nStations)
    subjectProb = allStationProb[:,:,s].copy()
    allRewards[s] = getReward(subjectProb)
# now for time points where prob was less than
subjAvg = np.nanmean(allImprovements,axis=0)
subjAvg_cor = np.nanmean(allImprovements_cor,axis=0)
subjAvg_all = np.nanmean(allImprovements_all,axis=0)

subjAvg_run_prob = np.mean(allStationProb,axis=0)
subjAvg_prob = np.mean(subjAvg_run_prob,axis=0)

allCheatingProb = allStationProb.copy()
for s in np.arange(nSubs):
    if interpretations[s] == 'P':
        allCheatingProb[:,:,s] = 1 - allCheatingProb[:,:,s]



run_avg_cheating_prob = np.mean(allCheatingProb,axis=0)
run = 0
run_avg_cheating_prob = allCheatingProb[run,:,:]
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

