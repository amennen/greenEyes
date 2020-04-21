# here: z score all runs by subject
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
# correlate each run to the next to see how much they're changing vs. just listening to the story
# for each subject's p(cheating), zscore across all stations/runs

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

allRewards = np.zeros((nSubs,))
interpretations = {}
for s in np.arange(nSubs):
    interpretations[s] = getSubjectInterpretation(allSubjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
for s in np.arange(nSubs): 
    subjectNum = allSubjects[s]
    allStationProb[:,:,s] = getCorrectProbability(subjectNum)


allCheatingProb = allStationProb.copy()
for s in np.arange(nSubs):
    if interpretations[s] == 'P':
        allCheatingProb[:,:,s] = 1 - allStationProb[:,:,s]

# now for a given subject, zscore each station score across all runs
zscored_cheating_prob = np.zeros((nRuns,nStations,nSubs))

for s in np.arange(nSubs):
    subject_data = allCheatingProb[:,:,s]
    zscored_cheating_prob[:,:,s] = scipy.stats.zscore(subject_data,axis=0,ddof=1)




# none of these things led to anything
# run_avg = np.mean(zscored_cheating_prob,axis=1)
# C_mean = np.mean(run_avg[:,C_ind],axis=1)
# P_mean = np.mean(run_avg[:,P_ind],axis=1)
# plt.figure()
# for s in np.arange(nSubs):
#     interp = interpretations[s]
#     if interp == 'C':
#         index = 0 
#     elif interp == 'P':
#         index = 1
#     plt.plot(np.arange(nRuns),run_avg[:,s], '-', color=colors[index],alpha=0.5)
# plt.xlabel('Run number')
# plt.ylabel('p(cheating)')
# plt.plot(np.arange(nRuns),P_mean, '-', color=colors[1],lw=5)
# plt.plot(np.arange(nRuns),C_mean, '-', color=colors[0],lw=5)
# plt.show()

# # look at the last run zscored mean and context score
# plt.figure()
# for st in np.arange(nStations):
#     plt.subplot(3,3,st+1)
#     for s in np.arange(nSubs):
#         interp = interpretations[s]
#         if interp == 'C':
#             index = 0 
#         elif interp == 'P':
#             index = 1
#         plt.plot(zscored_cheating_prob[3,st,s],all_context_scores[s], '.', color=colors[index],ms=20,alpha=0.3)
#     r,p = scipy.stats.pearsonr(zscored_cheating_prob[3,st,:],all_context_scores)
#     plt.title('STATION %i: Corr,pval = %2.2f, %2.2f' % (st,r,p),fontsize=10)
#     if st + 1 >= 7:
#         plt.xlabel('Avg diff from mean',fontsize=10)
    
#     plt.ylabel('Context score',fontsize=10)
#     #plt.xlim([-.1,.1]) 
# plt.show()