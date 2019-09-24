# purpose: retrain and then get new version ready that uses mean and standard deviation of 17 subjects for new version


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


def getClassificationAndScore(data):
    correct_prob = data['correct_prob'][0,:]
    max_ratio = data['max_ratio'][0,:]
    return correct_prob,max_ratio

def getPatternsData(subjectNum,runNum):
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    ses_id = 'ses-{0:02d}'.format(2)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/patternsData_r{2}_*.mat'.format(bids_id,ses_id,runNum)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data

def getClassificationData(data,stationInd):
    stationStr = 'station' + str(stationInd)
    data_for_classification = data['dataForClassification']
    station_data = data_for_classification[stationStr]
    return station_data

def reclassifyAll(subjectNum,cfg,clf,nRuns=4,nStations=7):
    all_cheating_prob = np.zeros((nRuns,nStations))
    all_pred = np.zeros((nRuns,nStations))
    all_agree = np.zeros((nRuns,nStations))
    for r in np.arange(nRuns):
        for st in np.arange(nStations):
            all_cheating_prob[r,st],all_pred[r,st],all_agree[r,st] = reclassifyStation(subjectNum,r,st,cfg,clf)
    return all_cheating_prob,all_pred,all_agree

def loadClassifier(clf,stationInd):
    if clf == 1:
        clf_str = cfg.cluster.classifierDir + cfg.classifierNamePattern
    full_str = clf_str.format(stationInd)
    loaded_model = pickle.load(open(full_str, 'rb'))
    return loaded_model

def reclassifyStation(subjectNum,runNum,stationInd,cfg,clf):
    data = getPatternsData(subjectNum,runNum+1)
    station_data = getClassificationData(data,stationInd)
    if clf == 1:
        clf_str = cfg.cluster.classifierDir + cfg.classifierNamePattern # this is the logistic version because the config is updated
    elif clf == 2:
        clf_str = cfg.cluster.classifierDir + "UPPERRIGHT_stationInd_{}_ROI_1_AVGREMOVE_1_filter_0_k1_0_k2_25.sav" # first SVM version

    full_str = clf_str.format(stationInd)
    loaded_model = pickle.load(open(full_str, 'rb'))
    this_station_TRs = np.array(cfg.stationsDict[stationInd])
    n_station_TRs = len(this_station_TRs)
    thisStationData = station_data[:,this_station_TRs]
    dataForClassification_reshaped = np.reshape(thisStationData,(1,cfg.nVox*n_station_TRs))
    cheating_probability = loaded_model.predict_proba(dataForClassification_reshaped)[0][1]
    pred = loaded_model.predict(dataForClassification_reshaped)[0]
    if np.argmax(loaded_model.predict_proba(dataForClassification_reshaped)) == pred:
        agree = 1
    else:
        agree = 0
    return cheating_probability, pred, agree


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


def getRunAvgInc(stationProb,avg_removed,nStations):
    low_ind = np.argwhere(stationProb < 0.5)[:,0] 
    n_low = len(low_ind)
    if n_low >= 1:
        if np.all(nStations-1 != low_ind):  
            pass
        else:
            n_low = n_low - 1 # don't include last station
        improvements = np.zeros((n_low,))
        baseline = np.zeros((n_low,))
        next_t = np.zeros((n_low,))
        for l in np.arange(n_low):
            this_index = low_ind[l]
            improvements[l] = avg_removed[this_index+1] - avg_removed[this_index]
            baseline[l] = avg_removed[this_index]
            next_t[l] = avg_removed[this_index+1]
        mean_improvement = np.mean(improvements)
        mean_baseline = np.mean(baseline)
        mean_next = np.mean(next_t)
    else:
        mean_improvement = np.nan
        mean_baseline = np.nan
        mean_next = np.nan
    return mean_improvement, mean_baseline,mean_next

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


def getMaxScore(cheating_prob,interp,c_max,p_max):
    scores = np.zeros((9,))
    for st in np.arange(9):
        if interp == 'C':
            prob = cheating_prob[st]
            scores[st] = prob/c_max[st]
        elif interp == 'P':
            prob = 1 - cheating_prob[st]
            scores[st] = prob/p_max[st]
        if scores[st] > 1:
            scores[st] = 1
    return scores

def getRatioScore(cheating_prob,interp,c_max,p_max):
    c_min = 1 - p_max
    p_min = 1 - c_max
    scores = np.zeros((9,))
    for st in np.arange(9):
        if interp == 'C':
            prob = cheating_prob[st]
            scores[st] = (prob - c_min[st])/(c_max[st] - c_min[st])
        elif interp == 'P':
            prob = 1 - cheating_prob[st]
            scores[st] = (prob - p_min[st])/(p_max[st] - p_min[st])
    return scores

def getTransferredZ(cheating_prob,station,allmean,allstd):
    z_val = (cheating_prob - allmean[station])/allstd[station]
    z_transferred = (z_val/3) + 0.5
    # now correct if greater or less than 2 std above or below the mean
    if z_transferred > 1:
        z_transferred = 1
    if z_transferred < 0:
        z_transferred = 0
        print('here')
    return z_transferred

labels = ['cheating', 'paranoid']
nRuns = 4
nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation()


#allSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22]
#handle training and new subjects separately
trainingSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19]
#allSubjects = [21]
nSubs = len(trainingSubjects)
log2_cheating_prob = np.zeros((nRuns,nStations,nSubs))
orig_cheating_prob = np.zeros((nRuns,nStations,nSubs))

interpretations = {}
for s in np.arange(nSubs):
    interpretations[s] = getSubjectInterpretation(trainingSubjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
for s in np.arange(nSubs): 
    subjectNum = trainingSubjects[s]

    orig_cheating_prob[:,:,s],_,_ = reclassifyAll(subjectNum,cfg,2)
    log2_cheating_prob[:,:,s],_,_ = reclassifyAll(subjectNum,cfg,1)

all_context_scores = {}
all_story_scores = {}
nR = 9
all_rating_scores = np.zeros((nSubs,nR))
for s in np.arange(nSubs):  
    subject = trainingSubjects[s]
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

C_ind_interp = [sub for sub, interp in all_context_scores.items() if interp > 0]
P_ind_interp = [sub for sub, interp in all_context_scores.items() if interp < 0]

# plot means and std of each station
z = [log2_cheating_prob[:,:,s] for s in np.arange(nSubs)]   
all_data_by_stations = np.concatenate(z,axis=0)
overall_mean = np.mean(all_data_by_stations,axis=0)
overall_std = np.std(all_data_by_stations,axis=0,ddof=1)
# maybe instead take mean and std 
plt.figure()
plt.errorbar(np.arange(nStations),overall_mean,yerr=overall_std*2,color='r', label='2 std')
plt.errorbar(np.arange(nStations),overall_mean,yerr=overall_std*1.5,color='m', label='1.5 std')
plt.errorbar(np.arange(nStations),overall_mean,yerr=overall_std*1.25,color='c', label='1.25 std')
plt.errorbar(np.arange(nStations),overall_mean,yerr=overall_std, label='1 std')
plt.plot([0,nStations],[1,1],'r--',lw=2)
plt.plot([0,nStations],[0,0],'r--',lw=2)

plt.ylabel('p(cheating)')
plt.xlabel('station #')
plt.title('mean & std p(cheating); n = 17')
plt.legend()
plt.show()

#or take average of subjects first
new_scores = np.zeros((nRuns,nStations,nSubs))
for s in np.arange(nSubs):
    for r in np.arange(nRuns):
        this_subj_score = log2_cheating_prob[r,:,s]
        for st in np.arange(nStations):
            z_transferred = getTransferredZ(this_subj_score[st],st,overall_mean,overall_std)
            new_scores[r,st,s] = z_transferred

        # z_val = (this_subj_score - overall_mean)/overall_std
        # z_transferred = (z_val/3) + 0.5
        # # now correct if greater or less than 2 std above or below the mean
        # too_high = np.argwhere(z_transferred > 1)
        # if len(too_high) > 0:
        #     z_transferred[too_high] = 1
        # too_low = np.argwhere(z_transferred < 0)
        # if len(too_low) > 0:
        #     z_transferred[too_low] = 0
run_avg = np.mean(new_scores,axis=0)
# plot run average by subject
plt.figure()
for s in np.arange(nSubs):
    plt.plot(run_avg[:,s])
plt.plot(np.mean(run_avg,axis=1),color='k', lw=5)
plt.xlabel('Station')
plt.ylabel('p(cheating')
plt.show()

C_avg = np.mean(new_scores[:,:,C_ind],axis=2)
P_avg = np.mean(new_scores[:,:,P_ind],axis=2)
C_avg_interp = np.mean(new_scores[:,:,C_ind_interp],axis=2)
P_avg_interp = np.mean(new_scores[:,:,P_ind_interp],axis=2)
v = list(all_context_scores.values())
v2 = np.array(v)

iter_val = 1

# first prev subjects only
plt.figure()
for r in np.arange(nRuns):
    plt.subplot(2,2,r+1)
    title = 'run % i' % (r)
    plt.title(title)
    for s in np.arange(17):
        if interpretations[s] == 'C':
            color='r'
        else:
            color = 'g'
        plt.plot(new_scores[r,:,s], color=color,alpha=0.5)
    plt.plot(C_avg[r,:],color='r',label='cheating',lw=5)
    plt.plot(P_avg[r,:],color='g', label='paranoid',lw=5)
    plt.xlim([-1,nStations])
    plt.ylim([0,1])
#plt.legend()
plt.show()


plt.figure()
for r in np.arange(nRuns):
    plt.subplot(2,2,r+1)
    title = 'run % i' % (r)
    plt.title(title)
    for s in np.arange(17):
        if v2[s]>0:
            color='r'
        else:
            color = 'g'
        plt.plot(new_scores[r,:,s], color=color,alpha=0.5)
    plt.plot(C_avg_interp[r,:],color='r',label='cheating',lw=5)
    plt.plot(P_avg_interp[r,:],color='g', label='paranoid',lw=5)
    plt.xlim([-1,nStations])
    plt.ylim([0,1])
#plt.legend()
plt.show()
##########################################################
# save everything
# new_station_dict = {}
# new_nstations = 7
# for st in np.arange(new_nstations):
#     new_station_dict[st] = stationDict[st]
# np.save('stations_upper_right_nofilter_7st.npy', new_station_dict)


# new_overall_mean = overall_mean[0:7]
# new_overall_std = overall_std[0:7]
# np.savez('station_stats.npz', m=new_overall_mean,s=new_overall_std)
f = cfg.classifierDir + 'station_stats.npz'
station_info = np.load(f)
overall_mean = station_info['m']
overall_std = station_info['s']
############################################################
# NOW DO FOR NEW SUBJECTS
new_subj = [20,21,22]
nNew = len(new_subj)
new_log2_cheating_prob = np.zeros((nRuns,nStations,nNew))

new_interpretations = {}
for s in np.arange(nNew):
    new_interpretations[s] = getSubjectInterpretation(new_subj[s])
new_C_ind = [sub for sub, interp in new_interpretations.items() if interp == 'C']
new_P_ind = [sub for sub, interp in new_interpretations.items() if interp == 'P']
for s in np.arange(nNew): 
    subjectNum = new_subj[s]
    new_log2_cheating_prob[:,:,s],_,_ = reclassifyAll(subjectNum,cfg,1)

new_all_context_scores = {}
nR = 9
for s in np.arange(nNew):  
    subject = new_subj[s]
    context = getSubjectInterpretation(subject)
    bids_id = 'sub-{0:03d}'.format(subject)
    response_mat = '/jukebox/norman/amennen/RT_prettymouth/data/laptopData/' + bids_id + '/' + 'responses_scored.mat'
    z = scipy.io.loadmat(response_mat)
    context_score =  z['mean_context_score'][0][0]
    new_all_context_scores[s] = context_score

new_C_ind_interp = [sub for sub, interp in new_all_context_scores.items() if interp > 0]
new_P_ind_interp = [sub for sub, interp in new_all_context_scores.items() if interp < 0]

new_new_scores = np.zeros((nRuns,nStations,nNew))
for s in np.arange(nNew):
    for r in np.arange(nRuns):
        this_subj_score = new_log2_cheating_prob[r,:,s]
        for st in np.arange(nStations):
            z_transferred = getTransferredZ(this_subj_score[st],st,overall_mean,overall_std)
            new_new_scores[r,st,s] = z_transferred
run_avg = np.mean(new_new_scores,axis=0)
new_C_avg = np.mean(new_new_scores[:,:,new_C_ind],axis=2)
P_avg = np.mean(new_scores[:,:,P_ind],axis=2)
new_C_avg_interp = np.mean(new_new_scores[:,:,new_C_ind_interp],axis=2)
P_avg_interp = np.mean(new_scores[:,:,P_ind_interp],axis=2)
v = list(new_all_context_scores.values())
v2 = np.array(v)

iter_val = 1

# first prev subjects only
plt.figure()
for r in np.arange(nRuns):
    plt.subplot(2,2,r+1)
    title = 'run % i' % (r)
    plt.title(title)
    for s in np.arange(nNew):
        if new_interpretations[s] == 'C':
            color='r'
        else:
            color = 'g'
        plt.plot(new_new_scores[r,:,s], color=color,alpha=0.5)
    plt.plot(new_C_avg[r,:],color='r',label='cheating',lw=5)
    plt.plot(P_avg[r,:],color='g', label='paranoid',lw=5)
    plt.xlim([-1,nStations])
    plt.ylim([0,1])
#plt.legend()
plt.show()
