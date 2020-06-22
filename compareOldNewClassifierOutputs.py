# Purpose: test old data with new classifier and compare outputs

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
import rtCommon.projectUtils as projUtils
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
defaultConfig = os.path.join(os.getcwd(), 'conf/greenEyes_cluster.toml')
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

def reclassifyAll(subjectNum,cfg,clf,nRuns=4,nStations=9):
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
        clf_str = cfg.cluster.classifierDir + cfg.classifierNamePattern
    elif clf == 2:
        clf_str = cfg.cluster.classifierDir + "UPPERRIGHT_stationInd_{}_ROI_1_AVGREMOVE_1_filter_0_k1_0_k2_25.sav"
    elif clf == 3:
        clf_str = cfg.cluster.classifierDir + 'TRAIN_TEST_BOOTSTRAP_UPPERRIGHT_stationInd_{}_ROI_1_AVGREMOVE_1_filter_0_k1_0_k2_25.sav'
    elif clf == 4:
        clf_str = cfg.cluster.classifierDir + 'leave_one_out_UPPERRIGHT_stationInd_{}_ROI_1_AVGREMOVE_1_filter_0_k1_0_k2_25.sav'
    elif clf == 5:
        clf_str = cfg.cluster.classifierDir + 'TRAIN_TEST_BOOTSTRAP_KEEP_ALL_UPPERRIGHT_stationInd_{}_ROI_1_AVGREMOVE_1_filter_0_k1_0_k2_25.sav'

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



a = np.load('all_station_log_info.npz')
c_max = a['max']
p_max = 1 - a['min']

labels = ['cheating', 'paranoid']
#allSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22]
#allSubjects = [21]
nSubs = len(allSubjects)
nRuns = 4
nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation()
allStationProb = np.zeros((nRuns,nStations,nSubs))
allImprovements = np.zeros((nRuns,nSubs))
allImprovements_cor = np.zeros((nRuns,nSubs))
allImprovements_all = np.zeros((nRuns,nSubs))
allCheatingProb = np.zeros((nRuns,nStations,nSubs))
log2_cheating_prob = np.zeros((nRuns,nStations,nSubs))
orig_cheating_prob = np.zeros((nRuns,nStations,nSubs))
bs_cheating_prob = np.zeros((nRuns,nStations,nSubs))
log2_pred = np.zeros((nRuns,nStations,nSubs))
orig_pred = np.zeros((nRuns,nStations,nSubs))
orig_agree = np.zeros((nRuns,nStations,nSubs))
bs_pred = np.zeros((nRuns,nStations,nSubs))
bs_agree = np.zeros((nRuns,nStations,nSubs))
log2_agree = np.zeros((nRuns,nStations,nSubs))
allRewards = np.zeros((nSubs,))
allMaxScore = np.zeros((nRuns,nStations,nSubs))
allRatioScore = np.zeros((nRuns,nStations,nSubs))
interpretations = {}
for s in np.arange(nSubs):
    interpretations[s] = getSubjectInterpretation(allSubjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
for s in np.arange(nSubs): 
    subjectNum = allSubjects[s]
    # allStationProb[:,:,s] = getCorrectProbability(subjectNum)
    # if interpretations[s] == 'P':
    #     allCheatingProb[:,:,s] = 1 - getCorrectProbability(subjectNum)
    # else: 
    #     allCheatingProb[:,:,s] = getCorrectProbability(subjectNum)
    orig_cheating_prob[:,:,s],orig_pred[:,:,s],orig_agree[:,:,s] = reclassifyAll(subjectNum,cfg,2)
    log2_cheating_prob[:,:,s],log2_pred[:,:,s],log2_agree[:,:,s] = reclassifyAll(subjectNum,cfg,1)
    bs_cheating_prob[:,:,s],bs_pred[:,:,s],bs_agree[:,:,s] = reclassifyAll(subjectNum,cfg,3)
    for r in np.arange(nRuns):
        if subjectNum != 21:   
            allMaxScore[r,:,s] = getMaxScore(log2_cheating_prob[r,:,s],interpretations[s],c_max,p_max)
            allRatioScore[r,:,s] = getRatioScore(log2_cheating_prob[r,:,s],interpretations[s],c_max,p_max)

        else:
            data = getPatternsData(21,r+1)
            _,allMaxScore[r,:,s] = getClassificationAndScore(data)
            allRatioScore[r,:,s] = getRatioScore(log2_cheating_prob[r,:,s],interpretations[s],c_max,p_max)




all_context_scores = {}
all_story_scores = {}
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


C_ind_interp = [sub for sub, interp in all_context_scores.items() if interp > 0]
P_ind_interp = [sub for sub, interp in all_context_scores.items() if interp < 0]

################################## NEW 9.6: compute z-score with old subjects #############################


all_log_old_values = log2_cheating_prob[:,:,0:17]
# want the mean value per station and the std per station
z = [all_log_old_values[:,:,s] for s in np.arange(17)]   
all_data_by_stations = np.concatenate(z,axis=0)
overall_mean = np.mean(all_data_by_stations,axis=0)
overall_std = np.std(all_data_by_stations,axis=0,ddof=1)
# maybe instead take mean and std 
plt.figure()
plt.errorbar(np.arange(nStations),overall_mean,yerr=overall_std)
plt.ylabel('p(cheating)')
plt.xlabel('station #')
plt.title('mean & std p(cheating); n = 17')
plt.show()

# now for each of the new subjects, show what their actual scores would be and then their adjusted scores
new_subj = [17,18,19] # in terms of index
new_scores = np.zeros((nRuns,nStations,nSubs))
for s in np.arange(nSubs):
    for r in np.arange(nRuns):
        this_subj_score = log2_cheating_prob[r,:,s]
        z_val = (this_subj_score - overall_mean)/overall_std
        z_transferred = (z_val/4) + 0.5
        # now correct if greater or less than 2 std above or below the mean
        too_high = np.argwhere(z_transferred > 1)
        if len(too_high) > 0:
            z_transferred[too_high] = 1
        too_low = np.argwhere(z_transferred < 0)
        if len(too_low) > 0:
            z_transferred[too_low] = 0
        new_scores[r,:,s] = z_transferred
# first do new subjects
colors=['k','b','m','r']
iter_val = 1
plt.figure()
for r in np.arange(nRuns):
    for s in np.arange(3):
        plt.subplot(4,3,iter_val)
        tile = 's %i run % i' % (s,r)
        plt.title(tile)
        label='og run %i' % r
        plt.plot(log2_cheating_prob[r,:,new_subj[s]], label=label,color=colors[r])
        label='z run %i' % r
        plt.plot(new_scores[r,:,new_subj[s]],'--', label=label ,color=colors[r])
        iter_val+=1
plt.show()


# now show what it would have been for that subject
# average over runs
run_avg = np.mean(new_scores[:,:,0:17],axis=0)
C_avg = np.mean(new_scores[:,:,C_ind[0:9]],axis=2)
P_avg = np.mean(new_scores[:,:,P_ind],axis=2)
C_avg_interp = np.mean(new_scores[:,:,C_ind_interp[0:-1]],axis=2)
P_avg_interp = np.mean(new_scores[:,:,P_ind_interp[0:-2]],axis=2)
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
#plt.legend()
plt.show()

run_avg = np.mean(new_scores[:,:,17:20],axis=0)
C_avg = np.mean(new_scores[:,:,C_ind[9:12]],axis=2)
#P_avg = np.mean(new_scores[:,:,P_ind],axis=2)
# next new subjects only
plt.figure()
for r in np.arange(nRuns):
    plt.subplot(2,2,r+1)
    title = 'run % i' % (r)
    plt.title(title)
    for s in np.arange(17,20):
        if interpretations[s] == 'C':
            color='r'
        else:
            color = 'g'
        plt.plot(new_scores[r,:,s], color=color,alpha=0.5)
    plt.plot(C_avg[r,:],color='r',label='cheating',lw=5)
    #plt.plot(P_avg[r,:],color='g', label='paranoid',lw=5)
    plt.xlim([-1,nStations])
#plt.legend()
plt.show()

# want average per group per run over all station
station_avg = np.mean(new_scores[:,0:7,0:17],axis=1)
C_mean = np.mean(station_avg[:,C_ind[0:9]],axis=1)
P_mean = np.mean(station_avg[:,P_ind],axis=1)
C_std = np.std(station_avg[:,C_ind[0:9]],axis=1,ddof=1)/np.sqrt(len(C_ind[0:9])-1)
P_std = np.std(station_avg[:,P_ind],axis=1,ddof=1)/np.sqrt(len(P_ind)-1)
C_mean_interp = np.mean(station_avg[:,C_ind_interp[0:-1]],axis=1)
P_mean_interp = np.mean(station_avg[:,P_ind_interp[0:-2]],axis=1)
C_std_interp = np.std(station_avg[:,C_ind_interp[0:-1]],axis=1,ddof=1)/np.sqrt(len(C_ind_interp[0:-1])-1)
P_std_interp = np.std(station_avg[:,P_ind_interp[0:-2]],axis=1,ddof=1)/np.sqrt(len(P_ind_interp[0:-2])-1)


x = np.arange(nRuns)  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, C_mean,  width,label='C',yerr=C_std,color='r')
rects2 = ax.bar(x + width/2, P_mean, width,label='P',yerr=P_std,color='g')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('p(cheating)')
ax.set_title('p(cheating) by group')
ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend(loc=1)
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
plt.ylim([0.4,0.65])
plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, C_mean_interp,  width,label='C',yerr=C_std_interp,color='r')
rects2 = ax.bar(x + width/2, P_mean_interp, width,label='P',yerr=P_std_interp,color='g')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('p(cheating)')
ax.set_title('p(cheating) by group')
ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax.legend(loc=1)
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
plt.ylim([0.4,0.65])
plt.show()


# now do it by group eventual interpretation


plt.figure()
for r in np.arange(nRuns):
    plt.subplot(2,2,r+1)
    title = 'run % i' % (r)
    plt.title(title)
    for s in np.arange(nSubs):
        if v2[s] > 0:
            color='r'
        else:
            color = 'g'
        plt.plot(new_scores[r,:,s], color=color,alpha=0.5)
    plt.plot(C_avg_interp[r,:],color='r',label='cheating',lw=5)
    plt.plot(P_avg_interp[r,:],color='g', label='paranoid',lw=5)
#plt.legend()
plt.show()

# plot in bar graphs

# to see how variable--take original means across all people  and the upddated version

subject_avg = np.mean(log2_cheating_prob,axis=0)
all_subject_avg = np.mean(subject_avg,axis=1)
matrix_all_average = np.repeat(all_subject_avg[:,np.newaxis],nSubs,axis=1)
diff_from_avg = subject_avg - matrix_all_average


################################## NEW 8/28: PLOTTING BY GROUP #############################

colors = ['r', 'g']
plt.figure()
for r in np.arange(nRuns):
    plt.subplot(2,2,r+1)
    for s in np.arange(nSubs):
        if interpretations[s] == 'C':
            color = colors[0]
        elif interpretations[s] == 'P':
            color = colors[1]
        plt.plot(log2_cheating_prob[r,:,s], color=color, alpha=0.3)
    plt.plot(np.mean(log2_cheating_prob[r,:,C_ind],axis=0),color=colors[0], label='Cheating',lw=5)
    plt.plot(np.mean(log2_cheating_prob[r,:,P_ind],axis=0),color=colors[1], label='Paranoid', lw=5)
    plt.title('RUN %i' % r)
    if r > 1:
        plt.xlabel('Station number')
    plt.ylabel('p(cheating)')
    plt.legend(fontsize=10)
plt.show()


colors = ['r', 'g']
plt.figure()
for r in np.arange(nRuns):
    plt.subplot(2,2,r+1)
    for s in np.arange(nSubs):
        if all_context_scores[s] > 0:
            color = colors[0]
        elif all_context_scores[s] < 0:
            color = colors[1]
        plt.plot(log2_cheating_prob[r,:,s], color=color, alpha=0.3)
    plt.plot(np.mean(log2_cheating_prob[r,:,C_ind_interp],axis=0),color=colors[0], label='Cheating',lw=5)
    plt.plot(np.mean(log2_cheating_prob[r,:,P_ind_interp],axis=0),color=colors[1], label='Paranoid', lw=5)
    plt.title('RUN %i' % r)
    if r > 1:
        plt.xlabel('Station number')
    plt.ylabel('p(cheating)')
    plt.legend(fontsize=10)
plt.show()
################################## NEW 8/28: PLOTTING BY GROUP #############################

log2_correct_prob = log2_cheating_prob.copy()
for s in np.arange(nSubs):
    if interpretations[s] == 'P':
        log2_correct_prob[:,:,s] = 1 - log2_cheating_prob[:,:,s]
delta_wrong = np.zeros((nRuns,nSubs))
delta_right = np.zeros((nRuns,nSubs))
bl = np.zeros((nRuns,nSubs))
next_t = np.zeros((nRuns,nSubs))
for s in np.arange(nSubs):
    for r in np.arange(nRuns):
        delta_wrong[r,s],bl[r,s],next_t[r,s] = getRunAvgInc(allMaxScore[r,:,s],allMaxScore[r,:,s],nStations)
        delta_right[r,s] = getRunAvgInc_correct(allMaxScore[r,:,s],allMaxScore[r,:,s],nStations)


plt.figure()
plt.subplot(1,2,1)
for s in np.arange(nSubs-3):
    plt.plot(delta_wrong[:,s], color='k',lw = 3,alpha=0.5)
plt.plot(delta_wrong[:,17], color='r', lw = 5)
plt.plot(delta_wrong[:,18],color='c', lw = 5)
plt.xlabel('Run')
plt.ylabel('Average delta next station')
plt.title('When wrong, average change')
plt.subplot(1,2,2)
for s in np.arange(nSubs-3):
    plt.plot(delta_right[:,s], color='k',lw = 3,alpha=0.5)
plt.plot(delta_right[:,17], color='r', lw = 5)
plt.plot(delta_right[:,18],color='c', lw = 5)
plt.xlabel('Run')
plt.ylabel('Average delta next station')
plt.title('When right, average change')
plt.show()

plt.figure()
plt.subplot(1,2,1)
for s in np.arange(nSubs-3):
    plt.plot(bl[:,s], color='k',lw = 3,alpha=0.5)
plt.plot(bl[:,17], color='r', lw = 5)
plt.plot(bl[:,18],color='c', lw = 5)
plt.xlabel('Run')
plt.ylabel('Average delta next station')
plt.title('When wrong, average change')
plt.subplot(1,2,2)
for s in np.arange(nSubs-3):
    plt.plot(next_t[:,s], color='k',lw = 3,alpha=0.5)
plt.plot(next_t[:,17], color='r', lw = 5)
plt.plot(next_t[:,18],color='c', lw = 5)
plt.xlabel('Run')
plt.ylabel('Average delta next station')
plt.title('When right, average change')
plt.show()



subject_avg = np.mean(log2_cheating_prob,axis=0)
all_subject_avg = np.mean(subject_avg,axis=1)
matrix_all_average = np.repeat(all_subject_avg[:,np.newaxis],nSubs,axis=1)
diff_from_avg = subject_avg - matrix_all_average

################################## NEW 9/3: COMPARING FB MEAN #############################
from matplotlib import cm

s=0
this_cheating = orig_cheating_prob[:,:,s]
cm_subsection = np.linspace(0,1,4)
colors = [cm.cool(x) for x in cm_subsection]
plt.figure()
for r in np.arange(nRuns):
    plt.plot(np.arange(nStations),this_cheating[r,:],color=colors[r])
plt.show()

low_cheating = np.zeros((nStations,nSubs))
high_cheating = np.zeros((nStations,nSubs))
low_score = np.zeros((nStations,nSubs))
high_score = np.zeros((nStations,nSubs))
for s in np.arange(nSubs):
    this_cheating = orig_cheating_prob[:,:,s]
    low_cheating[:,s] = np.sum(this_cheating < 0.3,axis=0) 
    high_cheating[:,s] = np.sum(this_cheating > 0.5,axis=0) 
    if interpretations[s] == 'C':
        low_score[:,s] = np.sum(this_cheating < 0.3,axis=0)
        high_score[:,s] = np.sum(this_cheating > 0.5,axis=0) 
    else:
        this_paranoid = 1 - this_cheating
        low_score[:,s] = np.sum(this_paranoid < 0.3,axis=0)
        high_score[:,s] = np.sum(this_paranoid > 0.5,axis=0)

v = list(all_context_scores.values())
v2 = np.array(v)

plt.figure()
for st in np.arange(nStations):
    plt.subplot(4,4,st+1)
    for s in np.arange(nSubs):
        if interpretations[s] == 'C':
            color = 'r'
        else:
            color = 'g'
        plt.plot(low_score[st,s],v2[s], '.', color=color,alpha=0.5)
plt.show()

plt.figure()
for st in np.arange(nStations):
    plt.subplot(4,4,st+1)
    for s in np.arange(nSubs):
        if interpretations[s] == 'C':
            color = 'r'
        else:
            color = 'g'
        plt.plot(high_score[st,s],v2[s], '.', color=color,alpha=0.5)
plt.show()

std_by_station = np.std(orig_cheating_prob,axis=0)
std_by_station_log = np.std(log2_cheating_prob,axis=0)
# now get std ratio of individual / station standard deviation
# for standard deviation of entire station
total_std_station = np.zeros((nStations,))
total_std_station_log = np.zeros((nStations,))
for st in np.arange(nStations):
    all_vals = orig_cheating_prob[:,st,:].flatten()
    total_std_station[st] = np.std(all_vals)
    all_vals_log = log2_cheating_prob[:,st,:].flatten()
    total_std_station_log[st] = np.std(all_vals_log)

plt.figure()
plt.errorbar(np.arange(nStations),np.mean(std_by_station,axis=1)/total_std_station,yerr=scipy.stats.sem(std_by_station,axis=1),label='original svm')
plt.errorbar(np.arange(nStations),np.mean(std_by_station_log,axis=1)/total_std_station_log,yerr=scipy.stats.sem(std_by_station_log,axis=1),label='logistic')
plt.xlabel('Station')
plt.ylabel('std(within subject)/total std')
plt.legend()
plt.show()

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

colors = [(0,1,0), (1,0,0)] # colors go from green --> red for paranoid --> cheating
n_bins = 100
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(cmap_name,colors,N=n_bins)
old_subject_avg = np.mean(log2_cheating_prob[:,:,0:17],axis=0)
norm=plt.Normalize(-1,1)

plt.figure()
for s in np.arange(nSubs):
    plt.scatter(np.arange(nStations),high_score[:,s]/nRuns,c=np.repeat(v2[s],nStations),cmap=cm,norm=norm,alpha=0.7,s=30)
    if interpretations[s] == 'C':
        color='r'
    else:
        color='g'
    abs_score = np.abs(v2[s])
    plt.plot(np.arange(nStations),high_score[:,s]/nRuns,'--', lw=abs_score*5,color=color,alpha=0.4)
plt.xlabel('Station #')
plt.ylabel('% Rewarded at Station')
plt.show()

plt.figure()
plt.subplot(1,2,1)
for s in np.arange(len(C_ind)):
    plt.scatter(np.arange(nStations),high_score[:,C_ind[s]]/nRuns,c=np.repeat(v2[C_ind[s]],nStations),cmap=cm,norm=norm,alpha=0.7,s=30)
    if v2[C_ind[s]] > 0:
        color='r'
    else:
        color='g'
    abs_score = np.abs(v2[C_ind[s]])
    plt.plot(np.arange(nStations),high_score[:,C_ind[s]]/nRuns,'--', lw=abs_score*5,color=color,alpha=0.4)
plt.xlabel('Station #')
plt.ylabel('% Rewarded at Station')
plt.title('CHEATING GROUP')
plt.subplot(1,2,2)
for s in np.arange(len(P_ind)):
    plt.scatter(np.arange(nStations),high_score[:,P_ind[s]]/nRuns,c=np.repeat(v2[P_ind[s]],nStations),cmap=cm,norm=norm,alpha=0.7,s=30)
    if v2[P_ind[s]] > 0:
        color='r'
    else:
        color='g'
    abs_score = np.abs(v2[P_ind[s]])
    plt.plot(np.arange(nStations),high_score[:,P_ind[s]]/nRuns,'--', lw=abs_score*5,color=color,alpha=0.4)
plt.xlabel('Station #')
plt.ylabel('% Rewarded at Station')
plt.title('PARANOID GROUP')

plt.show()

orig_subject_avg = np.mean(orig_cheating_prob[:,:,0:17],axis=0)
orig_all_subject_avg = np.mean(orig_subject_avg,axis=1)
orig_matrix_all_average = np.repeat(orig_all_subject_avg[:,np.newaxis],17,axis=1)
orig_diff_from_avg = orig_subject_avg - orig_matrix_all_average

old_all_subject_avg = np.mean(old_subject_avg,axis=1)
old_matrix_all_average = np.repeat(old_all_subject_avg[:,np.newaxis],17,axis=1)
old_diff_from_avg = old_subject_avg - old_matrix_all_average



interpretations = {}
for s in np.arange(17):
    interpretations[s] = getSubjectInterpretation(allSubjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
# see for which stations is that subject average agreeing with log classifier or not for high likelihood stations
# PLOT COLOR OF EVENTUAL INTERPRETATION
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

colors = [(0,1,0), (1,0,0)] # colors go from green --> red for paranoid --> cheating
n_bins = 100
cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(cmap_name,colors,N=n_bins)
old_subject_avg = np.mean(log2_cheating_prob[:,:,0:17],axis=0)
norm=plt.Normalize(-1,1)
v = list(all_context_scores.values())
v2 = np.array(v)
this_subj_c = v2[0:17]


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8.5, 5))
for st in np.arange(nStations):
    plt.subplot(3,3,st+1)

    im = plt.scatter(old_subject_avg[st,P_ind],orig_subject_avg[st,P_ind],c=this_subj_c[P_ind],cmap=cm,norm=norm,s=50)
    if st > 5:
        plt.xlabel('log avg',fontsize=15)
    if st in np.array([0,3,6]):
        plt.ylabel('original svm avg',fontsize=15)
    if st != 7:
        plt.xlim([0,1])
    plt.ylim([0.15,.7])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.1, hspace=0.1)
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_ticks([-1,1])
cbar.set_ticklabels(['paranoid', 'cheating'])
plt.show()

################### NOW PLOT MEAN SUBTRACTED ##########
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8.5, 5))
for st in np.arange(nStations):
    plt.subplot(3,3,st+1)

    im = plt.scatter(old_diff_from_avg[st,P_ind],orig_diff_from_avg[st,P_ind],c=this_subj_c[P_ind],cmap=cm,norm=norm,s=50)
    if st > 5:
        plt.xlabel('log avg',fontsize=15)
    if st in np.array([0,3,6]):
        plt.ylabel('original svm avg',fontsize=15)
    #if st != 7:
        #plt.xlim([0,1])
    plt.ylim([-0.2,0.2])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.1, hspace=0.1)
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_ticks([-1,1])
cbar.set_ticklabels(['paranoid', 'cheating'])
plt.show()

################### NOW PLOT MEAN SUBTRACTED ##########


# to do: plot with subject mean subtracted too 

# check mean and each station
plt.figure()
plt.plot(np.mean(diff_from_avg,axis=0),all_context_scores, '.')
plt.show()

plt.figure()
for r in np.arange(nRuns):
    plt.subplot(2,2,r+1)
    plt.plot(np.mean((log2_cheating_prob[r]-matrix_all_average),axis=0),all_context_scores, '.')
plt.show()

x = np.mean((log2_cheating_prob[r]-matrix_all_average),axis=0)
scipy.stats.pearsonr(x,all_context_scores)

colors= ['r', 'g']

plt.figure()
for st in np.arange(nStations):
    plt.subplot(3,3,st+1)
    for s in np.arange(nSubs):
        interp = interpretations[s]
        if interp == 'C':
            index = 0 
        elif interp == 'P':
            index = 1
        plt.plot(subject_avg[st,s],all_context_scores[s], '.', color=colors[index],ms=20,alpha=0.3)
    r,p = scipy.stats.pearsonr(subject_avg[st,:],all_context_scores)
    plt.title('STATION %i: Corr = %2.2f; p= %2.2f' % (st,r,p),fontsize=10)
    plt.xticks(fontsize=5)
    if st + 1 >= 7:
        plt.xlabel('Avg diff from mean',fontsize=10)
    plt.ylabel('Context score',fontsize=10)
    #plt.xlim([-.1,.1]) 
plt.show()


run=3
plt.figure()
for st in np.arange(nStations):
    plt.subplot(3,3,st+1)
    for s in np.arange(nSubs):
        interp = interpretations[s]
        if interp == 'C':
            index = 0 
        elif interp == 'P':
            index = 1
        plt.plot(log2_cheating_prob[run,st,s],all_context_scores[s], '.', color=colors[index],ms=20,alpha=0.3)
    r,p = scipy.stats.pearsonr(subject_avg[st,:],all_context_scores)
    plt.title('STATION %i: Corr = %2.2f; p= %2.2f' % (st,r,p),fontsize=10)
    plt.xticks(fontsize=5)
    if st + 1 >= 7:
        plt.xlabel('Avg diff from mean',fontsize=10)
    plt.ylabel('Context score',fontsize=10)
    #plt.xlim([-.1,.1]) 
plt.show()




plt.figure()
for s in np.arange(nSubs):
    plt.plot(subject_avg[:,s])
plt.plot(np.mean(subject_avg,axis=1), 'k', lw=7)
plt.xlabel('Station')
plt.ylabel('P(cheating)')
plt.ylim([0,1])
plt.show()

orig_subject_avg = np.mean(orig_cheating_prob,axis=0)
plt.figure()
for s in np.arange(nSubs):
    plt.plot(orig_subject_avg[:,s])
plt.plot(np.mean(orig_subject_avg,axis=1), 'k', lw=7)
plt.xlabel('Station')
plt.ylabel('P(cheating)')
plt.ylim([0,1])
plt.show()

orig_subject_correct = orig_subject_avg.copy()
to_flip = np.array([0,1,3,4,5,6])
orig_subject_correct[to_flip,:] = 1-orig_subject_avg[to_flip,:]
plt.figure()
for s in np.arange(nSubs):
    plt.plot(orig_subject_correct[:,s])
plt.plot(np.mean(orig_subject_correct,axis=1), 'k', lw=7)
plt.xlabel('Station')
plt.ylabel('P(cheating)')
plt.ylim([0,1])
plt.show()



all_avg = np.mean(subject_avg,axis=1)
max_high = np.zeros((nStations,))
max_low = np.zeros((nStations,))
for st in np.arange(nStations):
    max_high[st] = np.max(subject_avg[st,:])
    max_low[st] = np.min(subject_avg[st,:])

max_paranoid = 1 - max_low
#np.savez('all_station_log_info.npz',avg=all_avg,max=max_high,min=max_low)
a = np.load('all_station_log_info.npz')
c_max = a['max']
p_max = 1 - a['min']

subject_data = log2_cheating_prob[0,:,0]
for st in np.arange(nStations):
    interp = getSubjectInterpretation(s)
    if interp == 'C':
        score = subject_data[st]/c_max[st]
    elif interp == 'P':
        score = subject_data[st]/p_max[st]
    if score > 1:
        score = 1
    print(score)





colors = ['m', 'r', 'b', 'g']
for s in np.arange(nSubs):
    plt.figure()
    for r in np.arange(nRuns):
        plt.plot(orig_cheating_prob[r,:,s], color=colors[r])
        plt.plot(log2_cheating_prob[r,:,s], '--', color=colors[r])
    plt.show()

c = np.zeros((nRuns,nStations))
r = np.zeros((nRuns,nStations))
plt.figure()
for run in np.arange(nRuns):
    data = getPatternsData(21,run+1)
    c[run,:],r[run,:] = getClassificationAndScore(data)
    label = 'run %i' % run
    plt.plot(c[run,:], color=colors[run], label =label) 
    plt.plot(r[run,:],'--', color=colors[run])
    plt.plot(allRatioScore[run,:,19],'-.',color=colors[run])
plt.ylim([0,1])
plt.xlabel('Station')
plt.ylabel('p(cheating)')
plt.legend(fontsize=5,loc=1)
plt.show()

run_avg = np.mean(allMaxScore,axis=1)
plt.figure()
for s in np.arange(nSubs):
    label = 's % i' % allSubjects[s]
    plt.plot(run_avg[:,s], label=label)
plt.legend(fontsize=5)
plt.show()


change_of_interp = run_avg[3,:] - run_avg[0,:]
all_correct_scores = all_context_scores.copy()
all_correct_scores[P_ind] = all_context_scores[P_ind]*-1
plt.figure()
for s in np.arange(nSubs):
    interp = interpretations[s]
    if interp == 'C':
        index = 0 
        score = all_context_scores[s]
    elif interp == 'P':
        index = 1
        score = all_context_scores[s] * -1
    plt.plot(change_of_interp[s],score, '.', color=colors[index],ms=20,alpha=0.3)
plt.show()
all_ind = np.arange(nSubs)
all_ind = np.delete(all_ind,7)
stats.pearsonr(change_of_interp[all_ind],all_correct_scores[all_ind])
# plot classifiers
clf = 1 
stationInd=7
clf = loadClassifier(clf,stationInd)

plt.figure()
plt.hist(clf.coef_)
plt.show()