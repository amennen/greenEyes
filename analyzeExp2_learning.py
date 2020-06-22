# purpose: calculate learning behavior for each group


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
params = {'legend.fontsize': 'large',
          'figure.figsize': (5, 3),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
font = {'weight': 'normal',
        'size': 22}
plt.rc('font', **font)
defaultConfig = os.path.join(os.getcwd(), 'conf/greenEyes_cluster.toml')
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
    cheating_prob = data['cheating_probability']
    cheating_prob_z = data['zTransferred']
    correct_score = data['correct_prob']
    return data, cheating_prob, cheating_prob_z, correct_score

def getBehavData(subjectNum,runNum):
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    ses_id = 'ses-{0:02d}'.format(2)
    run_id = 'run-{0:03d}'.format(runNum)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/{2}/behavior_run{3}_*.mat'.format(bids_id,ses_id,run_id,runNum)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data


def getStationInformation(config='conf/greenEyes_cluster.toml'):
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

def createStationVector(stationDict):
    nStations = len(stationDict)
    allTRs = np.arange(25,475+1)
    nTRs_story = len(allTRs)
    recorded_TRs = np.zeros((nTRs_story,))
    for st in np.arange(nStations):
        this_station_TRs = np.array(stationDict[st]) + 1 # going from python --> matlab indexing
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
    nStations = 7
    probe_response = []
    for st in np.arange(nStations):
        LEFT = 0
        RIGHT = 0
        display_TR = np.argwhere(recorded_TRs == st+1)[0][0]

        if len(np.intersect1d(np.argwhere(display_TR - allLeftTRs[:,0] > 0),np.argwhere(display_TR - allLeftTRs[:,0] <= 3)) ) > 0:
            # then there was a left keypress for this station
            LEFT = 1
            probe_response_st = left_key
        if len(np.intersect1d(np.argwhere(display_TR - allRightTRs[:,0] > 0),np.argwhere(display_TR - allRightTRs[:,0] <= 3)) ) > 0:
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

def calculateSumValueInterpretations_behavior(this_sub_response,this_sub_score,this_sub_cheating,this_sub_interpretation):
    """Here we want to calculate the subject's value based on all stations"""
    nTrials = len(this_sub_response)
    # assume equal value to start
    value_c = 0
    value_p = 0
    n_c = 0
    n_p = 0
    for t in np.arange(nTrials):
        # see if subject pressed
        response = this_sub_response[t]
        if response == 'CHEATING':
            # now update value
            value_c = value_c + this_sub_score[t]
            n_c += 1
        elif response == 'PARANOID':
            value_p = value_p + this_sub_score[t]
            n_p += 1
    return value_c, value_p, value_c/n_c, value_p/n_p


def calculateValueInterpretations_behavior(this_sub_response,this_sub_score,this_sub_cheating,this_sub_interpretation):
    """Here we want to calculate the subject's value based on all stations"""
    nTrials = len(this_sub_response)
    # assume equal value to start
    value_c = 0.5 # initialize value but update nTrials
    value_p = 0.5
    lr = 0.1
    for t in np.arange(nTrials):
        # see if subject pressed
        response = this_sub_response[t]
        if response == 'CHEATING':
            # now update value
            value_c = value_c + lr*(this_sub_score[t]-value_c)
        elif response == 'PARANOID':
            value_p = value_p + lr*(this_sub_score[t]-value_p)
    return value_c, value_p

def calculateValueInterpretations_neural(this_sub_response,this_sub_score,this_sub_cheating,this_sub_interpretation):
    """Here we want to calculate the subject's value based on all stations"""
    # how about only go for neural
    nTrials = len(this_sub_response)
    # assume equal value to start
    value_c = 0.5 # initialize value but update nTrials
    value_p = 0.5

    for t in np.arange(nTrials):
        # see if subject pressed
        response = this_sub_response[t]
        # check if brain matched what they respond
        if this_sub_cheating[t] >= 0.5:
            # prediction error = actual value - expectation
            # reward - neural expectation
            value_c = value_c + (this_sub_score[t]-1)*this_sub_cheating[t]
        elif this_sub_cheating[t] < 0.5:
            value_p = value_p + (this_sub_score[t] - 1)*(1-this_sub_cheating[t])
    return value_c, value_p

    
def getStayWinLoseShift(nf_score,probe_response):
    nStations=7
    n_win = 0
    n_win_stay = 0
    n_lose = 0 
    n_lose_shift = 0
    for st in np.arange(nStations-1):
        this_reward = nf_score[0,st]
        this_response = probe_response[st]
        next_response = probe_response[st+1]
        if this_reward >= 0.5:
            n_win += 1
            if this_response == next_response:
                n_win_stay +=1
        else:
            n_lose += 1
            if this_response == 'PARANOID' and next_response == "CHEATING": 
                n_lose_shift +=1
            if this_response == 'CHEATING' and next_response == "PARANOID":
                n_lose_shift +=1
    if n_win > 0:
        p_stay_win = n_win_stay/n_win
    else:
        p_stay_win = np.nan
    if n_lose > 0:
        p_shift_lose = n_lose_shift/n_lose
    else:
        p_shift_lose = np.nan
    return p_stay_win, p_shift_lose

def getClfChange(nf_score):
    nStations=7
    changes_win = np.zeros((nStations-1,))*np.nan
    changes_lose = np.zeros((nStations-1,))*np.nan

    for st in np.arange(nStations-1):
        this_reward = nf_score[0,st]
        next_reward = nf_score[0,st+1]
        if this_reward >= 0.5:
            changes_win[st] = next_reward - this_reward
        else:
            changes_lose[st] = next_reward - this_reward
    changes_win_avg = np.nanmean(changes_win)
    changes_lose_avg = np.nanmean(changes_lose)
    return np.array([changes_win_avg,changes_lose_avg])

def convertToMatScore(subjectNum,this_sub_response,this_sub_score):
    """ set 1 = cheating, 2 = paranoid, 0 = no response """
    nTrials = len(this_sub_response)
    subject_data = {}
    choices = np.zeros((nTrials,))
    for t in np.arange(nTrials):
        if this_sub_response[t] == 'CHEATING':
            choices[t] = 1
        elif this_sub_response[t] == 'PARANOID':
            choices[t] = 2
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    subject_data['subj'] = bids_id
    subject_data['choices'] = choices
    subject_data['scores'] = this_sub_score

    fn = 'learning/{0}_behavior.mat'.format(bids_id)
    scipy.io.savemat(fn,subject_data)
    return

def makeChoicePlot(subjectNum,this_sub_response,this_sub_score,this_sub_interpretation):
    """ set 1 = cheating, 2 = paranoid, 0 = no response """
    nTrials = len(this_sub_response)
    subject_data = {}
    choices = np.zeros((nTrials,))
    for t in np.arange(nTrials):
        if this_sub_response[t] == 'CHEATING':
            choices[t] = 1
        elif this_sub_response[t] == 'PARANOID':
            choices[t] = 2
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    no_resp = np.argwhere(choices==0)[:,0]
    score_resp = this_sub_score.copy()
    score_resp[no_resp] = np.nan
    fig,ax = plt.subplots(figsize=(15,10))
    sns.despine()
    plt.plot(np.arange(nTrials),this_sub_score,color='k')
    for t in np.arange(1,nTrials):
        if choices[t] > 0 and choices[t-1] > 0:
            if choices[t] != choices[t-1]: # switched
                color = 'r'
            elif choices[t] == choices[t-1]:
                color = 'b'
            plt.plot(t,this_sub_score[t],'.', color=color, ms=10)
    plt.plot([6,6],[-1,2], '--', color='k')
    plt.plot([6+7,6+7],[-1,2], '--', color='k')
    plt.plot([6+7+7,6+7+7],[-1,2], '--', color='k')
    plt.ylim([-.05,1.05])
    plt.xlim([-1,nTrials])
    plt.xlabel('trial #')
    plt.ylabel('neurofeedback score')
    c_lines = makeCustomLegend('b', 'r', 4)
    ax.legend(c_lines,['stayed', 'switched'],loc=0)
    title = '{0}; Group {1}'.format(bids_id, this_sub_interpretation)
    plt.title(title)
    fn = 'learning/{0}_choices.pdf'.format(bids_id)
    plt.savefig(fn)
    #plt.show()
    #scipy.io.savemat(fn,subject_data)
    return

def makeChoicePlot2(subjectNum,this_sub_response,this_sub_score,this_sub_interpretation):
    """ set 1 = cheating, 2 = paranoid, 0 = no response """
    nTrials = len(this_sub_response)
    subject_data = {}
    choices = np.zeros((nTrials,))
    for t in np.arange(nTrials):
        if this_sub_response[t] == 'CHEATING':
            choices[t] = 1
        elif this_sub_response[t] == 'PARANOID':
            choices[t] = 2
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    no_resp = np.argwhere(choices==0)[:,0]
    score_resp = this_sub_score.copy()
    score_resp[no_resp] = np.nan
    fig,ax = plt.subplots(figsize=(15,10))
    sns.despine()
    plt.plot(np.arange(nTrials),this_sub_score,color='k')
    for t in np.arange(1,nTrials):
        if choices[t] > 0 and choices[t-1] > 0:
            if choices[t] != choices[t-1]: # switched
                color = 'r'
            elif choices[t] == choices[t-1]:
                color = 'b'
            plt.plot(t,this_sub_score[t],'.', color=color, ms=10)
    plt.plot([6,6],[-1,2], '--', color='k')
    plt.plot([6+7,6+7],[-1,2], '--', color='k')
    plt.plot([6+7+7,6+7+7],[-1,2], '--', color='k')
    plt.ylim([-.05,1.05])
    plt.xlim([-1,nTrials])
    plt.xlabel('trial #')
    plt.ylabel('neurofeedback score')
    c_lines = makeCustomLegend('b', 'r', 4)
    ax.legend(c_lines,['stayed', 'switched'],loc=0)
    title = '{0}; Group {1}'.format(bids_id, this_sub_interpretation)
    plt.title(title)
    fn = 'learning/{0}_choices.pdf'.format(bids_id)
    plt.savefig(fn)
    #plt.show()
    #scipy.io.savemat(fn,subject_data)
    return

def consecutive(data, stepsize=0):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def calculateCertainty(subjectNum,this_sub_response):
    # do for first run and last run separately
    nTrials = len(this_sub_response)
    choices = np.zeros((nTrials,))
    for t in np.arange(nTrials):
        if this_sub_response[t] == 'CHEATING':
            choices[t] = 1
        elif this_sub_response[t] == 'PARANOID':
            choices[t] = 2
    firstRun = choices[0:7]
    list_firstRun = consecutive(firstRun)
    if list_firstRun[0].size > 0:
        longest_first = 1
        for k in np.arange(len(list_firstRun)):
            this_length = len(list_firstRun[k])
            this_val = list_firstRun[k][0]
            if this_length > longest_first and this_val > 0:
                longest_first = this_length
    else:
        longest_first = np.nan
    lastRun = choices[21:]
    list_lastRun = consecutive(lastRun)
    if list_lastRun[0].size > 0:
        longest_last = 1
        for k in np.arange(len(list_lastRun)):
            this_length = len(list_lastRun[k])
            this_val = list_lastRun[k][0]

            if this_length > longest_last and this_val > 0:
                longest_last = this_length
    else:
        longest_last = np.nan
    return longest_first, longest_last
    # for each station # conseuctive choices


def calculateCertainty_correct(subjectNum,this_sub_response,this_sub_interpretation):
    # do for first run and last run separately - same as above but now look at CORRECT CONSECUTIVE CHOICES
    nTrials = len(this_sub_response)
    choices = np.zeros((nTrials,))
    for t in np.arange(nTrials):
        if this_sub_response[t] == 'CHEATING':
            choices[t] = 1
        elif this_sub_response[t] == 'PARANOID':
            choices[t] = 2
    firstRun = choices[0:7]
    list_firstRun = consecutive(firstRun)
    if list_firstRun[0].size > 0:
        longest_first = 1
        for k in np.arange(len(list_firstRun)):
            this_length = len(list_firstRun[k])
            this_val = list_firstRun[k][0]
            # only count it if their choice was their group assignment
            if this_val == 1 and this_sub_interpretation == 'C':
                if this_length > longest_first:
                    longest_first = this_length
            elif this_val == 2 and this_sub_interpretation == 'P':
                if this_length > longest_first:
                    longest_first = this_length
    else:
        longest_first = np.nan
    lastRun = choices[21:]
    list_lastRun = consecutive(lastRun)
    if list_lastRun[0].size > 0:
        longest_last = 1
        for k in np.arange(len(list_lastRun)):
            this_length = len(list_lastRun[k])
            this_val = list_lastRun[k][0]
            if this_val == 1 and this_sub_interpretation == 'C':
                if this_length > longest_last:
                    longest_last = this_length
            elif this_val == 2 and this_sub_interpretation == 'P':
                if this_length > longest_last:
                    longest_last = this_length
    else:
        longest_last = np.nan
    return longest_first, longest_last

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


p_stay_win = np.zeros((nSubs,nRuns)) * np.nan
p_shift_lose = np.zeros((nSubs,nRuns)) * np.nan
subj_means = np.zeros((nSubs,2)) * np.nan
score_change = np.zeros((nSubs,2,nRuns))*np.nan
all_choices = np.zeros((nSubs,nStations,nRuns))*np.nan
behavioral_cheating = np.zeros((nSubs,))
behavioral_paranoid = np.zeros((nSubs,))
consecutive_choices = np.zeros((nSubs,2)) * np.nan
consecutive_choices_correct = np.zeros((nSubs,2)) * np.nan
normalized_c_val = np.zeros((nSubs,))* np.nan
normalized_p_val = np.zeros((nSubs,))* np.nan
all_cheating_prob = np.zeros((nSubs,nStations,nRuns)) * np.nan
sum_c_val = np.zeros((nSubs,))* np.nan
sum_p_val = np.zeros((nSubs,))* np.nan
total_std = np.zeros((nSubs,)) * np.nan
subj_std = np.zeros((nSubs,2)) * np.nan
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
        this_sub_cheating = np.append(this_sub_cheating,c_prob[0,:])
        #this_sub_cheating = np.append(this_sub_cheating,c_prob[0,:])
        this_sub_score = np.append(this_sub_score,nf_score[0,:])
        # now get prob answers
        b = getBehavData(subjectNum,runNum+1)
        probe_response = getProbeKeyPresses(b,recorded_TRs)
        C_press = [i for i in np.arange(len(probe_response))  if probe_response[i]  == 'CHEATING'] 
        P_press = [i for i in np.arange(len(probe_response))  if probe_response[i]  == 'PARANOID'] 
        all_choices[s,C_press,runNum] = 1
        all_choices[s,P_press,runNum] = 0
        this_sub_response = this_sub_response + probe_response
        # calculate by run
        p_stay_win[s,runNum], p_shift_lose[s,runNum] = getStayWinLoseShift(nf_score,probe_response)
        score_change[s,:,runNum] = getClfChange(nf_score)
        all_cheating_prob[s,:,runNum] = c_prob[0,:]

    ind_cheating = [i for i, x in enumerate(this_sub_response) if x == "CHEATING"]  
    ind_paranoid = [i for i, x in enumerate(this_sub_response) if x == "PARANOID"]  
    subj_means[s,0] = np.nanmean(this_sub_cheating[ind_cheating])
    subj_means[s,1] = np.nanmean(this_sub_cheating[ind_paranoid])
    total_std[s] = np.nanstd(this_sub_cheating)
    subj_std[s,0] = np.nanstd(this_sub_cheating[ind_cheating])
    subj_std[s,1] = np.nanstd(this_sub_cheating[ind_paranoid])
    #convertToMatScore(subjectNum,this_sub_response,this_sub_score)
    #makeChoicePlot(subjectNum,this_sub_response,this_sub_score,this_sub_interpretation)
    sum_c_val[s], sum_p_val[s], normalized_c_val[s], normalized_p_val[s] = calculateSumValueInterpretations_behavior(this_sub_response,this_sub_score,this_sub_cheating,this_sub_interpretation)
    consecutive_choices[s,0], consecutive_choices[s,1] = calculateCertainty(subjectNum,this_sub_response)
    consecutive_choices_correct[s,0], consecutive_choices_correct[s,1] = calculateCertainty_correct(subjectNum,this_sub_response,this_sub_interpretation)
    behavioral_cheating[s],behavioral_paranoid[s] = calculateValueInterpretations_behavior(this_sub_response,this_sub_score,this_sub_cheating,this_sub_interpretation)


classifier_separation = subj_means[:,0] - subj_means[:,1]
c_sep_P = classifier_separation[P_ind]
c_sep_C = classifier_separation[C_ind]
keep_C = np.argwhere(c_sep_C>0)
keep_P = np.argwhere(c_sep_P>0)

ord_P = np.argsort(c_sep_P)[::-1]  # this is best to worst
P_ind2 = np.array(P_ind)
ord_C = np.argsort(c_sep_C)[::-1]  # this is best to worst
C_ind2 = np.array(C_ind)
top_P_subj = P_ind2[ord_P[0:5]]
bottom_P_subj = P_ind2[ord_P[5:]]
top_C_subj = C_ind2[ord_C[0:5]]
bottom_C_subj = C_ind2[ord_C[5:]]
keep_C = C_ind2[keep_C]
keep_P = P_ind2[keep_P]
keep_subj = np.concatenate((keep_P,keep_C))[:,0]
top_subj = np.concatenate((top_P_subj,top_C_subj))
bottom_subj = np.concatenate((bottom_P_subj,bottom_C_subj))

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

# look at trand with story score as well
z=all_context_scores*12
# now everyone answered "paranoid" for that so I must have subtracted one too many then
new_context_scores = (z.copy() + 1)/11
scores = np.concatenate((all_story_scores[:,np.newaxis],new_context_scores[:,np.newaxis]),axis=1)

#all_correct_context = all_context_scores.copy()
#all_correct_context[P_ind] = -1*all_context_scores[P_ind]
all_correct_context = new_context_scores.copy()
all_correct_context[P_ind] = -1*new_context_scores[P_ind]
arthur_minus_lee = all_rating_scores[:,0] - all_rating_scores[:,1]
arthur_minus_lee_cor = arthur_minus_lee.copy()
arthur_minus_lee_cor[P_ind] = -1*arthur_minus_lee[P_ind]
value_diff = behavioral_cheating - behavioral_paranoid
# question is value difference predictive of choices?
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for s in np.arange(nSubs):
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.plot(value_diff[s],new_context_scores[s],'.',ms=20,color=color,alpha=0.5)
x = value_diff
y = new_context_scores
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,1,text_f)
#plt.xlim([-1.1,1])
plt.ylim([-1.1,1.1])
plt.xlabel('value (cheating - paranoid)')
plt.ylabel('interpretation (cheating - paranoid)')
plt.savefig('savedPlots/value_context.pdf')
plt.show()

mat_data = scipy.io.loadmat('learning/results_firsthalf.mat',appendmat=True)
learning_rate = mat_data['results']['lr'][0,0][:,0]
beta = mat_data['results']['beta'][0,0][:,0]
lik = mat_data['results']['lik'][0,0][:,0]
# first make new data frame
best_worst = [ ''] * nSubs
for i in np.arange(nSubs):
    if i in top_subj:
        best_worst[i] = 'best'
    elif i in bottom_subj:
        best_worst[i] = 'worst'
subject_vector = np.arange(nSubs)
group_str = [''] * nSubs
for s in np.arange(nSubs):
  if s in C_ind:
    group_str[s] = 'C'
  elif s in P_ind:
    group_str[s] = 'P'

within_group_var = np.nanmean(subj_std,axis=1)
var_across_runs = np.nanmean(np.nanstd(all_cheating_prob,axis=2),axis=1)

data = {}
data['subjects'] = subject_vector
data['all_std'] = total_std
data['within_var'] = within_group_var
data['bestworst'] = best_worst
data['var_across_runs'] = var_across_runs
data['lr'] = learning_rate
data['beta'] = beta
data['lik'] = lik
data['group'] = group_str
data['init_certainty'] = consecutive_choices[:,0]
data['final_certainty'] = consecutive_choices[:,1]
data['certainty_diff'] = consecutive_choices[:,1] - consecutive_choices[:,0]
data['init_certainty_cor'] = consecutive_choices_correct[:,0]
data['final_certainty_corr'] = consecutive_choices_correct[:,1]
data['certainty_diff_corr'] = consecutive_choices_correct[:,1] - consecutive_choices_correct[:,0]
data['corr_context'] = all_correct_context
data['arthur_minus_lee_cor'] = arthur_minus_lee_cor
df = pd.DataFrame.from_dict(data)
P2 = makeColorPalette(['#99d8c9','#fc9272'])

#### MAKING LEARNING PLOTS####################################################################################
sns.despine()
sns.barplot(data=df,x='bestworst',y='lr',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='lr',hue='group',split=False,palette=P2,size=8,alpha=0.5)

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='beta',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='beta',hue='group',split=False,palette=P2,size=8,alpha=0.5)

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='lik',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='lik',hue='group',split=False,palette=P2,size=8,alpha=0.5)

scipy.stats.ttest_ind(learning_rate[top_subj],learning_rate[bottom_subj])
scipy.stats.ttest_ind(beta[top_subj],beta[bottom_subj])
scipy.stats.ttest_ind(lik[top_subj],lik[bottom_subj])
#########################################################################################################

### CERTAINTY ANALYSIS - CONSECUTIVE TIME POINTS
P2 = makeColorPalette(['#99d8c9','#fc9272'])
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='init_certainty',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='init_certainty',hue='group',split=False,palette=P2,size=8,alpha=0.5)
plt.xlabel('clf performance')
plt.ylabel('longest consecutive choice')
plt.savefig('savedPlots/initial_certainty.pdf')
plt.show()
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='init_certainty_cor',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='init_certainty_cor',hue='group',split=False,palette=P2,size=8,alpha=0.5)
plt.xlabel('clf performance')
plt.ylabel('longest consecutive CORRECT choice')
x,y=nonNan(consecutive_choices_correct[top_subj,0],consecutive_choices_correct[bottom_subj,0])
t,p = scipy.stats.ttest_ind(x,y)
maxH=7
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='best>worst')
plt.savefig('savedPlots/initial_certainty_cor.pdf')
plt.show()

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='final_certainty',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='final_certainty',hue='group',split=False,palette=P2,size=8,alpha=0.5)
x,y=nonNan(consecutive_choices[top_subj,1],consecutive_choices[bottom_subj,1])
plt.xlabel('clf performance')
plt.ylabel('longest consecutive choice')
x,y=nonNan(consecutive_choices[top_subj,1],consecutive_choices[bottom_subj,1])
t,p = scipy.stats.ttest_ind(x,y)
maxH=7
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='best>worst')
plt.savefig('savedPlots/final_certainty.pdf')
plt.show()

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='final_certainty_corr',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='final_certainty_corr',hue='group',split=False,palette=P2,size=8,alpha=0.5)
x,y=nonNan(consecutive_choices_correct[top_subj,1],consecutive_choices_correct[bottom_subj,1])
plt.xlabel('clf performance')
plt.ylabel('longest consecutive CORRECT choice')
x,y=nonNan(consecutive_choices_correct[top_subj,1],consecutive_choices_correct[bottom_subj,1])
t,p = scipy.stats.ttest_ind(x,y)
maxH=7
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='best>worst')
plt.savefig('savedPlots/final_certainty_cor.pdf')
plt.show()

certainty_diff = consecutive_choices[:,1] - consecutive_choices[:,0]
certainty_diff_correct = consecutive_choices_correct[:,1] - consecutive_choices_correct[:,0]
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='certainty_diff',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='certainty_diff',hue='group',split=False,palette=P2,size=8,alpha=0.5)
x,y=nonNan(certainty_diff[top_subj],certainty_diff[bottom_subj])
t,p = scipy.stats.ttest_ind(x,y)
maxH=7
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='best>worst')
plt.xlabel('clf performance')
plt.ylabel('change consecutive choice')
plt.ylim([-5,8])
plt.savefig('savedPlots/certainty_diff.pdf')
plt.show()

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='certainty_diff_corr',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='certainty_diff_corr',hue='group',split=False,palette=P2,size=8,alpha=0.5)
x,y=nonNan(certainty_diff_correct[top_subj],certainty_diff_correct[bottom_subj])
t,p = scipy.stats.ttest_ind(x,y)
maxH=7
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='best>worst')
plt.xlabel('clf performance')
plt.ylabel('change consecutive CORRECT choice')
plt.ylim([-5,8])
plt.savefig('savedPlots/certainty_diff_corr.pdf')
plt.show()

scipy.stats.ttest_ind(x,y)
#########################################################################################################


### VARIANCE ANALYSIS #########################################################################################################

P2 = makeColorPalette(['#99d8c9','#fc9272'])
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='all_std',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='all_std',hue='group',split=False,palette=P2,size=8,alpha=0.5)
plt.xlabel('clf performance')
plt.ylabel('standard deviation')
plt.savefig('savedPlots/all_std.pdf')
plt.show()
scipy.stats.ttest_ind(total_std[top_subj],total_std[bottom_subj])

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='within_var',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='within_var',hue='group',split=False,palette=P2,size=8,alpha=0.5)
plt.xlabel('clf performance')
plt.ylabel('within group standard deviation')
plt.savefig('savedPlots/within_group_std.pdf')
plt.show()
scipy.stats.ttest_ind(within_group_var[top_subj],within_group_var[bottom_subj])

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='bestworst',y='var_across_runs',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='var_across_runs',hue='group',split=False,palette=P2,size=8,alpha=0.5)
plt.xlabel('clf performance')
plt.ylabel('average std across runs')
plt.savefig('savedPlots/std_across_runs.pdf')
plt.show()
scipy.stats.ttest_ind(var_across_runs[top_subj],var_across_runs[bottom_subj])







normalized_value_diff = normalized_c_val - normalized_p_val
fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for s in np.arange(nSubs):
    # if interpretations[s] == 'C':
    #     color='r'
    # elif interpretations[s] == 'P':
    #     color='g'
    if s in top_subj:
        marker = 's'
        color = 'r'
    elif s in bottom_subj:
        marker = 'X'
        color = 'b'
    plt.plot(normalized_value_diff[s],new_context_scores[s],'.',ms=20,color=color,alpha=0.5,marker=marker)
x = normalized_value_diff
y = new_context_scores
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k', label='all')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,1,text_f)

x = normalized_value_diff[top_subj]
y = new_context_scores[top_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '--',alpha=0.6,lw=1, color='r', label='top')
x = normalized_value_diff[bottom_subj]
y = new_context_scores[bottom_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '--',alpha=0.6,lw=1, color='b', label='bottom')

#plt.xlim([-1.1,1])
#plt.ylim([-1.1,1.1])
plt.xlabel('average value (cheating - paranoid)')
plt.ylabel('interpretation (cheating - paranoid)')
plt.legend()
plt.savefig('savedPlots/context_avgscore.pdf')
plt.show()

fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for s in np.arange(nSubs):
    # if interpretations[s] == 'C':
    #     color='r'
    # elif interpretations[s] == 'P':
    #     color='g'
    if s in top_subj:
        marker = 's'
        color = 'r'
    elif s in bottom_subj:
        marker = 'X'
        color = 'b'
    plt.plot(normalized_value_diff[s],arthur_minus_lee[s],'.',ms=20,color=color,alpha=0.5,marker=marker)
x = normalized_value_diff
y = arthur_minus_lee
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,1,text_f)
x = normalized_value_diff[top_subj]
y = arthur_minus_lee[top_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '--',alpha=0.6,lw=1, color='r', label='top')
x = normalized_value_diff[bottom_subj]
y = arthur_minus_lee[bottom_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '--',alpha=0.6,lw=1, color='b', label='bottom')
#plt.xlim([-1.1,1])
#plt.ylim([-1.1,1.1])
plt.ylabel('empathy (athur - lee)')
plt.xlabel('average value (cheating - paranoid)')
plt.legend()
plt.savefig('savedPlots/empathy_avgscore.pdf')
plt.show()

fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for s in top_subj:
    #s = subjects
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.plot(normalized_value_diff[s],new_context_scores[s],'.',ms=20,color=color,alpha=0.5)
x = normalized_value_diff[top_subj]
y = new_context_scores[top_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,1,text_f)
#plt.xlim([-1.1,1])
#plt.ylim([-1.1,1.1])
plt.xlabel('value (cheating - paranoid)')
plt.ylabel('interpretation (cheating - paranoid)')
plt.show()

fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for s in top_subj:
    #s = subjects
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.plot(normalized_value_diff[s],arthur_minus_lee[s],'.',ms=20,color=color,alpha=0.5,marker=marker)
x = normalized_value_diff[top_subj]
y = arthur_minus_lee[top_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,1,text_f)
#plt.xlim([-1.1,1])
#plt.ylim([-1.1,1.1])
plt.xlabel('value (cheating - paranoid)')
plt.ylabel('empathy (athur - lee)')
plt.show()

fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for s in bottom_subj:
    #s = subjects
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.plot(normalized_value_diff[s],new_context_scores[s],'.',ms=20,color=color,alpha=0.5)
x = normalized_value_diff[bottom_subj]
y = new_context_scores[bottom_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,1,text_f)
#plt.xlim([-1.1,1])
#plt.ylim([-1.1,1.1])
plt.xlabel('value (cheating - paranoid)')
plt.ylabel('interpretation (cheating - paranoid)')
plt.show()

fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for s in bottom_subj:
    #s = subjects
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.plot(normalized_value_diff[s],arthur_minus_lee[s],'.',ms=20,color=color,alpha=0.5,marker=marker)
x = normalized_value_diff[bottom_subj]
y = arthur_minus_lee[bottom_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,1,text_f)
#plt.xlim([-1.1,1])
#plt.ylim([-1.1,1.1])
plt.xlabel('value (cheating - paranoid)')
plt.ylabel('empathy (athur - lee)')
plt.show()


fig,ax = plotPosterStyle_DF(learning_rate[top_subj],subjects[top_subj])

fig,ax = plotPosterStyle_DF(learning_rate[bottom_subj],subjects[bottom_subj])


fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for i in np.arange(len(top_subj)):
    s = top_subj[i]
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.plot(value_diff[s],new_context_scores[s],'.',ms=20,color=color,alpha=0.5)
x = value_diff[top_subj]
y = new_context_scores[top_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,1,text_f)
#plt.xlim([-1.1,1])
plt.ylim([-1.1,1.1])
plt.xlabel('value (cheating - paranoid)')
plt.ylabel('interpretation (cheating - paranoid)')
plt.show()

fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for i in np.arange(len(bottom_subj)):
    s = bottom_subj[i]
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.plot(value_diff[s],new_context_scores[s],'.',ms=20,color=color,alpha=0.5)
x = value_diff[bottom_subj]
y = new_context_scores[bottom_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,1,text_f)
#plt.xlim([-1.1,1])
plt.ylim([-1.1,1.1])
plt.xlabel('value (cheating - paranoid)')
plt.ylabel('interpretation (cheating - paranoid)')
plt.show()


fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for i in np.arange(len(keep_subj)):
    s = keep_subj[i]
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.plot(value_diff[s],new_context_scores[s],'.',ms=20,color=color,alpha=0.5)
x = value_diff[keep_subj]
y = new_context_scores[keep_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-.2,1,text_f)
#plt.xlim([-1.1,1])
plt.ylim([-1.1,1.1])
plt.xlabel('value (cheating - paranoid)')
plt.ylabel('interpretation (cheating - paranoid)')
plt.savefig('savedPlots/value_context_keep.pdf')
plt.show()

fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for s in np.arange(nSubs):
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.plot(value_diff[s],arthur_minus_lee[s],'.',ms=20,color=color,alpha=0.5)
x = value_diff
y = arthur_minus_lee
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-1,5,text_f)
plt.xlim([-1.1,1])
plt.ylim([-5,5])
plt.xlabel('value (cheating - paranoid)')
plt.ylabel('empathy (arthur - lee)')
plt.show()

fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
for i in np.arange(len(keep_subj)):
    s = keep_subj[i]
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.plot(value_diff[s],arthur_minus_lee[s],'.',ms=20,color=color,alpha=0.5)
x = value_diff[keep_subj]
y = arthur_minus_lee[keep_subj]
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-1,5,text_f)
plt.xlim([-1.1,1])
plt.ylim([-5,5])
plt.xlabel('value (cheating - paranoid)')
plt.ylabel('empathy (arthur - lee)')
plt.show()



stations_use = np.arange(nStations)
fig,ax = plotPosterStyle_DF_valence(all_choices[:,stations_use,:],subjects,'choice')
maxH=1.1
choices_by_run = np.nanmean(all_choices[:,stations_use,:],axis=1)
for r in np.arange(nRuns):
    x,y=nonNan(choices_by_run[C_ind,r],choices_by_run[P_ind,r])
    t,p = scipy.stats.ttest_ind(x,y)
    addComparisonStat_SYM(p/2,r-0.2,r+0.2,maxH,.05,0,text_above='C>P')
plt.ylabel('p(choose cheating)')
plt.savefig('savedPlots/choices_stations_averaged.pdf')
plt.show()


fig,ax = plotPosterStyle_DF_valence(all_choices[top_subj],subjects[top_subj],'choice')
for r in np.arange(nRuns):
    x,y=nonNan(choices_by_run[top_C_subj,r],choices_by_run[top_P_subj,r])
    t,p = scipy.stats.ttest_ind(x,y)
    addComparisonStat_SYM(p/2,r-0.2,r+0.2,maxH,.05,0,text_above='C>P')
plt.ylabel('p(choose cheating)')
plt.savefig('savedPlots/choices_stations_averaged_top.pdf')
plt.show()

fig,ax = plotPosterStyle_DF_valence(all_choices[keep_subj],subjects[keep_subj],'choice')
maxH=1
for r in np.arange(nRuns):
    x,y=nonNan(choices_by_run[keep_C,r],choices_by_run[keep_P,r])
    t,p = scipy.stats.ttest_ind(x,y)
    addComparisonStat_SYM(p/2,r-0.2,r+0.2,maxH,.05,0,text_above='C>P')
plt.ylabel('p(choose cheating)')
plt.savefig('savedPlots/choices_stations_averaged_keep.pdf')
plt.show()

fig,ax = plotPosterStyle_DF_valence(all_choices[bottom_subj],subjects[bottom_subj],'choice')
for r in np.arange(nRuns):
    x,y=nonNan(choices_by_run[bottom_C_subj,r],choices_by_run[bottom_P_subj,r])
    t,p = scipy.stats.ttest_ind(x,y)
    addComparisonStat_SYM(p/2,r-0.2,r+0.2,maxH,.05,0,text_above='C>P')
plt.ylabel('p(choose cheating)')
plt.savefig('savedPlots/choices_stations_averaged_bottom.pdf')
plt.show()

score_keep_reward = score_change[:,0,:]
score_keep_noreward = score_change[:,1,:]

fig,ax = plotPosterStyle_DF_valence(score_keep_reward[:,np.newaxis,:],subjects,'score change')
plt.ylim([-1,1])
fig,ax = plotPosterStyle_DF_valence(score_keep_reward[top_subj,np.newaxis,:],subjects[top_subj],'score change')
plt.ylim([-1,1])

fig,ax = plotPosterStyle_DF_valence(score_keep_reward[bottom_subj,np.newaxis,:],subjects[bottom_subj],'score change')
plt.ylim([-1,1])

fig,ax = plotPosterStyle_DF_valence(score_keep_noreward[:,np.newaxis,:],subjects,'score change')

# what about first 3 stations run 1, last 3 stations run 3
# first_choice = np.nanmean(all_choices[:,np.arange(3),0],axis=1)
# last_choice = np.nanmean(all_choices[:,np.arange(4,nStations),3],axis=1)
# final_score = last_choice - first_choice
# fig,ax = plotPosterStyle_DF(final_score,subjects)
# x,y=nonNan(final_score[C_ind],final_score[P_ind])
# t,p = scipy.stats.ttest_ind(x,y)
# maxH=0.5
# addComparisonStat_SYM(p/2,0-0.2,0+0.2,maxH,.05,0,text_above='C>P')
# plt.show()




fig,ax = plotPosterStyle_DF_valence(p_stay_win[:,np.newaxis,:],subjects,'p(stay|win)')
plt.savefig('savedPlots/p_stay_win.pdf')

fig,ax = plotPosterStyle_DF_valence(p_stay_win[keep_subj,np.newaxis,:],subjects[keep_subj],'p(stay|win)')
plt.savefig('savedPlots/p_stay_win_keep.pdf')

p_stay_lose = 1 - p_shift_lose
fig,ax = plotPosterStyle_DF_valence(p_stay_lose[:,np.newaxis,:],subjects,'p(stay|lose)')
plt.savefig('savedPlots/p_stay_lose.pdf')

fig,ax = plotPosterStyle_DF_valence(p_stay_lose[keep_subj,np.newaxis,:],subjects[keep_subj],'p(stay|lose)')
plt.savefig('savedPlots/p_stay_lose_keep.pdf')



fig,ax = plotPosterStyle_DF_valence(p_shift_lose[:,np.newaxis,:],subjects,'p(shift|lose)')
plt.savefig('savedPlots/p_shift_lose.pdf')

fig,ax = plotPosterStyle_DF_valence(p_shift_lose[keep_subj,np.newaxis,:],subjects[keep_subj],'p(shift|lose)')
plt.savefig('savedPlots/p_shift_lose_keep.pdf')


fig,ax = plotPosterStyle_DF_valence(p_stay_win[top_subj,np.newaxis,:],subjects[top_subj],'p(stay|win)')
fig,ax = plotPosterStyle_DF_valence(p_stay_win[bottom_subj,np.newaxis,:],subjects[bottom_subj],'p(stay|win)')
fig,ax = plotPosterStyle_DF_valence(p_shift_lose[top_subj,np.newaxis,:],subjects[top_subj],'p(shift|lose)')
fig,ax = plotPosterStyle_DF_valence(p_shift_lose[bottom_subj,np.newaxis,:],subjects[bottom_subj],'p(shift|lose)')


values = np.concatenate((behavioral_cheating[:,np.newaxis],behavioral_paranoid[:,np.newaxis]),axis=1)
fig,ax = plotPosterStyle_DF(values[bottom_subj],subjects[bottom_subj])
plt.show()

################################## SEPARATE BY TOP AND BOTTOM GROUPS NOW ################################
all_choices_correct = all_choices.copy() # change into being correct and incorrect instead
for s in np.arange(nSubs):
    this_sub_interpretation = interpretations[s]
    for r in np.arange(nRuns):
        for st in np.arange(nStations):
            if all_choices[s,st,r] == 1: # if they chose the cheating response on that run
                if this_sub_interpretation == 'C':
                    correct = 1
                elif this_sub_interpretation == 'P':
                    correct = 0
            elif all_choices[s,st,r] == 0: # if they chose teh paranoid response 
                if this_sub_interpretation == 'C':
                    correct = 0
                elif this_sub_interpretation == 'P':
                    correct = 1
            all_choices_correct[s,st,r] = correct

fig = plotPosterStyle_multiplePTS(all_choices_correct,subjects)
plt.subplot(1,4,1)
#plt.yticks(np.array([0,1]), [ 'p(incor)','p(cor)']) 
plt.ylabel('p(cor')
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.yticks([])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.yticks([])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.yticks([])
plt.xlabel('station')
plt.savefig('savedPlots/choices_stations_correct.pdf')
plt.show()

#### COMBINE GROUPS TO LOOK AT CORRECT CHOICES
fig,ax = plt.subplots(figsize=(20,9))
for d in np.arange(nRuns):
  plt.subplot(1,nRuns,d+1)
  sns.despine()
  nPoints = nStations
  plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_choices_correct[:,:,d],axis=0),yerr=scipy.stats.sem(all_choices_correct[:,:,d],axis=0),color='k',alpha=0.7,lw=3,fmt='-o',ms=10)
  plt.xlabel('point')
  #plt.ylabel('area under -0.1')
  plt.xticks(np.arange(nPoints))
plt.subplot(1,4,1)
#plt.yticks(np.array([0,1]), [ 'p(incor)','p(cor)']) 
plt.ylabel('p(correct choice)')
plt.ylim([0,1.15])
# test significance across all points?
cor = nStations*nRuns
for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[:,st,0],[])
    t,p = scipy.stats.ttest_1samp(x,0.5)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.yticks([])
plt.ylim([0,1.15])
for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[:,st,0],[])
    t,p = scipy.stats.ttest_1samp(x,0.5)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.ylim([0,1.15])
plt.yticks([])
for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[:,st,0],[])
    t,p = scipy.stats.ttest_1samp(x,0.5)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[:,st,0],[])
    t,p = scipy.stats.ttest_1samp(x,0.5)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
plt.yticks([])
plt.ylim([0,1.15])
#plt.xlabel('station')
#plt.legend()
plt.savefig('savedPlots/choices_stations_correct_incor_ALL.pdf')
plt.show()


# combine into one group mean for top and bototm and use those as different colors
fig,ax = plt.subplots(figsize=(20,9))
for d in np.arange(nRuns):
  plt.subplot(1,nRuns,d+1)
  sns.despine()
  nPoints = nStations
  plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_choices_correct[top_subj,:,d],axis=0),yerr=scipy.stats.sem(all_choices_correct[top_subj,:,d],axis=0),color='k',alpha=0.7,lw=3,label='top',fmt='-o',ms=10)
  plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_choices_correct[bottom_subj,:,d],axis=0),yerr=scipy.stats.sem(all_choices_correct[bottom_subj,:,d],axis=0),color='k',alpha=0.5,lw=3,label='bottom',fmt='--X',ms=10)
  plt.xlabel('point')
  #plt.ylabel('area under -0.1')
  plt.xticks(np.arange(nPoints))
plt.subplot(1,4,1)
plt.ylim([0,1.15])

#plt.yticks(np.array([0,1]), [ 'p(incor)','p(cor)']) 
plt.ylabel('p(correct choice)')
# test significance across all points?
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
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.ylim([0,1.15])

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
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.yticks([])
plt.ylim([0,1.15])

for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[top_subj,st,2],all_choices_correct[bottom_subj,st,2],)
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.ylim([0,1.15])

for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[top_subj,st,3],all_choices_correct[bottom_subj,st,3],)
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
plt.yticks([])
#plt.xlabel('station')
#plt.legend()
plt.savefig('savedPlots/choices_stations_correct_incor.pdf')
plt.show()
######### BEHAVIOR SEPARATION
fig,ax = plt.subplots(figsize=(20,9))

sns.despine()
sns.barplot(data=df,x='bestworst',y='corr_context',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='corr_context',hue='group',split=False,palette=P2,size=8,alpha=0.5)
plt.xlabel('')
plt.legend('')
r,p = scipy.stats.ttest_ind(all_correct_context[top_subj],all_correct_context[bottom_subj])
maxH = 1
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='best>worst')
plt.ylabel('context score')
plt.yticks(np.array([-1,1]), ['incorrect','correct'],fontsize=20,rotation=45) 
plt.savefig('savedPlots/context_score_correct_incorrect.pdf')
plt.show()


fig,ax = plt.subplots(figsize=(20,9))

sns.despine()
sns.barplot(data=df,x='bestworst',y='arthur_minus_lee_cor',ci=68,linewidth=2.5,color='k',alpha=0.5)#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='bestworst',y='arthur_minus_lee_cor',hue='group',split=False,palette=P2,size=8,alpha=0.5)
plt.xlabel('')
plt.legend('')
r,p = scipy.stats.ttest_ind(arthur_minus_lee_cor[top_subj],arthur_minus_lee_cor[bottom_subj])
maxH = 7
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='best>worst')
plt.ylabel('empathy diff (Arthur - Lee)')
#plt.yticks(np.array([-1,1]), ['incorrect','correct'],fontsize=20,rotation=45) 
plt.savefig('savedPlots/arthur_minus_lee_cor_correct_incorrect.pdf')
plt.show()


fig,ax = plotPosterStyle_DF(scores[top_subj,1],subjects[top_subj])
plt.xticks(np.array([-.2,.2]), ['paranoid','cheating'],fontsize=20) 
plt.ylabel('context score')
plt.xlabel('group')
plt.title('context score')
plt.yticks(np.array([-1,1]), ['paranoid','cheating'],fontsize=20,rotation=45) 
r,p = scipy.stats.ttest_ind(scores[top_P_subj,1],scores[top_C_subj,1])
maxH = 1
addComparisonStat_SYM(p/2,-.2,.2,maxH,.05,0,text_above='C > P')
plt.ylim([-1.2,1.4])
plt.savefig('savedPlots/context_score_TOP.pdf')
plt.show()

# instead plot correct context - for top and worst plot correct side of context
data = {}
data['bestworst']
fig,ax = plt.subplots(figsize=(20,9))
sns,despine()
sns.barplot()


fig = plotPosterStyle_multiplePTS(all_choices_correct[top_subj,:,:],subjects[top_subj])
plt.subplot(1,4,1)
#plt.yticks(np.array([0,1]), [ 'p(incor)','p(cor)']) 
plt.ylabel('p(cor')
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.yticks([])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.yticks([])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.yticks([])
plt.xlabel('station')
plt.savefig('savedPlots/choices_stations_correct_TOP.pdf')
plt.show()

fig = plotPosterStyle_multiplePTS(all_choices_correct[bottom_subj,:,:],subjects[bottom_subj])
plt.subplot(1,4,1)
#plt.yticks(np.array([0,1]), [ 'p(incor)','p(cor)']) 
plt.ylabel('p(cor')
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.yticks([])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.yticks([])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.yticks([])
plt.xlabel('station')
plt.savefig('savedPlots/choices_stations_correct_BOTTOM.pdf')
plt.show()


fig = plotPosterStyle_multiplePTS(all_choices,subjects)
plt.subplot(1,4,1)
plt.yticks(np.array([0,1]), [ 'paranoid','cheating']) 
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.yticks([])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.yticks([])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.yticks([])
plt.xlabel('station')
#plt.savefig('savedPlots/choices_stations.pdf')
plt.show()

# now show for kept subjects
fig = plotPosterStyle_multiplePTS(all_choices[top_subj,:,:],subjects[top_subj])
plt.subplot(1,4,1)
plt.yticks(np.array([0,1]), [ 'paranoid','cheating']) 
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.yticks([])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.yticks([])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.yticks([])
plt.xlabel('station')
plt.savefig('savedPlots/choices_stations_TOP.pdf')
plt.show()

fig = plotPosterStyle_multiplePTS(all_choices[bottom_subj,:,:],subjects[bottom_subj])
plt.subplot(1,4,1)
plt.yticks(np.array([0,1]), [ 'paranoid','cheating']) 
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.yticks([])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.yticks([])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.yticks([])
plt.xlabel('station')
plt.savefig('savedPlots/choices_stations_TOP.pdf')
plt.show()

# plot P_switch for each by group
# make 20 subjects by two columns
fig = plotPosterStyle_multiplePTS(np.concatenate((p_stay_win[:,np.newaxis,:],p_shift_lose[:,np.newaxis,:]),axis=1),subjects)
plt.subplot(1,4,1)
plt.ylim([0,1])
plt.title('run 1')
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.subplot(1,4,2)
plt.yticks([])
plt.title('run 2')
plt.ylim([0,1])
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.subplot(1,4,3)
plt.yticks([])
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.title('run 3')
plt.ylim([0,1])
plt.subplot(1,4,4)
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.title('run 4')
plt.ylim([0,1])
plt.yticks([])
plt.show()

fig = plotPosterStyle_multiplePTS(np.concatenate((p_stay_win[top_subj,np.newaxis,:],p_shift_lose[top_subj,np.newaxis,:]),axis=1),subjects[top_subj])
plt.subplot(1,4,1)
plt.ylim([0,1])
plt.title('run 1')
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.subplot(1,4,2)
plt.yticks([])
plt.title('run 2')
plt.ylim([0,1])
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.subplot(1,4,3)
plt.yticks([])
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.title('run 3')
plt.ylim([0,1])
plt.subplot(1,4,4)
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.title('run 4')
plt.ylim([0,1])
plt.yticks([])
plt.show()

fig = plotPosterStyle_multiplePTS(np.concatenate((p_stay_win[bottom_subj,np.newaxis,:],p_shift_lose[bottom_subj,np.newaxis,:]),axis=1),subjects[bottom_subj])
plt.subplot(1,4,1)
plt.ylim([0,1])
plt.title('run 1')
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.subplot(1,4,2)
plt.yticks([])
plt.title('run 2')
plt.ylim([0,1])
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.subplot(1,4,3)
plt.yticks([])
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.title('run 3')
plt.ylim([0,1])
plt.subplot(1,4,4)
plt.xticks(np.array([0,1]), [ 'stay|win','shift|lose'],fontsize=10) 
plt.title('run 4')
plt.ylim([0,1])
plt.yticks([])
plt.show()



fig = plotPosterStyle_multiplePTS(score_change[bottom_subj,:,:],subjects[bottom_subj])
plt.subplot(1,4,1)
plt.ylim([-.5,.5])
plt.title('run 1')
plt.xticks(np.array([0,1]), [ 'correct','incorrect'],fontsize=10) 
plt.subplot(1,4,2)
plt.yticks([])
plt.title('run 2')
plt.ylim([-.5,.5])
plt.xticks(np.array([0,1]), [ 'correct','incorrect'],fontsize=10) 
plt.subplot(1,4,3)
plt.yticks([])
plt.xticks(np.array([0,1]), [ 'correct','incorrect'],fontsize=10) 
plt.title('run 3')
plt.ylim([-.5,.5])
plt.subplot(1,4,4)
plt.xticks(np.array([0,1]), [ 'correct','incorrect'],fontsize=10) 
plt.title('run 4')
plt.ylim([-.5,.5])
plt.yticks([])
plt.show()
