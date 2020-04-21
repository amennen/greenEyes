# purpose: think of ways that you can prove NF is causing the difference in empathy between the two groups
# 1. look if NF scores are anticorrelated between groups



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
        this_station_TRs = np.array(stationDict[st]) # removing this + 1 # going from python --> matlab indexing
        recorded_TRs[this_station_TRs - 3] = st +1 # this is the literal matlab TRs
    return recorded_TRs

def checkProbeTiming(behavData,recorded_TRs):
    # timing check
    timing = behavData['timing'] 
    leftRT = runData['leftPressRT']
    rightRT = runData['rightPressRT']
    TRstart = timing['actualOnsets']['story']  
    display_TR = np.argwhere(recorded_TRs == 6+1)[0][0]
    pressRT = rightRT[~np.isnan(rightRT)]
    # check time lapse between press and when the display was shown
    pressRT[-1] - TRstart[display_TR]  
    # check that it happened after the probe
    pressRT[-1] - timing['actualOnsets']['probeON'][-1] 
    # in this case for some reason the screen flipped 1 TR late so teh display happened 3TRs later
    for st in np.arange(nStations):
        display_TR = np.argwhere(recorded_TRs == st+1)[0][0] # first display
        #print(timing['actualOnsets']['probeOFF'][st] - TRstart[display_TR])
        #print(timing['plannedOnsets']['probeON'][0,st] - TRstart[display_TR])
        #print(timing['plannedOnsets']['probeON'][0,st] - timing['plannedOnsets']['stationREC'][0,st] ) # 4 seconds apart
        print(timing['plannedOnsets']['stationREC'][0,st] - timing['plannedOnsets']['story'][0,display_TR-1 ])
        # but here you're still indexing within python

        #print(timing['actualOnsets']['probeON'][st] - timing['plannedOnsets']['probeON'][0][st])
        #print(timing['actualOnsets']['story'][display_TR,0] - timing['plannedOnsets']['story'][0,display_TR])
    return

def getProbeKeyPresses(behavData,recorded_TRs):
    # the display for the station happens AFTER the probe press
    # so here we're checking when the first display of the station was
    # and as long as it's within 3 TRs then it counts
    # it just shouldn't happen on the same TR
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
        display_TR = np.argwhere(recorded_TRs == st+1)[0][0] # first display
        print(display_TR)
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

def calculateValueInterpretations_behavior(this_sub_response,this_sub_score,this_sub_cheating,this_sub_interpretation):
    """Here we want to calculate the subject's value based on all stations"""
    nTrials = len(this_sub_response)
    # assume equal value to start
    value_c = 0.5 # initialize value but update nTrials
    value_p = 0.5
    for t in np.arange(nTrials):
        # see if subject pressed
        response = this_sub_response[t]
        if response == 'CHEATING':
            # now update value
            value_c = value_c + (this_sub_score[t]-1)
        elif response == 'PARANOID':
            value_p = value_p + (this_sub_score[t]-1)
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

        # if response == 'CHEATING':
        #     param = this_sub_cheating[t]
        #     if this_sub_cheating[t] >=0.5:
        #         # consistent
        #         consistent = 1
        #     else: # inconsistent
        #         consistent = -1

        # elif response == 'PARANOID':
        #     param = 1 - this_sub_cheating[t]
        #     if this_sub_cheating[t] >=0.5: 
        #         # inconsistent
        #         consistent = -1
        #     else: # consistent
        #         consistent = 1
        # if response == 'CHEATING':
        #     # now update value, but also update based on if classification score was right or wrong

        #     value_c = value_c + (this_sub_score[t]-param)*consistent
        # elif response == 'PARANOID':
        #     value_p = value_p + (this_sub_score[t]-param)*consistent
    
def lookUpStates(run_behav,run_c_prob,run_z_score):
    nStatios = len(run_behav)
    all_states = np.zeros((nStations,))*np.nan
    for st in np.arange(nStations):
        choice = run_behav[st]
        c_prob = run_c_prob[0,st]
        z_score = run_z_score[0,st]
        if choice == 'CHEATING':
            if c_prob >= 0.5:
                if z_score >= 0.5:
                    state = 1 
                else:
                    state = 2
            else:
                if z_score >= 0.5: 
                    state = 3
                else:
                    state = 4
            all_states[st] = state
        elif choice == 'PARANOID':
            if c_prob > 0.5:
                if z_score >= 0.5:
                    state = 5
                else:
                    state = 6
            else:
                if z_score >= 0.5: 
                    state = 7
                else:
                    state = 8
            all_states[st] = state
    # consistent states: 1,3 for cheating choice, 6, 8 forparanoid score
    return all_states



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
        checkProbeTiming(b,recorded_TRs)
        C_press = [i for i in np.arange(len(probe_response))  if probe_response[i]  == 'CHEATING'] 
        P_press = [i for i in np.arange(len(probe_response))  if probe_response[i]  == 'PARANOID'] 
        all_choices[s,C_press,runNum] = 1
        all_choices[s,P_press,runNum] = 0
        this_sub_response = this_sub_response + probe_response
        states[s,:,runNum] = lookUpStates(probe_response,c_prob,c_prob_z)
    ind_cheating = [i for i, x in enumerate(this_sub_response) if x == "CHEATING"]  
    ind_paranoid = [i for i, x in enumerate(this_sub_response) if x == "PARANOID"]  
    subj_means[s,0] = np.nanmean(this_sub_cheating[ind_cheating])
    subj_means[s,1] = np.nanmean(this_sub_cheating[ind_paranoid])
    behavioral_cheating[s],behavioral_paranoid[s] = calculateValueInterpretations_behavior(this_sub_response,this_sub_score,this_sub_cheating,this_sub_interpretation)
    neural_cheating[s],neural_paranoid[s] = calculateValueInterpretations_neural(this_sub_response,this_sub_score,this_sub_cheating,this_sub_interpretation)

classifier_separation = subj_means[:,0] - subj_means[:,1]
# sort top order
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
plt.savefig('savedPlots/choices_stations.pdf')
plt.show()

fig = plotPosterStyle_multiplePTS(all_choices[top_subj,:,:],subjects[top_subj])
plt.subplot(1,4,1)
plt.yticks(np.array([0,1]), [ 'paranoid','cheating']) 
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.title('run 2')
plt.yticks([])
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
plt.title('run 2')
plt.xlabel('station')
plt.yticks([])
plt.subplot(1,4,3)
plt.title('run 3')
plt.xlabel('station')
plt.yticks([])
plt.subplot(1,4,4)
plt.title('run 4')
plt.xlabel('station')
plt.yticks([])
plt.savefig('savedPlots/choices_stations_BOTTOM.pdf')
plt.show()



fig = plotPosterStyle_multiplePTS(all_cheating_prob,subjects)
plt.subplot(1,4,1)
plt.ylabel('p(cheating)')
plt.ylim([0,1])
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.ylim([0,1])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.ylim([0,1])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.ylim([0,1])
plt.xlabel('station')
plt.savefig('savedPlots/p_cheating.pdf')
plt.show()

fig = plotPosterStyle_multiplePTS(all_cheating_prob[top_subj],subjects[top_subj])
plt.subplot(1,4,1)
plt.ylabel('p(cheating)')
plt.ylim([0,1])
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.ylim([0,1])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.ylim([0,1])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.ylim([0,1])
plt.xlabel('station')
plt.savefig('savedPlots/p_cheating_TOP.pdf')
plt.show()

fig = plotPosterStyle_multiplePTS(all_cheating_prob[bottom_subj],subjects[bottom_subj])
plt.subplot(1,4,1)
plt.ylabel('p(cheating)')
plt.ylim([0,1])
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.ylim([0,1])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.ylim([0,1])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.ylim([0,1])
plt.xlabel('station')
plt.savefig('savedPlots/p_cheating_BOTTOM.pdf')
plt.show()


fig = plotPosterStyle_multiplePTS(all_cheating_prob_z,subjects)
plt.subplot(1,4,1)
plt.ylabel('z-scored(p(cheating))')
plt.ylim([0,1])
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.ylim([0,1])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.ylim([0,1])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.ylim([0,1])
plt.xlabel('station')
plt.savefig('savedPlots/p_cheating_zscored.pdf')
plt.show()

fig = plotPosterStyle_multiplePTS(all_cheating_prob_z[top_subj],subjects[top_subj])
plt.subplot(1,4,1)
plt.ylabel('z-scored(p(cheating))')
plt.ylim([0,1])
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.ylim([0,1])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.ylim([0,1])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.ylim([0,1])
plt.xlabel('station')
plt.savefig('savedPlots/p_cheating_zscored_TOP.pdf')
plt.show()

fig = plotPosterStyle_multiplePTS(all_cheating_prob_z[bottom_subj],subjects[bottom_subj])
plt.subplot(1,4,1)
plt.ylabel('z-scored(p(cheating))')
plt.ylim([0,1])
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.ylim([0,1])
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.ylim([0,1])
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.ylim([0,1])
plt.xlabel('station')
plt.savefig('savedPlots/p_cheating_zscored_BOTTOM.pdf')
plt.show()


fig = plotPosterStyle_multiplePTS(all_nf_score,subjects)
plt.subplot(1,4,1)
plt.ylabel('NF score ($)')
plt.title('run 1')
plt.ylim([0,1])
plt.xlabel('station')
plt.subplot(1,4,2)
plt.title('run 2')
plt.ylim([0,1])
plt.xlabel('station')
plt.subplot(1,4,3)
plt.title('run 3')
plt.ylim([0,1])
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.ylim([0,1])
plt.xlabel('station')
plt.savefig('savedPlots/nf_score.pdf')
plt.show()

fig = plotPosterStyle_multiplePTS(all_nf_score[top_subj],subjects[top_subj])
plt.subplot(1,4,1)
plt.ylabel('NF score ($)')
plt.title('run 1')
plt.ylim([0,1])
plt.xlabel('station')
plt.subplot(1,4,2)
plt.title('run 2')
plt.ylim([0,1])
plt.xlabel('station')
plt.subplot(1,4,3)
plt.title('run 3')
plt.ylim([0,1])
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.ylim([0,1])
plt.xlabel('station')
plt.savefig('savedPlots/nf_score_TOP.pdf')
plt.show()

fig = plotPosterStyle_multiplePTS(all_nf_score[bottom_subj],subjects[bottom_subj])
plt.subplot(1,4,1)
plt.ylabel('NF score ($)')
plt.title('run 1')
plt.ylim([0,1])
plt.xlabel('station')
plt.subplot(1,4,2)
plt.title('run 2')
plt.ylim([0,1])
plt.xlabel('station')
plt.subplot(1,4,3)
plt.title('run 3')
plt.ylim([0,1])
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.ylim([0,1])
plt.xlabel('station')
plt.savefig('savedPlots/nf_score_BOTTOM.pdf')
plt.show()
###### SEPARATE INTO GOOD AND BAD 




consistent_states = np.zeros((nSubs,nStations,nRuns))*np.nan
consistent_states[states==1]=1
consistent_states[states==3]=1
consistent_states[states==6]=1
consistent_states[states==8]=1
consistent_states[states==2]=0
consistent_states[states==4]=0
consistent_states[states==5]=0
consistent_states[states==6]=0
fig = plotPosterStyle_multiplePTS(consistent_states,subjects)
plt.subplot(1,4,1)
plt.ylabel('p(consistent)')
plt.title('run 1')
plt.xlabel('station')
plt.subplot(1,4,2)
plt.title('run 2')
plt.xlabel('station')
plt.subplot(1,4,3)
plt.title('run 3')
plt.xlabel('station')
plt.subplot(1,4,4)
plt.title('run 4')
plt.xlabel('station')
plt.show()

plt.subplots(figsize=(50,10))
for r in np.arange(nRuns):
    plt.subplot(1,4,r+1)
    sns.despine()

    for s in np.arange(nSubs):
        if interpretations[s] == 'C':
            color='r'
        else:
            color='g'
        plt.plot(states[s,:,r],color=color,alpha=0.1)
plt.show()

# behavior plot only
# states 1 - 4 are for cheating
# states 5 - 8 are for paranoid
states_behavior = states.copy()
states_behavior[states<=4] = 1
states_behavior[states>4] = 2

plt.subplots(figsize=(50,10))
for r in np.arange(nRuns):
    plt.subplot(1,4,r+1)
    sns.despine()

    for s in np.arange(nSubs):
        if interpretations[s] == 'C':
            color='r'
        else:
            color='g'
        plt.plot(states_behavior[s,:,r],color=color,alpha=0.3)
plt.show()

# first, how corrleated is each person across run
average_score_across_run = np.nanmean(all_nf_score,axis=2)
average_score_cheating = np.nanmean(average_score_across_run[C_ind,:],axis=0)
average_score_paranoid = np.nanmean(average_score_across_run[P_ind,:],axis=0)
scipy.stats.pearsonr(average_score_cheating,average_score_paranoid)
corr_by_run = np.zeros((nRuns,2))
corr_by_run_cheating_prob = np.zeros((nRuns,2))
corr_by_run_cheating_prob_z = np.zeros((nRuns,2))
# plt.plot()
within_c = np.zeros((nRuns,2))
within_p = np.zeros((nRuns,2))
across = np.zeros((nRuns,2))
for r in np.arange(nRuns):
    cheating_avg = np.nanmean(all_nf_score[C_ind,:,r],axis=0)
    paranoid_avg = np.nanmean(all_nf_score[P_ind,:,r],axis=0)
    corr_by_run[r,0],corr_by_run[r,1] = scipy.stats.pearsonr(cheating_avg,paranoid_avg)

    cheating_avg = np.nanmean(all_cheating_prob[C_ind,:,r],axis=0)
    paranoid_avg = np.nanmean(all_cheating_prob[P_ind,:,r],axis=0)
    # plt.plot(cheating_avg, color='r', lw=r)
    # plt.plot(paranoid_avg,color='g',lw=r)
    corr_by_run_cheating_prob[r,0],corr_by_run_cheating_prob[r,1] = scipy.stats.pearsonr(cheating_avg,paranoid_avg)
    cheating_avg = np.nanmean(all_cheating_prob_z[C_ind,:,r],axis=0)
    paranoid_avg = np.nanmean(all_cheating_prob_z[P_ind,:,r],axis=0)
    corr_by_run_cheating_prob_z[r,0],corr_by_run_cheating_prob_z[r,1] = scipy.stats.pearsonr(cheating_avg,paranoid_avg)

    all_score= all_nf_score[:,:,r].T
    # what if instead it's brain behavior?
    all_c_prob = all_cheating_prob[:,:,r].T
    all_c_prob_z = all_cheating_prob_z[:,:,r].T
    df = pd.DataFrame(all_nf_score[:,:,r]).T
    all_pairwise_corr = np.array(df.corr(method='pearson'))
    all_pairwise_corr[np.tril_indices(np.shape(all_pairwise_corr)[0])]=np.nan
    [x,y] = np.meshgrid(C_ind,C_ind)
    within_c[r,0] = np.nanmean(all_pairwise_corr[x,y])
    within_c[r,1] = np.nanstd(all_pairwise_corr[x,y])
    [x,y] = np.meshgrid(P_ind,P_ind)
    within_p[r,0] = np.nanmean(all_pairwise_corr[x,y])
    within_p[r,1] = np.nanstd(all_pairwise_corr[x,y])
    [x,y] = np.meshgrid(C_ind,P_ind)
    across[r,0] = np.nanmean(all_pairwise_corr[x,y])
    across[r,1] = np.nanstd(all_pairwise_corr[x,y])


# plot within/across group correlation
colors_dark = ['#2ca25f','#de2d26']
colors_light = ['#2ca25f','#de2d26']
nsubs = len(subjects)

fig,ax = plt.subplots(figsize=(17,9))
sns.despine()
plt.errorbar(x=np.arange(nRuns),y=within_c[:,0],yerr=within_c[:,1],color=colors_dark[1],alpha=0.5,lw=5,label='within C',fmt='-o',ms=10)
plt.errorbar(x=np.arange(nRuns),y=within_p[:,0],yerr=within_p[:,1],color=colors_dark[0],alpha=0.5,lw=5,label='within P',fmt='-o',ms=10)
plt.errorbar(x=np.arange(nRuns),y=across[:,0],yerr=across[:,1],color='k',alpha=0.5,lw=5,label='across C-P',fmt='-o',ms=10)
plt.legend()
plt.xlabel('run')
plt.ylabel('average correlation')
plt.title('neurofeedback score correlation')
plt.ylim([-1,1.2])
#plt.ylabel('area under -0.1')
plt.xticks(np.arange(nRuns))
plt.show()


# so all paranoid, for each subject, goes through all runs
subject_vector = np.repeat(np.arange(nSubs),nRuns)
run_vector = np.tile(np.arange(nRuns),nSubs)
group_vector = ['P']*10*nRuns+['C']*10*nRuns
data = {}
data['corr'] = all_within
data['subject'] = subject_vector
data['run'] = run_vector
data['group'] = group_vector
df = pd.DataFrame.from_dict(data)
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
P1 = makeColorPalette(['#2ca25f','#de2d26']) # COLORS ARE PARANOID THEN CHEATING
P2 = makeColorPalette(['#99d8c9','#fc9272'])
P3 = makeColorPalette(['#e5f5f9','#fee0d2'])
#sns.set_palette(sns.color_palette(colors))
sns.barplot(data=df,x='run',y='corr',hue='group',ci=68,linewidth=2.5,palette=P3)#errcolor=".2", edgecolor=".2")
#sns.barplot(data=df,x='day',y='data',hue='group',ci=68,linewidth=2.5,palette=P1,errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='run',y='corr',hue='group',split=True,palette=P1,size=8,alpha=0.5)
ax.get_legend().remove()
plt.title('z-score within group correlation')
plt.show()

all_across = np.concatenate((across_p.flatten(),across_c.flatten()),axis=0)
data = {}
data['corr'] = all_across
data['subject'] = subject_vector
data['run'] = run_vector
data['group'] = group_vector
df = pd.DataFrame.from_dict(data)
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
P1 = makeColorPalette(['#2ca25f','#de2d26']) # COLORS ARE PARANOID THEN CHEATING
P2 = makeColorPalette(['#99d8c9','#fc9272'])
P3 = makeColorPalette(['#e5f5f9','#fee0d2'])
#sns.set_palette(sns.color_palette(colors))
sns.barplot(data=df,x='run',y='corr',hue='group',ci=68,linewidth=2.5,palette=P3)#errcolor=".2", edgecolor=".2")
#sns.barplot(data=df,x='day',y='data',hue='group',ci=68,linewidth=2.5,palette=P1,errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='run',y='corr',hue='group',split=True,palette=P1,size=8,alpha=0.5)
ax.get_legend().remove()
plt.show()


# get final response
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

# change context scores
z=all_context_scores*12
# now everyone answered "paranoid" for that so I must have subtracted one too many then
new_context_scores = (z.copy() + 1)/11
all_correct_context = new_context_scores.copy()
all_correct_context[P_ind] = -1*new_context_scores[P_ind]
all_context_scores = new_context_scores
arthur_minus_lee = all_rating_scores[:,0] - all_rating_scores[:,1]
# if cheating, empathize
arthur_minus_lee_cor = arthur_minus_lee.copy()
arthur_minus_lee_cor[P_ind] = -1*arthur_minus_lee[P_ind]




behav_diff = behavioral_cheating - behavioral_paranoid
neural_diff = neural_cheating/neural_paranoid
plt.figure()
plt.scatter(behav_diff,all_context_scores, color='k', label='context score')
plt.scatter(behav_diff,arthur_minus_lee, color='b', label='empathy diff')
plt.xlabel('behavior learning')
plt.ylabel('interpretation score')
plt.legend()
plt.show()
behav_diff_cheating = np.argwhere(behav_diff>0)[:,0]
behav_diff_paranoid = np.argwhere(behav_diff<0)[:,0]
scipy.stats.ttest_ind(arthur_minus_lee[behav_diff_cheating],arthur_minus_lee[behav_diff_paranoid])
scipy.stats.ttest_ind(all_context_scores[behav_diff_cheating],all_context_scores[behav_diff_paranoid])

plt.figure()
for s in np.arange(nSubs):
    #plt.scatter(neural_diff,all_context_scores, color='k', label='context score')
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.scatter(neural_diff[s],arthur_minus_lee[s], color=color, label='empathy diff')
plt.xlabel('neural learning')
plt.ylabel('interpretation score')
#plt.legend()
plt.show()
scipy.stats.pearsonr(neural_paranoid,arthur_minus_lee)
neural_diff_cheating = np.argwhere(neural_diff>0)[:,0]
neural_diff_paranoid = np.argwhere(neural_diff<0)[:,0]

scipy.stats.ttest_ind(arthur_minus_lee[neural_diff_cheating],arthur_minus_lee[neural_diff_paranoid])
scipy.stats.ttest_ind(all_context_scores[neural_diff_cheating],all_context_scores[neural_diff_paranoid])

# can put into model to solve but probably nothing significant
combined_cheating = behavioral_cheating + neural_cheating
combined_paranoid = behavioral_paranoid + neural_paranoid
combined_diff = combined_cheating - combined_paranoid
plt.figure()
plt.scatter(combined_diff,all_context_scores, color='k', label='context score')
plt.scatter(combined_diff,arthur_minus_lee, color='b', label='empathy diff')
plt.xlabel('combined learning')
plt.ylabel('interpretation score')
plt.legend()
plt.show()

consistency_mean_run = np.nanmean(consistent_states,axis=2)
consistency_mean_station = np.nanmean(consistency_mean_run,axis=1)
plt.figure()
for s in np.arange(nSubs):
    #plt.scatter(neural_diff,all_context_scores, color='k', label='context score')
    if interpretations[s] == 'C':
        color='r'
    elif interpretations[s] == 'P':
        color='g'
    plt.scatter(consistency_mean_station[s],arthur_minus_lee_cor[s], color=color, label='empathy diff')
plt.xlabel('neural learning')
plt.ylabel('interpretation score')
#plt.legend()
plt.show()
scipy.stats.pearsonr(consistency_mean_station,all_correct_context)

