# Purpose: see if good/bad people deflected more

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
font = {'size': 22,
        'weight': 'normal'}
plt.rc('axes', linewidth=5)
plt.rc('xtick.major', size=10, width = 4)
plt.rc('ytick.major', size=10, width = 4)

# define plot vars 
lw = 8
ms = 10
alpha = 0.8

defaultConfig = os.path.join(os.getcwd(), 'conf/greenEyes_cluster.toml')
cfg = loadConfigFile(defaultConfig)
params = StructDict({'config':defaultConfig, 'runs': '1', 'scans': '9', 'webpipe': 'None', 'webfilesremote': False})
cfg = greenEyes.initializeGreenEyes(defaultConfig,params)


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
        prob_resopnse_st = np.nan
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
            probe_response_st = np.nan
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

interpretations = {}
for s in np.arange(nSubs):
    interpretations[s] = getSubjectInterpretation(subjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
group_str = [''] * nSubs
for s in np.arange(nSubs):
    if s in C_ind:
        group_str[s] = 'CHEATING'
    elif s in P_ind:
        group_str[s] = 'PARANOID'

def calculate_prob_stats(subjects):
    nRuns = 4
    all_cheating_prob = np.zeros((nSubs,nStations,nRuns))*np.nan
    all_choices = np.zeros((nSubs,nStations,nRuns))*np.nan
    subj_means = np.zeros((nSubs,2))
    all_nf_score = np.zeros((nSubs,nStations,nRuns))*np.nan

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
            all_cheating_prob[s,:,runNum] = c_prob
            this_sub_cheating = np.append(this_sub_cheating,c_prob[0,:])
            all_nf_score[s,:,runNum] = nf_score
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
    return all_cheating_prob, all_choices, subj_means, all_nf_score

def calculate_clf_sep(subj_means, P_ind, C_ind):
    # classifier separation is the difference of p(cheating) for each response
    classifier_separation = subj_means[:,0] - subj_means[:,1]
    print('classifier separation')
    print(classifier_separation)
    print('n correct side')
    all_correct_classification = np.argwhere(classifier_separation>0)[:,0]
    print(len(all_correct_classification))
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
    return classifier_separation, top_subj, bottom_subj


# next get the average classification and plot over all runs
cfg.station_stats = cfg.classifierDir + 'station_stats.npz'
a = np.load(cfg.station_stats)
all_means = a['m']


P2 = makeColorPalette(['#99d8c9','#fc9272'])
paranoid_c = '#99d8c9'
cheating_c = '#fc9272'

if __name__ == '__main__':

    #######################################################################################
    all_cheating_prob, all_choices, subj_means, all_nf_score = calculate_prob_stats(subjects)
    classifier_separation, top_subj, bottom_subj = calculate_clf_sep(subj_means, P_ind, C_ind)

    #######################################################################################
    # first show all subjects
    fig = plotPosterStyle_multiplePTS(all_cheating_prob,subjects,alpha=alpha)
    plt.subplot(1,4,1)
    plt.ylabel('',fontsize=25)
    plt.plot(np.arange(nStations),all_means,'--',color='k',alpha=0.5, linewidth=lw-1)
    plt.ylim([0,1])
    plt.xticks(np.arange(nStations), fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.title('',fontsize=30) # run1
    plt.xlabel('',fontsize=25) #station
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    plt.subplot(1,4,2)
    plt.xticks(np.arange(nStations), fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    plt.plot(np.arange(nStations),all_means,'--',color='k',alpha=0.5, linewidth=lw-1)
    plt.ylim([0,1])
    plt.title('',fontsize=30)
    plt.xlabel('',fontsize=25)
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    plt.subplot(1,4,3)
    plt.xticks(np.arange(nStations), fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    plt.ylim([0,1])
    plt.title('',fontsize=30)
    plt.plot(np.arange(nStations),all_means,'--',color='k',alpha=0.5, linewidth=lw-1)
    plt.xlabel('',fontsize=25)
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    plt.subplot(1,4,4)
    plt.title('',fontsize=30)
    plt.plot(np.arange(nStations),all_means,'--',color='k',alpha=0.5, linewidth=lw-1)
    plt.xticks(np.arange(nStations), fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    plt.ylim([0,1])
    plt.xlabel('',fontsize=25)
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])


    #plt.show()
    plt.savefig('savedPlots_checked/cprob_deflections.pdf')
    plt.close(fig)
    # calculate statistics - new comment 12/4/21
    # all_cheating_prob: n_subs x n_stations x nruns
    # want: for each run, each group diff
    all_cheating_prob_run = np.nanmean(all_cheating_prob, axis=1) # now just subjects x runs
    print('NEW STATISTICS OVER ALL RUNS FOR SUBJECTS')
    for i in np.arange(4):
        x,y=nonNan(all_cheating_prob_run[C_ind,i],all_cheating_prob_run[P_ind,i])
        t,p = scipy.stats.ttest_ind(x,y)
        text = f'DID CHEATING PROB DIFF SIGNIFICANTLY FOR ANY RUN\n2-sided - RUN {i}'
        printStatsResults(text, t, p, x, y)


    ##### ADD PLOT BY RUN
    fig,ax = plt.subplots(figsize=(20,9))
    sns.despine()
    plt.errorbar(
        x=np.arange(4),
        y=np.nanmean(all_cheating_prob_run[P_ind,:],axis=0),
        yerr=scipy.stats.sem(all_cheating_prob_run[P_ind,:],
        axis=0,nan_policy='omit'
        ),
        color=paranoid_c,
        lw=lw,
        label='top',
        fmt='-o',
        ms=ms)
    plt.errorbar(
        x=np.arange(4),
        y=np.nanmean(all_cheating_prob_run[C_ind,:],axis=0),
        yerr=scipy.stats.sem(all_cheating_prob_run[C_ind,:],
        axis=0,nan_policy='omit'
        ),
        color=cheating_c,
        lw=lw,
        label='top',
        fmt='-o',
        ms=ms)
    plt.xlabel('',fontsize=25)
    plt.xticks(np.arange(4),fontsize=20)

    plt.ylabel('')
    plt.yticks(np.arange(0,1.25,.25))
    plt.ylim([0,1])
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    plt.savefig('savedPlots_checked/cprob_run.pdf')
    plt.close(fig)


    ##########################################################################################
    # cprob divided by top and bottom
    ##########################################################################################
    all_correct_prob = all_cheating_prob.copy()
    # reverse so subtract 1 to get paranoid prob
    all_correct_prob[P_ind,:,:] = 1 - all_cheating_prob[P_ind,:,:]
    fig,ax = plt.subplots(figsize=(20,9))
    for d in np.arange(nRuns):
      plt.subplot(1,nRuns,d+1)
      sns.despine()
      nPoints = nStations
      plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_correct_prob[top_subj,:,d],axis=0),
                 yerr=scipy.stats.sem(all_correct_prob[top_subj,:,d],axis=0,nan_policy='omit'),
                 color='k',alpha=1,lw=lw-1,label='top',fmt='-o',ms=ms)
      plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_correct_prob[bottom_subj,:,d],axis=0),
                 yerr=scipy.stats.sem(all_correct_prob[bottom_subj,:,d],axis=0,nan_policy='omit'),
                 color='k',alpha=alpha-.1,lw=lw-1,label='bottom',fmt=':o',ms=ms)
      
      plt.xlabel('',fontsize=25)
      #plt.ylabel('area under -0.1')
      plt.xticks(np.arange(nStations), fontsize=20)
      plt.yticks(np.arange(0,1.25,.25))
      plt.ylim([0,1])
      ax = plt.gca()
      ax.axes.xaxis.set_ticklabels([])
      ax.axes.yaxis.set_ticklabels([])

    plt.subplot(1,4,1)
    # test significance across all points and do Bonferroni correction
    cor = nStations*nRuns
    print('cprob correct top/bottom')
    print(f'cor is {cor}')
    for st in np.arange(nStations):
        x,y=nonNan(all_correct_prob[top_subj,st,0],all_correct_prob[bottom_subj,st,0])
        t,p = scipy.stats.ttest_ind(x,y)
        p =p * cor
        if np.mod(st,2):
            maxH = 1
        else:
            maxH = 1.05
        addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
        if p/2 < 0.1:
            text = '1-sided r %i station %i' % (0, st)
            printStatsResults(text, t, p/2, x, y)
    plt.title('',fontsize=30)
    plt.ylabel('',fontsize=25)

    plt.subplot(1,4,2)
    for st in np.arange(nStations):
        x,y=nonNan(all_correct_prob[top_subj,st,1],all_correct_prob[bottom_subj,st,1])
        t,p = scipy.stats.ttest_ind(x,y)
        p =p * cor
        if np.mod(st,2):
            maxH = 1
        else:
            maxH = 1.05
        addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
        if p/2 < 0.1:
            text = '1-sided r %i station %i' % (1, st)
            printStatsResults(text, t, p/2, x, y)
    plt.title('',fontsize=30)
    plt.ylabel('',fontsize=25)

    plt.subplot(1,4,3)
    for st in np.arange(nStations):
        x,y=nonNan(all_correct_prob[top_subj,st,2],all_correct_prob[bottom_subj,st,2])
        t,p = scipy.stats.ttest_ind(x,y)
        p =p * cor
        if np.mod(st,2):
            maxH = 1
        else:
            maxH = 1.05
        addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
        if p/2 < 0.1:
            text = '1-sided r %i station %i' % (2, st)
            printStatsResults(text, t, p/2, x, y)
    plt.ylabel('',fontsize=25)
    plt.title('',fontsize=30)

    plt.subplot(1,4,4)
    for st in np.arange(nStations):
        x,y=nonNan(all_correct_prob[top_subj,st,3],all_correct_prob[bottom_subj,st,3])
        t,p = scipy.stats.ttest_ind(x,y)
        p =p * cor
        if np.mod(st,2):
            maxH = 1
        else:
            maxH = 1.05
        addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
        if p/2 < 0.1:
            text = '1-sided r %i station %i' % (3, st)
            printStatsResults(text, t, p/2, x, y)
    plt.title('',fontsize=30)
    plt.ylabel('',fontsize=25)

    plt.savefig('savedPlots_checked/cprob_correct_incor.pdf')
    #plt.show()
    plt.close(fig)



    # NOW AVERAGE OVER ALL STATIONS IN A RUN FOR EACH SUBJECT!
    all_correct_run = np.nanmean(all_correct_prob,axis=1)
    cor = nRuns
    fig,ax = plt.subplots(figsize=(20,9))
    sns.despine()
    plt.errorbar(x=np.arange(4),y=np.nanmean(all_correct_run[top_subj,:],axis=0),yerr=scipy.stats.sem(all_correct_run[top_subj,:],
                axis=0,nan_policy='omit'),color='k',alpha=1,lw=lw-1,label='top',fmt='-o',ms=ms)
    plt.errorbar(x=np.arange(4),y=np.nanmean(all_correct_run[bottom_subj,:],axis=0),yerr=scipy.stats.sem(all_correct_run[bottom_subj,:],
                axis=0,nan_policy='omit'),color='k',alpha=alpha-.1,lw=lw-1,label='bottom',fmt=':o',ms=ms)
    plt.xlabel('',fontsize=25)
    plt.ylabel('')
    plt.xticks(np.arange(4), fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    plt.ylim([0,1])
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])


    print('NEW STATISTICS OVER ALL RUNS FOR SUBJECTS')
    print('Cheating prob - correct incorrect')
    print(f'cor is {cor}')
    for i in np.arange(4):
        x,y=nonNan(all_correct_run[top_subj,i],all_correct_run[bottom_subj,i])
        t,p = scipy.stats.ttest_ind(x,y)
        p = p * cor
        text = f'DID CHEATING PROB COR/INCOR DIFF SIGNIFICANTLY FOR ANY RUN (1-sided) - RUN {i}'
        printStatsResults(text, t, p/2, x, y)
        addComparisonStat_SYM(p/2,i,i,maxH,.05,0,text_above='')
    #plt.plot(np.arange(4),np.ones(4,)*np.mean(all_means),'--',color='k')
    plt.savefig('savedPlots_checked/cprob_correct_incor_run.pdf')
    plt.close(fig)



    ##########################################################################################
    # nurofeedback rewards divided
    ##########################################################################################


    ##########################################################################################
    # go to zero if score < 0.5 like for actual reward amount
    ##########################################################################################
    all_nf_score_reward = all_nf_score.copy()
    # set all values <= 0.5 to be 0
    all_nf_score_reward[all_nf_score<=0.5] = 0
    all_nf_score_reward[np.isnan(all_nf_score)] = np.nan
    all_nf_score_reward_run = np.nanmean(all_nf_score_reward,axis=1)

    # first do for all subjects
    # first show all subjects
    fig = plotPosterStyle_multiplePTS(all_nf_score_reward,subjects,alpha=alpha)
    plt.subplot(1,4,1)
    plt.ylabel('',fontsize=25)
    plt.ylim([0,1])
    plt.title('',fontsize=30)
    plt.xlabel('',fontsize=25)

    plt.xticks(np.arange(nStations),fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])


    plt.subplot(1,4,2)
    plt.xticks(np.arange(nStations),fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.ylim([0,1])
    plt.title('',fontsize=30)
    plt.xlabel('',fontsize=25)

    plt.subplot(1,4,3)
    plt.xticks(np.arange(nStations),fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.ylim([0,1])
    plt.title('',fontsize=30)
    plt.xlabel('',fontsize=25)

    plt.subplot(1,4,4)
    plt.title('',fontsize=30)
    plt.xticks(np.arange(nStations),fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.ylim([0,1])
    plt.xlabel('',fontsize=25)
    #plt.show()
    plt.savefig('savedPlots_checked/nf_score_reward_checked.pdf')
    plt.close(fig)

    print('NEW STATS')
    print('AVG AMOUNT OF REWARD DIFFERENT')
    x,y=nonNan(all_nf_score_reward_run[:,0], all_nf_score_reward_run[:,3],paired=True)
    t,p = scipy.stats.ttest_rel(x,y)
    text = 'Did amount of reward differ significantly (2-sided) for all subjects between first and last run?'
    printStatsResults(text, t, p, x, y)


    # PLOT BY RUN
    fig,ax = plt.subplots(figsize=(20,9))
    sns.despine()
    plt.errorbar(
        x=np.arange(4),
        y=np.nanmean(all_nf_score_reward_run[P_ind,:],axis=0),
        yerr=scipy.stats.sem(all_nf_score_reward_run[P_ind,:],
        axis=0,nan_policy='omit'
        ),
        color=paranoid_c,
        lw=lw,
        label='top',
        fmt='-o',
        ms=ms)
    plt.errorbar(
        x=np.arange(4),
        y=np.nanmean(all_nf_score_reward_run[C_ind,:],axis=0),
        yerr=scipy.stats.sem(all_nf_score_reward_run[C_ind,:],
        axis=0,nan_policy='omit'
        ),
        color=cheating_c,
        lw=lw,
        label='top',
        fmt='-o',
        ms=ms)
    plt.xlabel('',fontsize=25)
    plt.ylabel('')
    plt.ylim([0,1])
    plt.xticks(np.arange(4),fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.savefig('savedPlots_checked/nf_score_reward_run.pdf')
    plt.close(fig)


##########################################
## BY COR/INCOR
    fig,ax = plt.subplots(figsize=(20,9))
    for d in np.arange(nRuns):
      plt.subplot(1,nRuns,d+1)
      sns.despine()
      nPoints = nStations
      plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_nf_score_reward[top_subj,:,d],axis=0),
                    yerr=scipy.stats.sem(all_nf_score_reward[top_subj,:,d],axis=0,nan_policy='omit'),
                    color='k',alpha=1,lw=lw-1,label='top',fmt='-o',ms=ms)
      plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_nf_score_reward[bottom_subj,:,d],axis=0),
                    yerr=scipy.stats.sem(all_nf_score_reward[bottom_subj,:,d],axis=0,nan_policy='omit'),
                    color='k',alpha=alpha-.1,lw=lw-1,label='bottom',fmt=':o',ms=ms)
      plt.xlabel('',fontsize=25)
      #plt.ylabel('area under -0.1')
      plt.xticks(np.arange(nPoints),fontsize=20)
      plt.yticks(np.arange(0,1.25,.25))
      plt.ylim([0,1])
      ax = plt.gca()
      ax.axes.xaxis.set_ticklabels([])
      ax.axes.yaxis.set_ticklabels([])

    plt.subplot(1,4,1)
    # test significance across all points and do Bonferroni correction
    cor = nStations*nRuns
    print(f'cor is {cor}')
    for st in np.arange(nStations):
        x,y=nonNan(all_nf_score_reward[top_subj,st,0],all_nf_score_reward[bottom_subj,st,0])
        t,p = scipy.stats.ttest_ind(x,y)
        p =p * cor
        if np.mod(st,2):
            maxH = 1
        else:
            maxH = 1.05
        addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
        if p/2 < 0.1:
            text = '1-sided r %i station %i' % (0, st)
            printStatsResults(text, t, p/2, x, y)
    plt.title('',fontsize=30)
    plt.ylabel('',fontsize=25)

    plt.subplot(1,4,2)
    for st in np.arange(nStations):
        x,y=nonNan(all_nf_score_reward[top_subj,st,1],all_nf_score_reward[bottom_subj,st,1])
        t,p = scipy.stats.ttest_ind(x,y)
        p =p * cor
        if np.mod(st,2):
            maxH = 1
        else:
            maxH = 1.05
        addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
        if p/2 < 0.1:
            text = '1-sided r %i station %i' % (1, st)
            printStatsResults(text, t, p/2, x, y)
    plt.title('',fontsize=30)
    plt.ylabel('',fontsize=25)

    plt.subplot(1,4,3)
    for st in np.arange(nStations):
        x,y=nonNan(all_nf_score_reward[top_subj,st,2],all_nf_score_reward[bottom_subj,st,2])
        t,p = scipy.stats.ttest_ind(x,y)
        p =p * cor
        if np.mod(st,2):
            maxH = 1
        else:
            maxH = 1.05
        addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
        if p/2 < 0.1:
            text = '1-sided r %i station %i' % (2, st)
            printStatsResults(text, t, p/2, x, y)
    plt.title('',fontsize=30)
    plt.ylabel('',fontsize=25)

    plt.subplot(1,4,4)
    for st in np.arange(nStations):
        x,y=nonNan(all_nf_score_reward[top_subj,st,3],all_nf_score_reward[bottom_subj,st,3])
        t,p = scipy.stats.ttest_ind(x,y)
        p =p * cor
        if np.mod(st,2):
            maxH = 1
        else:
            maxH = 1.05
        addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
        if p/2 < 0.1:
            text = '1-sided r %i station %i' % (3, st)
            printStatsResults(text, t, p/2, x, y)
    plt.title('',fontsize=30)
    plt.ylabel('',fontsize=25)

    plt.savefig('savedPlots_checked/nf_score_correct_incor_actual_reward.pdf')
    #plt.show()
    plt.close(fig)

    # NOW AVERAGE OVER ALL STATIONS IN A RUN FOR EACH SUBJECT!
    cor = nRuns
    fig,ax = plt.subplots(figsize=(20,9))
    sns.despine()
    plt.errorbar(x=np.arange(4),y=np.nanmean(all_nf_score_reward_run[top_subj,:],axis=0),
                yerr=scipy.stats.sem(all_nf_score_reward_run[top_subj,:],axis=0,nan_policy='omit'),
                color='k',alpha=1,lw=lw-1,label='top',fmt='-o',ms=ms)
    plt.errorbar(x=np.arange(4),y=np.nanmean(all_nf_score_reward_run[bottom_subj,:],axis=0),
                yerr=scipy.stats.sem(all_nf_score_reward_run[bottom_subj,:],axis=0,nan_policy='omit'),
                color='k',alpha=alpha-.1,lw=lw-1,label='bottom',fmt=':o',ms=ms)
    plt.xlabel('',fontsize=25)
    plt.ylabel('')
    plt.xticks(np.arange(4),fontsize=20)
    plt.yticks(np.arange(0,1.25,.25))
    #remove tick labels
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    print("NEW RUN STATISTICS FOR NEUROFEEDBACK")
    for i in np.arange(4):
        x,y=nonNan(all_nf_score_reward_run[top_subj,i],all_nf_score_reward_run[bottom_subj,i])
        t,p = scipy.stats.ttest_ind(x,y)
        p = p * cor
        text = f'DID ACTUAL REWARD DIFF SIGNIFICANTLY (1-sided) FOR ANY RUN - RUN {i}'
        printStatsResults(text, t, p/2, x, y)
        addComparisonStat_SYM(p/2,i,i,maxH,.05,0,text_above='')
    plt.savefig('savedPlots_checked/nf_score_correct_incor_actual_reward_run.pdf')
    plt.close(fig)




