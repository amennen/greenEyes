# Purpose to plot everything like w/ experiment 1
# but first we have to retrain data w/ logistic classifier


import os
import glob
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
font = {'size': 22,
        'weight': 'normal'}
plt.rc('axes', linewidth=5)
plt.rc('xtick.major', size=10, width = 4)
plt.rc('ytick.major', size=10, width = 4)

# define plot vars 
lw = 8
ms = 10
alpha = 0.8

import pandas as pd
import json 
import datetime
from dateutil import parser
from subprocess import call
import time
import nilearn
from scipy import stats
import nibabel as nib
import argparse
import sys
import logging
import matplotlib.cm as cmx
import scipy
from numpy.polynomial.polynomial import polyfit

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
from commonPlotting import *

# params = {'legend.fontsize': 'large',
#           'figure.figsize': (5, 3),
#           'axes.labelsize': 'x-large',
#           'axes.titlesize': 'x-large',
#           'xtick.labelsize': 'x-large',
#           'ytick.labelsize': 'x-large'}
# font = {'weight': 'bold',
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
    #print(filename)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data

def getClassificationData(data,station_ind):
    stationStr = 'station' + str(station_ind)
    data_for_classification = data['dataForClassification']
    station_data = data_for_classification[stationStr]
    return station_data

def reclassifyAll(subject_num,cfg,clf,nRuns=4,n_stations=7):
    all_cheating_prob = np.zeros((n_stations,nRuns))
    all_pred = np.zeros((n_stations,nRuns))
    all_agree = np.zeros((n_stations,nRuns))
    for r in np.arange(nRuns):
        for st in np.arange(n_stations):
            all_cheating_prob[st,r],all_pred[st,r],all_agree[st,r] = reclassifyStation(subject_num,r,st,cfg,clf)
    return all_cheating_prob,all_pred,all_agree

def loadClassifier(clf,station_ind):
    if clf == 1:
        clf_str = cfg.cluster.classifierDir + cfg.classifierNamePattern
    full_str = clf_str.format(stationInd)
    loaded_model = pickle.load(open(full_str, 'rb'))
    return loaded_model

def reclassifyStation(subject_num,run_num,station_ind,cfg,clf):
    data = getPatternsData(subject_num,run_num+1)
    station_data = getClassificationData(data,station_ind)
    if clf == 1:
        clf_str = cfg.cluster.classifierDir + cfg.classifierNamePattern # this is the logistic version because the config is updated
    elif clf == 2:
        clf_str = cfg.cluster.classifierDir + "UPPERRIGHT_stationInd_{}_ROI_1_AVGREMOVE_1_filter_0_k1_0_k2_25.sav" # first SVM version

    full_str = clf_str.format(station_ind)
    loaded_model = pickle.load(open(full_str, 'rb'))
    this_station_TRs = np.array(cfg.stationsDict[station_ind])
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

def getCorrectProbability(subjectNum,nRuns=4,n_stations=9):
    all_correct_prob = np.zeros((n_stations,nRuns))
    for r in np.arange(nRuns):
        run_data = getPatternsData(subjectNum,r+1) # for matlab indexing
        all_correct_prob[:,r] = run_data.correct_prob[0,:]
    return all_correct_prob


def getStationInformation(config='conf/greenEyes_cluster.toml'):
    allinfo = {}
    cfg = loadConfigFile(config)
    # make it so it automatically uses the 9 station version
    #station_FN = cfg.cluster.classifierDir + '/' + cfg.stationDict
    station_FN = cfg.cluster.classifierDir + '/' + 'upper_right_winners_nofilter.npy'
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

def loadClassifier(cfg,station):
    thisClassifierFileName = cfg.classifierDir + cfg.classifierNamePattern.format(station)
    loaded_model = pickle.load(open(thisClassifierFileName, 'rb'))
    return loaded_model

def getNumberofTRs(stationsDict,st):
    this_station_TRs = np.array(stationsDict[st])
    n_station_TRs = len(this_station_TRs)
    return n_station_TRs

def getTransferredZ(cheating_prob,station,all_mean,all_std):
    z_val = (cheating_prob - all_mean[station])/all_std[station]
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
nStations, cfg.stationsDict, last_tr_in_station, all_station_TRs = getStationInformation() # returns 9 stations

Exp1Subjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19]
nSubs = len(Exp1Subjects)
all_cheating_prob = np.zeros((nSubs,nStations,nRuns))*np.nan
all_nf_scores = np.zeros((nSubs,nStations,nRuns))*np.nan
orig_cheating_prob = np.zeros((nSubs,nStations,nRuns))*np.nan 

interpretations = {}
for s in np.arange(nSubs):
    interpretations[s] = getSubjectInterpretation(Exp1Subjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']

data_dict = {}
data_dict['subject'] = Exp1Subjects
list_interp = [''] * nSubs
for s in np.arange(nSubs):
    list_interp[s] = interpretations[s]
data_dict['interp'] = list_interp
df = pd.DataFrame.from_dict(data_dict)

nRuns=4
for s in np.arange(nSubs): 
    subjectNum = Exp1Subjects[s]
    all_cheating_prob[s,:,:],_,_ = reclassifyAll(subjectNum,cfg,1,nRuns,nStations)
    all_nf_scores[s,:,:] = getCorrectProbability(subjectNum)
    orig_cheating_prob[s,:,:],_,_ = reclassifyAll(subjectNum,cfg,2,nRuns,nStations)


# now get behavior
projectDir = '/jukebox/norman/amennen/RT_prettymouth/data/laptopData/'

all_context_scores = np.zeros((nSubs,))
all_story_scores = np.zeros((nSubs,))
nR = 9
all_rating_scores = np.zeros((nSubs,nR))
for s in np.arange(nSubs):  
    subject = Exp1Subjects[s]
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

# change interpretation scores [context_score] to account for the 1 question that all subjects answered the same
z=all_context_scores*12
new_context_scores = (z.copy() + 1)/11
all_correct_context = new_context_scores.copy()
all_correct_context[P_ind] = -1*new_context_scores[P_ind]
all_context_scores = new_context_scores


# (1) Comprehension scores
maxH=1.1
scores = np.concatenate((all_story_scores[:,np.newaxis],all_context_scores[:,np.newaxis]),axis=1)
fig,ax = plotPosterStyle_DF(scores[:,0],Exp1Subjects)
plt.xticks(np.array([-.2,.2]), ['paranoid','cheating'],fontsize=20) 
plt.ylabel('accuracy',fontsize=25)
plt.xlabel('assigned group',fontsize=25)
plt.title('Comprehension scores',fontsize=30)
x,y=nonNan(scores[P_ind,0],[])
plt.ylim([0.5,1.05])
plt.yticks(np.array([0.5,0.75,1]),fontsize=20)
plt.savefig('savedPlots_checked/comprehension_score_EXP1.pdf')
#plt.show()
# do they differ in comprehension scores - t-test
x,y=nonNan(scores[P_ind,0],scores[C_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
printStatsResults('comprehension, group diff', t, p)

# (2) Interpretation scores
fig,ax = plotPosterStyle_DF(scores[:,1],Exp1Subjects)
plt.xticks(np.array([-.2,.2]), ['paranoid','cheating'],fontsize=20) 
plt.ylabel('skewness to one interpretation',fontsize=25)
plt.xlabel('assigned group',fontsize=25)
plt.title('Interpretation scores',fontsize=30)
plt.plot([-2,2],[0,0], '--', color='k')
plt.yticks(np.array([-1,0,1]), ['paranoid','neutral','cheating'],fontsize=20,rotation=45) 
maxH=1.03
r,p = scipy.stats.ttest_ind(scores[P_ind,1],scores[C_ind,1])
addComparisonStat_SYM(p/2,-.2,.2,maxH,.05,0,text_above='C > P')
plt.ylim([-1.2,1.5])
plt.savefig('savedPlots_checked/context_score_EXP1.pdf')
#plt.show()
printStatsResults('interpretation, 1-sided group diff', r, p/2)


# Next: Plot empathy ratings
subject_ratings_empathy = all_rating_scores[:,0:4]
arthur_minus_lee = all_rating_scores[:,0] - all_rating_scores[:,1]

# (4) Plot empathy for each character by group
maxH=5.1
fig,ax = plotPosterStyle_DF(subject_ratings_empathy,Exp1Subjects)
plt.title('How much do you empathize with...', fontsize = 30)
nq = 4
labels=['Arthur','Lee','Joanie','the girl']
plt.ylabel('empathy rating', fontsize=25)
plt.xticks(np.arange(nq), labels,fontsize=25) 
plt.yticks(np.arange(1,6),fontsize=20)
plt.ylim([.1,6.5])
x,y=nonNan(all_rating_scores[C_ind,0],all_rating_scores[P_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,0.2,maxH,.05,0,text_above='C>P')
printStatsResults('empathy, Arthur 1-sided group diff', t, p/2)
x,y=nonNan(all_rating_scores[C_ind,1],all_rating_scores[P_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,maxH,.05,0,text_above='C<P')
printStatsResults('empathy, Lee 1-sided group diff', t, p/2)
x,y=nonNan(all_rating_scores[C_ind,2],all_rating_scores[P_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,maxH,.05,0,text_above='C<P')
printStatsResults('empathy, Joanie 1-sided group diff', t, p/2)
x,y=nonNan(all_rating_scores[C_ind,3],all_rating_scores[P_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,maxH,.05,0,text_above='')
printStatsResults('empathy, the girl 1-sided group diff', t, p/2)
plt.xlabel('')
plt.savefig('savedPlots_checked/all_ratings_EXP1.pdf')
#plt.show()

# (5) Plot empathy difference for Arthur minus Lee empathy
fig,ax = plotPosterStyle_DF(arthur_minus_lee,Exp1Subjects)
plt.title('Arthur minus Lee',fontsize=30)
labels=['paranoid','cheating']
plt.xticks(np.array([-.2,.2]), labels,fontsize=20) 
plt.yticks(np.array([-5,0,5]),fontsize=20)
plt.ylim([-5,6.5])
plt.xlabel('assigned group', fontsize=25)
plt.ylabel('empathy difference',fontsize=25)
plt.plot([-2,2],[0,0], '--', color='k')
x,y=nonNan(arthur_minus_lee[C_ind],arthur_minus_lee[P_ind])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,0.2,maxH,.05,0,text_above='C>P')
printStatsResults('empathy, Arthur - Lee 1-sided group diff', t, p/2)
plt.savefig('savedPlots_checked/arthur_minus_lee_EXP1.pdf')
#plt.show()

# (6) Look for a relationship between empathy and interpretation score
paranoid_c = '#99d8c9'
cheating_c = '#fc9272'
arthur_minus_lee_cor = arthur_minus_lee.copy()
arthur_minus_lee_cor[P_ind] = -1*arthur_minus_lee[P_ind]
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
for s in np.arange(nSubs):
    if interpretations[s] == 'C':
        color=cheating_c
    elif interpretations[s] == 'P':
        color=paranoid_c
    plt.plot(all_correct_context[s],arthur_minus_lee_cor[s],'.',ms=20,color=color,alpha=1)
plt.xlabel('correct interpretation score',fontsize=25)
plt.ylabel('correct empathy difference',fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Empathy and interpretation relationship',fontsize=30)
x = all_correct_context
y = arthur_minus_lee_cor
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.6,lw=3, color='k')
r,p=scipy.stats.pearsonr(x,y)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-1,3,text_f,fontsize=25)
plt.ylim([-3,4])
plt.savefig('savedPlots_checked/empathy_context_EXP1.pdf')
#plt.show()
printStatsResults('Empathy-Interpretation linear relationship', r, p)

###################################################################################
# NEUROFEEDBACK CLASSIFICATION PLOTS
fig = plotPosterStyle_multiplePTS(orig_cheating_prob,Exp1Subjects)
plt.subplot(1,4,1)
plt.ylabel('p(cheating)',fontsize=25)
plt.ylim([0,1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('run 1',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.subplot(1,4,2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0,1])
plt.title('run 2',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.subplot(1,4,3)
plt.ylim([0,1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('run 3',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.subplot(1,4,4)
plt.title('run 4',fontsize=30)
plt.ylim([0,1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('station',fontsize=25)
plt.savefig('savedPlots_checked/p_cheating_EXP1.pdf')
#plt.show()

# NOW PLOT P(CHEATING) FOR LOGISTIC CLF
fig = plotPosterStyle_multiplePTS(all_cheating_prob,Exp1Subjects)
plt.subplot(1,4,1)
plt.ylabel('',fontsize=25)
plt.ylim([0,1])
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,1.25,.25))
plt.title('',fontsize=30)
plt.xlabel('',fontsize=25)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.subplot(1,4,2)
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,1.25,.25))
plt.ylim([0,1])
plt.title('',fontsize=30)
plt.xlabel('',fontsize=25)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.subplot(1,4,3)
plt.ylim([0,1])
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,1.25,.25))
plt.title('',fontsize=30)
plt.xlabel('',fontsize=25)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.subplot(1,4,4)
plt.title('',fontsize=30)
plt.ylim([0,1])
plt.xticks(fontsize=20)
plt.yticks(np.arange(0,1.25,.25))
plt.xlabel('',fontsize=25)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.savefig('savedPlots_checked/p_cheating_EXP1_logistic.pdf')
#plt.show()

# calculate variance
orig_cheating_prob_mean = np.nanmean(orig_cheating_prob,axis=2)
# calculate within-group variance 
stationVariance = np.zeros((nStations,2)) # 1 for each group
for st in np.arange(nStations):
    C_data = orig_cheating_prob_mean[C_ind,st]
    P_data = orig_cheating_prob_mean[P_ind,st]
    stationVariance[st,0] = scipy.stats.sem(C_data)
    stationVariance[st,1] = scipy.stats.sem(P_data)
printStatsResults('Within group variance C correct_prob', np.mean(stationVariance),-1)

fig = plotPosterStyle_multiplePTS(all_nf_scores,Exp1Subjects)
plt.subplot(1,4,1)
plt.ylabel('NF score ($)',fontsize=25)
plt.ylim([0,1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('run 1',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.subplot(1,4,2)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0,1])
plt.title('run 2',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.subplot(1,4,3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0,1])
plt.title('run 3',fontsize=30)
plt.xlabel('station',fontsize=25)
plt.subplot(1,4,4)
plt.title('run 4',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0,1])
plt.xlabel('station',fontsize=25)
plt.savefig('savedPlots_checked/nf_score_EXP1.pdf')
#plt.show()

# calculate variance
nf_score_mean = np.nanmean(all_nf_scores,axis=2)
# calculate within-group variance 
stationVariance = np.zeros((nStations,2)) # 1 for each group
for st in np.arange(nStations):
    C_data = nf_score_mean[C_ind,st]
    P_data = nf_score_mean[P_ind,st]
    stationVariance[st,0] = scipy.stats.sem(C_data)
    stationVariance[st,1] = scipy.stats.sem(P_data)
printStatsResults('Within group variance NF scores', np.mean(stationVariance),-1)

# not used - Plot interpretation scores according to correctness on the y-axis instead of interpretation
# fig,ax = plotPosterStyle_DF(all_correct_context,Exp1Subjects)
# plt.xticks(np.array([-.2,.2]), ['paranoid','cheating'],fontsize=20) 
# plt.ylabel('context score')
# plt.xlabel('group')
# plt.title('context score')
# plt.yticks(np.array([-1,1]), ['incorrect','correct'],fontsize=20,rotation=45) 
# x,y=nonNan(all_correct_context[P_ind],[])
# t,p = scipy.stats.ttest_1samp(x,0)
# addComparisonStat_SYM(p/2,-.2,-0.2,maxH,.05,0,text_above='P>0')
# x,y=nonNan(all_correct_context[C_ind],[])
# t,p = scipy.stats.ttest_1samp(x,0)
# addComparisonStat_SYM(p/2,.2,0.2,maxH,.05,0,text_above='C>0')
# #r,p = scipy.stats.ttest_1samp(all_correct_context[C_ind],0)
# #addComparisonStat_SYM(p,-.2,.2,0.9,.05,0,text_above='C P')
# plt.ylim([-1.2,1.2])
# plt.savefig('savedPlots_checked/correct_context_score_EXP1.pdf')
# plt.show()
