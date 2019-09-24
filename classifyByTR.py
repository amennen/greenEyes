# purpose: train given subject


import numpy as np
import pickle
import nibabel as nib
import nilearn
import glob
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.plotting import show
from nilearn.plotting import plot_roi
from nilearn import image
from nilearn.masking import apply_mask
# get_ipython().magic('matplotlib inline')
import scipy
import matplotlib
import matplotlib.pyplot as plt
from nilearn import image
from nilearn.input_data import NiftiMasker
#from nilearn import plotting
import nibabel
from nilearn.masking import apply_mask
from nilearn.image import load_img
from nilearn.image import new_img_like
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, svm, metrics
from sklearn.linear_model import Ridge
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.feature_selection import SelectFwe
from scipy import signal
from scipy.fftpack import fft, fftshift
from scipy import interp
import csv
params = {'legend.fontsize': 'large',
          'figure.figsize': (5, 3),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
font = {'weight': 'bold',
        'size': 22}
plt.rc('font', **font)

import os
import pandas as pd
import csv
from scipy import stats
import brainiak
import brainiak.funcalign.srm
import sys
from sklearn.utils import shuffle
import random
from datetime import datetime
random.seed(datetime.now())
import shutil
#currPath = os.path.dirname(os.path.realpath(__file__))
#rootPath = os.path.dirname(os.path.dirname(currPath))
#sys.path.append(rootPath)

sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
from rtCommon.utils import loadConfigFile, dateStr30, DebugLevels, writeFile, loadMatFile

offline_path = '/jukebox/norman/amennen/prettymouth_fmriprep2/code/saved_classifiers'
fmriprep_path = '/jukebox/norman/amennen/RT_prettymouth/data/bids/Norman/Mennen/5516_greenEyes/derivatives/fmriprep'
TOM_large = '/jukebox/norman/amennen/prettymouth_fmriprep2/ROI/TOM_large_resampled_maskedbybrain.nii.gz'
TOM_cluster = '/jukebox/norman/amennen/prettymouth_fmriprep2/ROI/TOM_cluster_resampled_maskedbybrain.nii.gz'
DMNmask='/jukebox/norman/amennen/MNI_things/Yeo_JNeurophysiol11_MNI152/Yeo_Network7mask_reoriented_resampledBOLD2.nii.gz'

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

def getPatternsData(subjectNum,runNum):
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    ses_id = 'ses-{0:02d}'.format(2)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/patternsData_r{2}_*.mat'.format(bids_id,ses_id,runNum)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data

def findTRNumber(pulseArray,time):
    closestTR = np.argmin(np.abs(pulseArray-time))
    # check that time happened after pulse array
    if pulseArray[closestTR] > time:
        closestTR = closestTR - 1
    return closestTR

maskType = 1
removeAvg = 1
filterType = 0
k1 = 0
k2 = 25
classifierType = 1


allSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14]
nSub = len(allSubjects)
interpretations = {}
for s in np.arange(nSub):
    interpretations[s] = getSubjectInterpretation(allSubjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
story_TR_1 = 14
story_TR_2 = 464
run_TRs = 450
nRuns = 4 # for subject 101 nRuns = 3, for subject 102 nRuns = 4
zscore_data = 1
nVoxels = 2414
allData = np.zeros((nVoxels,run_TRs,nRuns,nSub))
originalData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub))
zscoredData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub))
removedAvgData_all = np.zeros((nVoxels,run_TRs,nRuns,nSub))


# get all subject data first
for s in np.arange(nSub):
    subjectNum = allSubjects[s]
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    ses_id = 'ses-{0:02d}'.format(2)
    data_dir = fmriprep_path + '/' + bids_id + '/' + ses_id + '/' + 'func'

    
    for r in np.arange(nRuns):
        subjectRun = r + 1
        run_id = 'run-{0:02d}'.format(subjectRun)
        subjData = data_dir + '/' + bids_id + '_' + ses_id + '_' + 'task-story' + '_' + run_id + '_' + 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
        print(subjData)
        if filterType == 0:
            if maskType == 0:
                masked_data = apply_mask(subjData,DMNmask)
            elif maskType == 1:
                masked_data = apply_mask(subjData,TOM_large)
            elif maskType == 2:
                masked_data = apply_mask(subjData,TOM_cluster)
            masked_data_remove10 = masked_data[10:,:]
            originalData_all[:,:,r,s] = masked_data_remove10[story_TR_1:story_TR_2,:].T # now should be voxels x TR
        # B. zscore within that subject across all time points
        if zscore_data:
            originalData_all[:,:,r,s] = stats.zscore(originalData_all[:,:,r,s],axis=1,ddof = 1)
            originalData_all[:,:,r,s] = np.nan_to_num(originalData_all[:,:,r,s])
            zscoredData_all[:,:,r,s] = originalData_all[:,:,r,s]
        # C. remove average signal
        if removeAvg:
            average_signal_fn = offline_path + '/' +  'averageSignal' + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2) + '.npy'
            average_signal = np.load(average_signal_fn)
            #average_signal = 0
            SRM_removed = originalData_all[:,:,r,s] - average_signal
            removedAvgData_all[:,:,r,s] = SRM_removed
        else:
            removedAvgData_all[:,:,r,s] = originalData_all[:,:,r,s]
        allData[:,:,r,s] = removedAvgData_all[:,:,r,s]


# now test for each subject
prob_cheating = np.zeros((run_TRs,nRuns,nSub))
for s in np.arange(nSub):
    for r in np.arange(nRuns):
        this_testing_data = allData[:,:,r,s]
        for t in np.arange(run_TRs):
            testing_data_reshaped = this_testing_data[:,t].reshape(1,-1)
            filename_clf = offline_path + '/' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg) + '_classifierType_' + str(classifierType) + '_filter_' + str(filterType)  + '_k1_' + str(k1) + '_k2_' + str(k2)  + '_TR_' + str(t) + '.sav'
            loaded_model = pickle.load(open(filename_clf, 'rb'))
            prob_cheating[t,r,s] = loaded_model.predict_proba(testing_data_reshaped)[0][1]

# Timing to get:
# 1. station times (shifted)
nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation() # these are the ACTUAL station shifted, (recorded symbol shown 3 TRs before)

# 2. when received feedback (shifted) - 3 TRs after received feedback compared to station activity?
average_station_ev = np.zeros((nStations,nRuns,nSub))
after_station_ev = np.zeros((nStations,nRuns,nSub))

for s in np.arange(nSub):
    subjectNum = allSubjects[s]
    for r in np.arange(nRuns):
        data = getBehavData(subjectNum,r+1) #subject,runnum where run num is 1-4
        firstTrigger = data.timing.trig.wait
        # convert these to TR numbers
        pulseTiming = data.timing.trig.pulses[0,:] # indexed by volume
        flipTimes = data.timing.actualOnsets.story[:,0] # all flip times as indexed by story index

        for st in np.arange(nStations):
            startFb = data.timing.startFeedbackDisplay[0,st]
            stopFb = data.timing.stopFeedbackDisplay[0,st]
            tr_shift = 3
            startTR = findTRNumber(flipTimes,startFb) + tr_shift # HRF lag
            stopTR = findTRNumber(flipTimes,stopFb) + tr_shift # HRF lag
            these_station_TRs = stationDict[st] 
            average_station_ev[st,r,s] = np.mean(prob_cheating[these_station_TRs,r,s])
            if stopTR < last_tr_in_station[st]:
                print('ERR IN TIMING')
                after_station_ev[st,r,s] = np.nan
            else:
                after_station_ev[st,r,s] = np.mean(prob_cheating[startTR:stopTR+1,r,s])
# Comparison: what was TR x TR classifier evidence (1) during station and (2) immediately after feedback?
# maybe average across runs and stations?
# make two plots: one for C and one for paranoid
colors = ['r', 'g']
labels = ['cheating', 'paranoid']
plt.figure()
for r in np.arange(nRuns):

    for s in np.arange(nSub):
        subjectNum = allSubjects[s]
        interpretation = getSubjectInterpretation(subjectNum)
        if interpretation == 'C':
            # separate by run
            plt.subplot(2,4,r+1)
            #for st in np.arange(st):
            plt.plot([0, 1], [average_station_ev[:,r,s],after_station_ev[:,r,s]], '-',ms=10,alpha=0.2,color=colors[0])
                #plt.plot([0,1],[average_station_ev[st,r,s],after_station_ev[st,r,s]], '.',ms=10,alpha=0.2,color=colors[0])
            plt.ylim([0,1])
        else:
            plt.subplot(2,4,r+5)
            plt.plot([0, 1], [average_station_ev[:,r,s],after_station_ev[:,r,s]], '-',ms=10,alpha=0.2,color=colors[1])
            #plt.plot([0,1],[average_station_ev[st,r,s],after_station_ev[st,r,s]], '.',ms=10,alpha=0.2,color=colors[1])
            plt.ylim([0,1])
plt.show()

# now do average by subject across runs and stations
run_avg_station = np.nanmean(average_station_ev,axis=1)
avg_station = np.nanmean(run_avg_station,axis=0) # average over stations
run_avg_after = np.nanmean(after_station_ev,axis=1)
avg_after = np.nanmean(run_avg_after,axis=0) # average over stations
plt.figure(figsize=(20,10))
for s in np.arange(nSub):
    subjectNum = allSubjects[s]
    interpretation = getSubjectInterpretation(subjectNum)
    if interpretation == 'C':
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '-',lw=3,alpha=0.2,color=colors[0])
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '.',ms=20,alpha=0.2,color=colors[0])
    else:
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '-',lw=3,alpha=0.2,color=colors[1])
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '.',ms=20,alpha=0.2,color=colors[1])
plt.ylabel('p(cheating)')
plt.ylim([0,1])
labels2 = ['Avg station p(cheating)', 'Avg p(cheating) during fb']
plt.xticks(np.arange(2),labels2,fontsize=25)
C_means = np.array([np.mean(avg_station[C_ind]),np.mean(avg_after[C_ind])])
P_means = np.array([np.mean(avg_station[P_ind]),np.mean(avg_after[P_ind])])
C_sems = np.array([scipy.stats.sem(avg_station[C_ind]),scipy.stats.sem(avg_after[C_ind])])
P_sems = np.array([scipy.stats.sem(avg_station[P_ind]),scipy.stats.sem(avg_after[P_ind])])

plt.errorbar(np.arange(2), C_means, C_sems, color=colors[0], lw = 4)
plt.errorbar(np.arange(2), P_means, P_sems, color=colors[1], lw = 4)
plt.show()


# maybe try separately for run 1 and run 4 to see if they learn anything
# now do average by subject across runs and stations

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
STATION = 0
run_avg_station = average_station_ev[STATION]
avg_station = np.nanmean(run_avg_station,axis=0) # average over stations
run_avg_after = after_station_ev[STATION]
avg_after = np.nanmean(run_avg_after,axis=0) # average over stations
for s in np.arange(nSub):
    subjectNum = allSubjects[s]
    interpretation = getSubjectInterpretation(subjectNum)
    if interpretation == 'C':
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '-',lw=3,alpha=0.2,color=colors[0])
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '.',ms=20,alpha=0.2,color=colors[0])
    else:
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '-',lw=3,alpha=0.2,color=colors[1])
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '.',ms=20,alpha=0.2,color=colors[1])
plt.ylabel('p(cheating)')
plt.ylim([0,1])
labels2 = ['Avg station p(cheating)', 'Avg p(cheating) during fb']
plt.xticks(np.arange(2),labels2,fontsize=25)
C_means = np.array([np.mean(avg_station[C_ind]),np.mean(avg_after[C_ind])])
P_means = np.array([np.mean(avg_station[P_ind]),np.mean(avg_after[P_ind])])
C_sems = np.array([scipy.stats.sem(avg_station[C_ind]),scipy.stats.sem(avg_after[C_ind])])
P_sems = np.array([scipy.stats.sem(avg_station[P_ind]),scipy.stats.sem(avg_after[P_ind])])

plt.errorbar(np.arange(2), C_means, C_sems, color=colors[0], lw = 4)
plt.errorbar(np.arange(2), P_means, P_sems, color=colors[1], lw = 4)
plt.subplot(1,2,2)
STATION = 3
run_avg_station = average_station_ev[STATION]
avg_station = np.nanmean(run_avg_station,axis=0) # average over stations
run_avg_after = after_station_ev[STATION]
avg_after = np.nanmean(run_avg_after,axis=0) # average over stations
for s in np.arange(nSub):
    subjectNum = allSubjects[s]
    interpretation = getSubjectInterpretation(subjectNum)
    if interpretation == 'C':
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '-',lw=3,alpha=0.2,color=colors[0])
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '.',ms=20,alpha=0.2,color=colors[0])
    else:
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '-',lw=3,alpha=0.2,color=colors[1])
        plt.plot([0, 1], [avg_station[s],avg_after[s]], '.',ms=20,alpha=0.2,color=colors[1])
plt.ylabel('p(cheating)')
plt.ylim([0,1])
labels2 = ['Avg station p(cheating)', 'Avg p(cheating) during fb']
plt.xticks(np.arange(2),labels2,fontsize=25)
C_means = np.array([np.mean(avg_station[C_ind]),np.mean(avg_after[C_ind])])
P_means = np.array([np.mean(avg_station[P_ind]),np.mean(avg_after[P_ind])])
C_sems = np.array([scipy.stats.sem(avg_station[C_ind]),scipy.stats.sem(avg_after[C_ind])])
P_sems = np.array([scipy.stats.sem(avg_station[P_ind]),scipy.stats.sem(avg_after[P_ind])])
plt.title('STATION %i' % STATION)
plt.errorbar(np.arange(2), C_means, C_sems, color=colors[0], lw = 4)
plt.errorbar(np.arange(2), P_means, P_sems, color=colors[1], lw = 4)
plt.show()

subject_learning = avg_after - avg_station


# are the peopel who go in the right direction score better?
all_context_scores = np.zeros((nSub,))
all_story_scores = np.zeros((nSub,))
nR = 9
all_rating_scores = np.zeros((nSub,nR))
for s in np.arange(nSub):  
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
for s in np.arange(nSub):
    subjectNum = allSubjects[s]
    interpretation = getSubjectInterpretation(subjectNum)
    if interpretation == 'C':
        index = 0 
    elif interpretation == 'P':
        index = 1
    plt.plot(subject_learning[s],all_context_scores[s],'.',color=colors[index],ms=30,alpha=0.3)
#plt.title('Diff prob cheating R4 - R1')
#plt.xticks(np.array([0,1]), labels) 
plt.xlabel('Average difference feedback - station')
plt.ylabel('Interpretation score')
plt.show()

# # now load interpretation and plot
# # (1) plot story scores
# plt.figure()
# colors = ['r', 'g']
# labels = ['cheating', 'paranoid']
# for s in np.arange(nSub):
#     subjectNum = allSubjects[s]
#     interpretation = getSubjectInterpretation(subjectNum)
#     if interpretation == 'C':
#         index = 0 
#     elif interpretation == 'P':
#         index = 1
#     plt.plot(np.arange(nRuns),prob_cheating[:,s] , '-', color=colors[index], ms=20,alpha=0.2)
#     plt.plot(np.arange(nRuns),prob_cheating[:,s] , '.', color=colors[index], ms=20,alpha=0.2)
# #plt.xlim([-0.5,1.5])
# #plt.ylim([0, 1])
# plt.title('Cheating probability by run')
# plt.ylabel('Cheating probability')
# plt.xlabel('Run #')
# plt.show()
# #plt.xticks(np.array([0,1]), labels) 

# # calculate diff from first to last run and plot
# run_diff = prob_cheating[3,:] - prob_cheating[0,:]
# plt.figure()
# for s in np.arange(nSub):
#     subjectNum = allSubjects[s]
#     interpretation = getSubjectInterpretation(subjectNum)
#     if interpretation == 'C':
#         index = 0 
#     elif interpretation == 'P':
#         index = 1
#     plt.plot(index,run_diff[s] , '.', color=colors[index], ms=20,alpha=0.3)
# plt.title('Diff prob cheating R4 - R1')
# plt.xticks(np.array([0,1]), labels) 
# plt.ylabel('Diff in cheating prob')
# plt.show()
