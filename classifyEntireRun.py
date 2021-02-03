# purpose: train given subject


import numpy as np
import pickle
import nibabel as nib
import nilearn
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
# from commonPlotting import *

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

maskType = 1
removeAvg = 1
filterType = 0
k1 = 0
k2 = 25
filename_clf = offline_path + '/' 'LOGISTIC_lbfgs_UPPERRIGHT_NOstations_' + '_' + 'ROI_' + str(maskType) + '_AVGREMOVE_' + str(removeAvg)  + '_filter_' + str(filterType) + '_k1_' + str(k1) + '_k2_' + str(k2)  + '.sav'
loaded_model = pickle.load(open(filename_clf, 'rb'))

# allSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14]
# allSubjects = np.array([25,26,28,29,30,31,32,33,35,36,37,38,39,41,40,42,43,44,45,46])
allSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19]

nSub = len(allSubjects)

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
# subject r41 only completed two runs
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
        if os.path.exists(subjData):
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
        else:
            print('DOES NOT EXIST!! SKIPPING')
            allData[:,:,r,s] = np.nan

# save the data
np.save('allSubjectsData_fmripreped_Exp1.npy', allData)

###############################################################
# # once completed just load 
# allData_loaded = np.load('allSubjectsData_fmripreped.npy')
# # this is in teh shape of [nVox x nTRs x 4 runs x 20 subjects]
# # now test for each subject
# prob_cheating = np.zeros((nRuns,nSub))
# for s in np.arange(nSub):
#     for r in np.arange(nRuns):
#         this_testing_data = allData_loaded[:,:,r,s]
#         # check if any nans
#         if np.isnan(this_testing_data).any():
#             prob_cheating[r,s] = np.nan
#         else:
#             testing_data_reshaped = np.reshape(this_testing_data,(1,nVoxels*run_TRs))
#             prob_cheating[r,s] = loaded_model.predict_proba(testing_data_reshaped)[0][1]

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

# # are the peopel who go in the right direction score better?
# all_context_scores = np.zeros((nSub,))
# all_story_scores = np.zeros((nSub,))
# nR = 9
# all_rating_scores = np.zeros((nSub,nR))
# for s in np.arange(nSub):  
#     subject = allSubjects[s]
#     context = getSubjectInterpretation(subject)
#     bids_id = 'sub-{0:03d}'.format(subject)
#     response_mat = '/jukebox/norman/amennen/RT_prettymouth/data/laptopData/' + bids_id + '/' + 'responses_scored.mat'
#     z = scipy.io.loadmat(response_mat)
#     ratings =  z['key_rating'][0]
#     all_rating_scores[s,:] = ratings
#     context_score =  z['mean_context_score'][0][0]
#     all_context_scores[s] = context_score
#     story_score = z['story_score'][0][0]
#     all_story_scores[s] = story_score

# # plot context score x and run diff on y
# plt.figure()
# for s in np.arange(nSub):
#     subjectNum = allSubjects[s]
#     interpretation = getSubjectInterpretation(subjectNum)
#     if interpretation == 'C':
#         index = 0 
#     elif interpretation == 'P':
#         index = 1
#     plt.plot(all_context_scores[s],run_diff[s],'.',color=colors[index],ms=20,alpha=0.3)
# #plt.title('Diff prob cheating R4 - R1')
# #plt.xticks(np.array([0,1]), labels) 
# plt.ylabel('Diff in cheating prob')
# plt.xlabel('Interp score')
# plt.show()

# plt.figure()
# for s in np.arange(nSub):
#     subjectNum = allSubjects[s]
#     interpretation = getSubjectInterpretation(subjectNum)
#     if interpretation == 'C':
#         index = 0 
#     elif interpretation == 'P':
#         index = 1
#     plt.plot(np.mean(prob_cheating[:,s]),all_context_scores[s],'.',color=colors[index],ms=30,alpha=0.3)
# #plt.title('Diff prob cheating R4 - R1')
# #plt.xticks(np.array([0,1]), labels) 
# plt.yticks(fontsize=15)
# plt.xlabel('Average p(cheating) over all runs',fontsize=20)
# plt.ylabel('Interpretation score')
# plt.show()


