# purpose: test if z-scoring in a different way helps but it turned out to not matter
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

def getPatternsData(subjectNum,runNum):
    bids_id = 'sub-{0:03d}'.format(subjectNum)
    ses_id = 'ses-{0:02d}'.format(2)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/patternsData_r{2}_*.mat'.format(bids_id,ses_id,runNum)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data

def getLastBadVoxels(data):
    stationStr = 'station' + str(8)
    bad_vox = data['badVoxels']
    last_bad_vox = bad_vox[stationStr]
    return last_bad_vox

def getAvgSignal(cfg):
    averageSignal = np.load(cfg.classifierDir +  cfg.averageSignal)
    return averageSignal

def preprocessData(data):
    # remove average
    story_data = data['story_data']
    stopTR = 29
    mean_beginning = np.tile(np.nanmean(story_data[:,0:stopTR],axis=1),(451,1))
    std_beginning = np.tile(np.nanstd(story_data[:,0:stopTR],axis=1,ddof=1),(451,1))
    zscoredData = (story_data - mean_beginning)/std_beginning
    #zscoredData = stats.zscore(story_data,axis=1,ddof = 1)
    zscoredData = np.nan_to_num(zscoredData)
    # remove story TRs
    # remove story average
    bad_vox = getLastBadVoxels(data)
    # now combine with previously made badvoxels
    signalAvg = getAvgSignal(cfg) 
    preprocessedData = zscoredData[:,0:450] - signalAvg
    #preprocessed[bad_vox,:] = 0
    return preprocessedData

def reclassifyStation(preprocessedData,stationInd,cfg,clf):
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
    thisStationData = preprocessedData[:,this_station_TRs]

    dataForClassification_reshaped = np.reshape(thisStationData,(1,cfg.nVox*n_station_TRs))
    cheating_probability = loaded_model.predict_proba(dataForClassification_reshaped)[0][1]
    pred = loaded_model.predict(dataForClassification_reshaped)[0]
    if np.argmax(loaded_model.predict_proba(dataForClassification_reshaped)) == pred:
        agree = 1
    else:
        agree = 0
    return cheating_probability, pred, agree


labels = ['cheating', 'paranoid']
allSubjects = [2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22]
#allSubjects = [21]
nSubs = len(allSubjects)
nRuns = 4
nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation()
c_prob = np.zeros((nRuns,nSubs))
for s in np.arange(nSubs):
    subjectNum = allSubjects[s]
    for r in np.arange(nRuns):
        data = getPatternsData(subjectNum,r+1)
        preprocessed = preprocessData(data)
        c_prob[r,s],p,ag = reclassifyStation(preprocessed,7,cfg,1)

a = np.load('all_station_log_info.npz')
c_max = a['max']
p_max = 1 - a['min']