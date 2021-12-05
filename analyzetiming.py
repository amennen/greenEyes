# purpose collapse across groups, only top/bottom clf performance

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
# params = {'legend.fontsize': 'large',
#           'figure.figsize': (5, 3),
#           'axes.labelsize': 'x-large',
#           'axes.titlesize': 'x-large',
#           'xtick.labelsize': 'x-large',
#           'ytick.labelsize': 'x-large'}
# font = {'weight': 'normal',
#         'size': 22}
# plt.rc('font', **font)
defaultConfig = os.path.join(os.getcwd(), 'conf/greenEyes_cluster.toml')
cfg = loadConfigFile(defaultConfig)
params = StructDict({'config':defaultConfig, 'runs': '1', 'scans': '9', 'webpipe': 'None', 'webfilesremote': False})
cfg = greenEyes.initializeGreenEyes(defaultConfig,params)
# date doesn't have to be right, but just make sure subject number, session number, computers are correct
paranoid_c = '#99d8c9'
cheating_c = '#fc9272'
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

def getBehavData(subject_num,run_num):
    bids_id = 'sub-{0:03d}'.format(subject_num)
    ses_id = 'ses-{0:02d}'.format(2)
    run_id = 'run-{0:03d}'.format(run_num)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/{0}/{1}/{2}/behavior_run{3}_*.mat'.format(bids_id,ses_id,run_id,run_num)
    fn = glob.glob(filename)[-1]
    data = loadMatFile(fn)
    return data    

def createStationVector(stationDict):
    n_stations = len(stationDict)
    allTRs = np.arange(25,475+1)
    nTRs_story = len(allTRs)
    recorded_TRs = np.zeros((nTRs_story,))
    for st in np.arange(n_stations):
        this_station_TRs = np.array(stationDict[st]) # remove + 1 # going from python --> matlab indexing
        recorded_TRs[this_station_TRs - 3] = st +1
    return recorded_TRs

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

all_timing = np.zeros((nSubs,nStations,nRuns))*np.nan
all_trs = np.zeros((nSubs,nStations,nRuns))*np.nan

def getTiming(behavData):
    subject_timing = np.zeros((nStations,))*np.nan
    subject_ntrs = np.zeros((nStations,))*np.nan
    timingData  = behavData['timing']
    clf_load_start = timingData['classifierLoadStart'] # nTRs x nstations
    clf_load_time = timingData['classifierStationFound']
    for st in np.arange(nStations):
        this_station_timing = clf_load_start[:,st]
        this_station_found = clf_load_time[:,st]
        TRs_looking = this_station_timing.nonzero()[0] # indices of TRs where you're looking
        subject_timing[st] = this_station_found[TRs_looking[-1]] - this_station_timing[TRs_looking[0]]
        subject_ntrs[st] = len(TRs_looking)
    return subject_timing, subject_ntrs

for s in np.arange(nSubs):
    subjectNum = subjects[s]
    if subjectNum==41:
        nRuns = 2
    else:
        nRuns = 4
    for runNum in np.arange(nRuns):
        b = getBehavData(subjectNum,runNum+1)
        timing, ntrs = getTiming(b)
        all_timing[s,:,runNum] = timing
        all_trs[s,:,runNum] = ntrs