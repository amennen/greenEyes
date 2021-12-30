# purpose collapse across groups, only top/bottom clf performance

import os
import glob
import numpy as np
import sys

sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
# when running not in file: sys.path.append(os.getcwd())
#WHEN TESTING
#sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
from rtCommon.utils import loadConfigFile, dateStr30, DebugLevels, writeFile, loadMatFile
from rtCommon.structDict import StructDict
import greenEyes

defaultConfig = os.path.join(os.getcwd(), 'conf/greenEyes_cluster.toml')
cfg = loadConfigFile(defaultConfig)
params = StructDict({'config':defaultConfig, 'runs': '1', 'scans': '9', 'webpipe': 'None', 'webfilesremote': False})
cfg = greenEyes.initializeGreenEyes(defaultConfig,params)
# date doesn't have to be right, but just make sure subject number, session number, computers are correct

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


all_timing = np.zeros((nSubs,nStations,nRuns))*np.nan
all_trs = np.zeros((nSubs,nStations,nRuns))*np.nan
all_timing_diff = []

def getTiming(behavData):
    subject_timing = np.zeros((nStations,))*np.nan
    subject_ntrs = np.zeros((nStations,))*np.nan
    timingData  = behavData['timing']
    clf_load_start = timingData['classifierLoadStart'] # nTRs x nstations
    clf_load_time = timingData['classifierStationFound']
    tr_flip_timing = timingData['actualOnsets']['story'][:,0]
    all_timing_diff = []
    for st in np.arange(nStations):
        load_start_st_timing = clf_load_start[:,st]
        found_st_timing = clf_load_time[:,st]
        TRs_looking = load_start_st_timing.nonzero()[0] # indices of ALL TRs where you're were looking
        TR_started = TRs_looking[0]
        TR_found = found_st_timing.nonzero()[0][0] 
        # for the timing, go from the first time we started looking to the TR when we found it
        subject_timing[st] = found_st_timing[TR_found] - load_start_st_timing[TR_started]
        subject_ntrs[st] = len(TRs_looking)
        all_timing_diff.extend(list(load_start_st_timing[TRs_looking] - tr_flip_timing[TRs_looking]))
    return subject_timing, subject_ntrs, all_timing_diff

for s in np.arange(nSubs):
    subjectNum = subjects[s]
    if subjectNum==41:
        nRuns = 2
    else:
        nRuns = 4
    for runNum in np.arange(nRuns):
        b = getBehavData(subjectNum,runNum+1)
        timing, ntrs, timing_diff = getTiming(b)
        all_timing[s,:,runNum] = timing
        all_trs[s,:,runNum] = ntrs
        all_timing_diff.extend(timing_diff)

# now discount anything within .5 seconds - no way that happened that fast
thresh = 0.5 # in seconds -- this had to have been an error
corrected_timing = all_timing.copy()
corrected_trs = all_trs.copy()
corrected_timing[all_timing <= thresh] = np.nan
corrected_trs[all_timing <= thresh] = np.nan

print('looking at timing of pulsese and search for .txt file')
print('min tr pulse diff')
print(np.nanmin(np.array(all_timing_diff)))
print('max tr pulse diff')
print(np.nanmax(np.array(all_timing_diff)))
print('mean tr pulse diff')
print(np.nanmean(np.array(all_timing_diff)))
print('std tr pulse diff')
print(np.nanstd(np.array(all_timing_diff)))


print('timing description')
print('mean in s')
print(np.nanmean(corrected_timing))
print('std in s')
print(np.nanstd(corrected_timing))
print(corrected_timing[corrected_timing<1.5])

# get all non-nan TRs
n_non_nans = len(corrected_trs[~np.isnan(corrected_trs)])  
print('TR description for %i TRs' % n_non_nans)
print('mean in TRs')
print(np.nanmean(corrected_trs))
print('std in TRs')
print(np.nanstd(corrected_trs))
print(corrected_trs[corrected_trs!=2])