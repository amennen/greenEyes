
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
currPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(os.path.dirname(currPath))
sys.path.append(rootPath)
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

defaultConfig = os.path.join(currPath, 'conf/greenEyes_organized.toml')
# date doesn't have to be right, but just make sure subject number, session number, computers are correct

def getCorrectProbability(cfg):
    nRuns = len(cfg.Runs)
    all_correct_prob = np.zeros((nRuns,cfg.nStations))
    for r in np.arange(nRuns):
        fileStr = '{0}/patternsData_r{1}*'.format(cfg.subject_full_day_path,r+1)
        run_pat = glob.glob(fileStr)[-1]
        run_data = loadMatFile(run_pat)
        all_correct_prob[r,:] = run_data.correct_prob[0,:]
    return all_correct_prob

def getReward(cfg,all_correct_prob):
    nRuns = len(cfg.Runs)
    run_avg = np.zeros((nRuns,))
    for r in np.arange(nRuns):
        this_run = all_correct_prob[r,:]
        this_run[this_run < 0.5] = 0
        run_avg[r] = np.nanmean(this_run)
    total_money_reward = np.nansum(run_avg)*5
    rewardStr = 'TOTAL REWARD: %5.2f\n' % total_money_reward
    print(rewardStr)
    return total_money_reward

def plotCorrectProbability(cfg,all_correct_prob):
    # now plot everything
    nRuns = len(cfg.Runs)
    cmap = plt.get_cmap('Blues')
    color_idx = np.linspace(0, 1, nRuns)
    plt.figure()
    for r in np.arange(nRuns):
        label = 'Run %i' % r
        plt.plot(np.arange(cfg.nStations),all_correct_prob[r,:],color=plt.cm.cool(color_idx[r]), label = label )
    plt.plot([0,cfg.nStations], [0.5,0.5], '--', color='red', label='chance')
    plt.legend()
    plt.xlabel('Station')
    plt.ylabel('Correct prob')
    plt.ylim([0 ,1 ])
    plt.xlim([0,cfg.nStations-1])
    plt.show()


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                       help='experiment config file (.json or .toml)')
    argParser.add_argument('--runs', '-r', default='1', type=str,
                       help='Comma separated list of run numbers')
    argParser.add_argument('--scans', '-s', default='5', type=str,
                       help='Comma separated list of scan number')
    # creates web pipe communication link to send/request responses through web pipe
    argParser.add_argument('--webpipe', '-w', default=None, type=str,
                       help='Named pipe to communicate with webServer')
    argParser.add_argument('--filesremote', '-x', default=False, action='store_true',
                       help='dicom files retrieved from remote server')
    argParser.add_argument('--getReward', '-g', default=False, action='store_true',
                       help='if you want to calculate reward earned')
    argParser.add_argument('--makePlots', '-p', default=False, action='store_true',
                       help='if you want to plot correct probability over time')
    args = argParser.parse_args()
    print(args)
    cfg = greenEyes.initializeGreenEyes(args.config,args)
    correct_prob = getCorrectProbability(cfg)
    print('Prob for each run (col) over all stations (row)')
    print(correct_prob.T)
    if args.getReward:
        total_reward = getReward(cfg,correct_prob)
    if args.makePlots:
        plotCorrectProbability(cfg,correct_prob)


if __name__ == "__main__":
    # execute only if run as a script
    main()

