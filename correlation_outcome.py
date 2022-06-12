# analysis: correlate
# (1) final interpretation score
# (2) mean probe response

# purpose collapse across groups, only top/bottom clf performance

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import sys
import matplotlib.cm as cmx
import seaborn as sns
from numpy.polynomial.polynomial import polyfit
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
from analyzeExp2_deflections2 import getStationInformation, calculate_prob_stats, calculate_clf_sep, getStationInformation
from display.plotBehavior_Exp2 import get_behav_scores, get_empathy_diff

defaultConfig = os.path.join(os.getcwd(), 'conf/greenEyes_cluster.toml')
cfg = loadConfigFile(defaultConfig)
params = StructDict({'config':defaultConfig, 'runs': '1', 'scans': '9', 'webpipe': 'None', 'webfilesremote': False})
cfg = greenEyes.initializeGreenEyes(defaultConfig,params)
# date doesn't have to be right, but just make sure subject number, session number, computers are correct
paranoid_c = '#99d8c9'
cheating_c = '#fc9272'

nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation()

nRuns = 4
subjects = np.array([25,26,28,29,30,31,32,33,35,36,37,38,39,41,40,42,43,44,45,46])
nSubs = len(subjects)
# get time course by run by subject
interpretations = {}
for s in np.arange(nSubs):
    interpretations[s] = getSubjectInterpretation(subjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']


all_cheating_prob, all_choices, subj_means, all_nf_score = calculate_prob_stats(subjects)
all_story_scores, all_context_scores, all_correct_context, all_rating_scores = get_behav_scores(subjects)
classifier_separation, top_subj, bottom_subj = calculate_clf_sep(subj_means, P_ind, C_ind)


# now get "correct" and "incorrect" probe responses based on group
all_choices_correct = np.zeros_like(all_choices)*np.nan # change into being correct and incorrect instead
for s in np.arange(nSubs):
    this_sub_interpretation = interpretations[s]
    for r in np.arange(nRuns):
        for st in np.arange(nStations):
            correct = np.nan
            if all_choices[s,st,r] == 1: # if they chose the cheating response on that run
                if this_sub_interpretation == 'C':
                    correct = 1
                elif this_sub_interpretation == 'P':
                    correct = 0
            elif all_choices[s,st,r] == 0: # if they chose the paranoid response
                if this_sub_interpretation == 'C':
                    correct = 0
                elif this_sub_interpretation == 'P':
                    correct = 1
            all_choices_correct[s,st,r] = correct

# get average statistics by run - average over stations
all_choices_correct_run = np.nanmean(all_choices_correct,axis=1)

# get average across all runs per subject
all_choices_correct_exp = np.nanmean(all_choices_correct_run, axis=1)

# 1. Take correlation: total interpretation and total choices_stations
print('FIRST AVERAGE OVER ALL RUNS')
x=all_choices_correct_exp
y=all_correct_context
r,p=scipy.stats.pearsonr(x,y)
printStatsResults('interpretation and classifier linear relationship',r, p)
text_f = 'r = %2.4f\np = %2.4f' % (r,p)

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-',alpha=0.8,lw=lw, color='k')
for s in np.arange(nSubs):
    if s in top_subj:
        alpha=1
    elif s in bottom_subj:
        alpha=0.5
    plt.plot(x[s],y[s],'.',ms=20,color='k',alpha=alpha)
plt.xlabel('correct probe response',fontsize=25)
plt.ylabel('correct interpretation', fontsize=25)
plt.title(f'averaged over runs',fontsize=30)
plt.yticks([-1,0,1])
plt.xticks([0,.5,1])
ax = plt.gca()
# ax.axes.xaxis.set_ticklabels([])
# ax.axes.yaxis.set_ticklabels([])
text_f = 'r = %2.4f\np = %2.4f' % (r,p)
plt.text(0.1,0.8,text_f,fontsize=15)
plt.savefig('savedPlots_checked/probe_context.pdf')
#plt.show()


# 2. GET CORRELATION FOR EACH run
n_runs = 4
for r in np.arange(n_runs):
    print('THIS RUN')
    print(r)
    x,y = nonNan(all_choices_correct_run[:,r],
                all_correct_context,
                paired=True)
    x,subs = nonNan(all_choices_correct_run[:,r],
                np.arange(nSubs),
                paired=True)
    this_r, this_p = scipy.stats.pearsonr(x,
                                        y)
    printStatsResults('interpretation and classifier linear relationship',
                    this_r, this_p)
    fig,ax = plt.subplots(figsize=(12,9))
    sns.despine()
    b, m = polyfit(x, y, 1)
    plt.plot(x, b + m * x, '-',alpha=0.8,lw=lw, color='k')
    for i in np.arange(len(subs)):
        s = subs[i]
        if s in top_subj:
            alpha=1
        elif s in bottom_subj:
            alpha=0.5
        plt.plot(x[i],y[i],'.',ms=20,color='k',alpha=alpha)
    plt.xlabel('correct probe response',fontsize=25)
    plt.ylabel('correct interpretation', fontsize=25)
    plt.title(f'run {r}',fontsize=30)
    plt.yticks([-1,0,1])
    plt.xticks([0,.5,1])
    ax = plt.gca()
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])
    text_f = 'r = %2.4f\np = %2.4f' % (this_r,this_p)
    plt.text(0.1,0.8,text_f,fontsize=15)
    plt.savefig(f'savedPlots_checked/probe_context_run_{r}.pdf')
