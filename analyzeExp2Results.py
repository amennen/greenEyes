# Purpose: analyze data from EXP 2 (when probes are pressed)
# LEFT OFF - check if all the plots were used and comment out which ones weren't, add to notes
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
from analyzeExp2_deflections2 import getStationInformation, calculate_prob_stats, calculate_clf_sep, getStationInformation
from display.plotBehavior_Exp2 import get_behav_scores, get_empathy_diff
font = {'size': 22,
        'weight': 'normal'}
plt.rc('axes', linewidth=5)
plt.rc('xtick.major', size=10, width = 4)
plt.rc('ytick.major', size=10, width = 4)
paranoid_c = '#99d8c9'
cheating_c = '#fc9272'
# define plot vars 
lw = 8
ms = 10
alpha = 0.8
defaultConfig = os.path.join(os.getcwd(), 'conf/greenEyes_cluster.toml')
cfg = loadConfigFile(defaultConfig)
params = StructDict({'config':defaultConfig, 'runs': '1', 'scans': '9', 'webpipe': 'None', 'webfilesremote': False})
cfg = greenEyes.initializeGreenEyes(defaultConfig,params)
# date doesn't have to be right, but just make sure subject number, session number, computers are correct


# get general experiment information
nStations, stationDict, last_tr_in_station, all_station_TRs = getStationInformation()
nRuns = 4
subjects = np.array([25,26,28,29,30,31,32,33,35,36,37,38,39,41,40,42,43,44,45,46])
nSubs = len(subjects)
# get interpretations group
interpretations = {}
for s in np.arange(nSubs):
  interpretations[s] = getSubjectInterpretation(subjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']

all_cheating_prob, all_choices, subj_means, all_nf_score = calculate_prob_stats(subjects)
classifier_separation, top_subj, bottom_subj = calculate_clf_sep(subj_means, P_ind, C_ind)
all_story_scores, all_context_scores, all_correct_context, all_rating_scores = get_behav_scores(subjects)
arthur_minus_lee, arthur_minus_lee_cor = get_empathy_diff(all_rating_scores)

# reformat data into a data frame for plotting
group_str = [''] * nSubs
for s in np.arange(nSubs):
  if s in C_ind:
    group_str[s] = 'C'
  elif s in P_ind:
    group_str[s] = 'P'
vec_subj_means = subj_means.flatten()
resp = ['CHEATING','PARANOID']*nSubs
subj = np.repeat(np.arange(nSubs),2)
group_vector = np.repeat(group_str,2)
DATA = {}
DATA['resp'] = resp
DATA['subj'] = subj
DATA['p(cheating)'] = vec_subj_means
DATA['group'] = group_vector
df = pd.DataFrame.from_dict(DATA)

# (1) plot average classifier separation by subject
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='resp',y='p(cheating)',ci=68,order=['CHEATING', 'PARANOID'],color='k',alpha=0.5, errcolor='k')
g = sns.swarmplot(data=df[df['group']=='C'],x='resp',y='p(cheating)',order=['CHEATING', 'PARANOID'],color=cheating_c,alpha=0.7,size=ms)
g = sns.swarmplot(data=df[df['group']=='P'],x='resp',y='p(cheating)',order=['CHEATING', 'PARANOID'],color=paranoid_c,alpha=0.7,size=ms)
for s in np.arange(nSubs):
    plt.plot([0,1],subj_means[s,:], '--', color='k', lw=3, alpha=0.3)
plt.yticks(np.arange(0,1.25,.25))
plt.title('',fontsize=30)
plt.ylim([0,1])
plt.xlabel('',fontsize=25)
plt.ylabel('',fontsize=25)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

x,y=nonNan(subj_means[:,0],subj_means[:,1],paired=True)
r,p=scipy.stats.ttest_rel(x,y)
addComparisonStat_SYM(p/2,0,1,0.92,.05,0,text_above='C>P')
printStatsResults('avg classification based on probe responses 1-sided', r, p/2, x, y)
plt.savefig('savedPlots_checked/classification_averaged.pdf')
#plt.show()

# (2) plot the linear relationship between correct context responses and classifier accuracy
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
b, m = polyfit(classifier_separation, all_correct_context, 1)
plt.plot(classifier_separation, b + m * classifier_separation, '-',alpha=0.8,lw=lw, color='k')
for s in np.arange(nSubs):
    if interpretations[s] == 'C':
        color=cheating_c
    elif interpretations[s] == 'P':
        color=paranoid_c
    plt.plot(classifier_separation[s],all_correct_context[s],'.',ms=20,color=color,alpha=1)
plt.xlabel('',fontsize=25)
plt.ylabel('', fontsize=25)
plt.title('',fontsize=30)
plt.yticks([-1,0,1])
plt.xticks([-.25,0,.25, 0.5])
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
r,p=scipy.stats.pearsonr(classifier_separation,all_correct_context)
printStatsResults('interpretation and classifier linear relationship',r, p)
text_f = 'r = %2.2f\np = %2.2f' % (r,p)
plt.text(-0.2,0.8,text_f,fontsize=25)
plt.savefig('savedPlots_checked/classification_context.pdf')
#plt.show()



##########################################################################################
# same plot but now go to zero if score < 0.5 like for actual reward amount
##########################################################################################
all_nf_score_reward = all_nf_score.copy()
# set all values <= 0.5 to be 0
all_nf_score_reward[all_nf_score<=0.5] = 0
all_nf_score_reward[np.isnan(all_nf_score)] = np.nan

fig = plotPosterStyle_multiplePTS(all_nf_score_reward,subjects)
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
plt.savefig('savedPlots_checked/nf_score_reward.pdf')

all_nf_C = np.nanmean(np.nanstd(all_nf_score_reward[C_ind,:,:],axis=0)) # standard deviation across subjects for every station and run 
all_nf_P = np.nanmean(np.nanstd(all_nf_score_reward[P_ind,:,:],axis=0)) 
all_avg_std_reward = np.mean([all_nf_C, all_nf_P])
print('AVG WITHIN GROUP STD REWARD')
print(all_avg_std_reward)

#############################################################


