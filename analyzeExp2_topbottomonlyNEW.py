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
classifier_separation, top_subj, bottom_subj = calculate_clf_sep(subj_means, P_ind, C_ind)
all_story_scores, all_context_scores, all_correct_context, all_rating_scores = get_behav_scores(subjects)
arthur_minus_lee, arthur_minus_lee_cor = get_empathy_diff(all_rating_scores)


np.save('top_subj.npy', top_subj)
np.save('bottom_subj.npy', bottom_subj)

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


# (1) Plot probe responses by group first
print('CHOICES STATIONS')
fig = plotPosterStyle_multiplePTS(all_choices,subjects)
plt.subplot(1,4,1)
plt.ylim([0,1])
plt.xticks(np.arange(nStations), fontsize=20)

cor = nStations*nRuns
for st in np.arange(nStations):
    x,y=nonNan(all_choices[C_ind,st,0],all_choices[P_ind,st,0])
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (0, st)
        printStatsResults(text, t, p/2)
plt.yticks(np.array([0,0.5,1]), [ 'all paranoid','neutral','all cheating'],fontsize=20,rotation=0) 
plt.plot([0,nStations-1],[0.5,0.5], '--', color='k', linewidth=lw-1, alpha=0.5)
plt.ylabel('', fontsize=25)
plt.title('')
plt.xlabel('')
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.subplot(1,4,2)
plt.ylim([0,1])
plt.xticks(np.arange(nStations), fontsize=20)
plt.yticks(np.array([0,0.5,1]), [ 'all paranoid','neutral','all cheating'],fontsize=20,rotation=0) 

plt.plot([0,nStations-1],[0.5,0.5], '--', color='k', linewidth=lw-1, alpha=0.5)
for st in np.arange(nStations):
    x,y=nonNan(all_choices[C_ind,st,1],all_choices[P_ind,st,1])
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (1, st)
        printStatsResults(text, t, p/2)
plt.ylabel('', fontsize=25)
plt.title('')
plt.xlabel('')
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.subplot(1,4,3)
plt.ylim([0,1])
plt.xticks(np.arange(nStations), fontsize=20)
plt.yticks(np.array([0,0.5,1]), [ 'all paranoid','neutral','all cheating'],fontsize=20,rotation=0) 

plt.plot([0,nStations-1],[0.5,0.5], '--', color='k', linewidth=lw-1, alpha=0.5)
for st in np.arange(nStations):
    x,y=nonNan(all_choices[C_ind,st,2],all_choices[P_ind,st,2])
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (2, st)
        printStatsResults(text, t, p/2)
plt.ylabel('', fontsize=25)
plt.title('')
plt.xlabel('')
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.subplot(1,4,4)
plt.ylim([0,1])
plt.xticks(np.arange(nStations), fontsize=20)
plt.yticks(np.array([0,0.5,1]), [ 'all paranoid','neutral','all cheating'],fontsize=20,rotation=0) 
plt.plot([0,nStations-1],[0.5,0.5], '--', color='k', linewidth=lw-1, alpha=0.5)

for st in np.arange(nStations):
    x,y=nonNan(all_choices[C_ind,st,3],all_choices[P_ind,st,3])
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (3, st)
        printStatsResults(text, t, p/2)
plt.ylabel('', fontsize=25)
plt.title('')
plt.xlabel('')
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.savefig('savedPlots_checked/choices_stations.pdf')
#plt.show()

# get average statistics by run
print('all_choices by run')
print('changing stats to be consistent')
cor=nRuns
print(f'cor is {cor}')
all_choices_run = np.nanmean(all_choices,axis=1)
for i in np.arange(4):
    x,y=nonNan(all_choices_run[C_ind,i],all_choices_run[P_ind,i])
    t,p = scipy.stats.ttest_ind(x,y)
    p = p * cor
    text = f'ALL CHOICES - (1-sided) did they differ significantly for run {i}'
    printStatsResults(text,
                      t,
                      p/2,
                      x,
                      y)


##### ADD PLOT BY RUN
fig,ax = plt.subplots(figsize=(20,9))
sns.despine()
plt.errorbar(
    x=np.arange(4),
    y=np.nanmean(all_choices_run[P_ind,:],axis=0),
    yerr=scipy.stats.sem(all_choices_run[P_ind,:],
    axis=0,nan_policy='omit'
    ),
    color=paranoid_c,
    lw=lw,
    label='top',
    fmt='-o',
    ms=ms)
plt.errorbar(
    x=np.arange(4),
    y=np.nanmean(all_choices_run[C_ind,:],axis=0),
    yerr=scipy.stats.sem(all_choices_run[C_ind,:],
    axis=0,nan_policy='omit'
    ),
    color=cheating_c,
    lw=lw,
    label='top',
    fmt='-o',
    ms=ms)
plt.ylim([0,1])
plt.yticks(np.array([0,0.5,1]), [ 'all paranoid','neutral','all cheating'],fontsize=20,rotation=0) 
plt.plot([0,nRuns-1],[0.5,0.5], '--', color='k', lw=lw-1, alpha=0.5)
plt.ylabel('', fontsize=25)
plt.xlabel('',fontsize=25)
plt.xlabel('',fontsize=25)
plt.xticks(np.arange(4),fontsize=20)
plt.ylim([0,1])
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.savefig('savedPlots_checked/all_choices_run.pdf')


# (2) plot probe responses, now divided by the top and bottom classifier performances (collapsing across interpretation groups)
print('CHOICES STATIONS CORRECT INCOR')
fig,ax = plt.subplots(figsize=(20,9))
for d in np.arange(nRuns):
  plt.subplot(1,nRuns,d+1)
  sns.despine()
  nPoints = nStations
  plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_choices_correct[top_subj,:,d],axis=0),
    yerr=scipy.stats.sem(all_choices_correct[top_subj,:,d],axis=0, nan_policy='omit'),
    color='k',
    alpha=1,
    lw=lw-1,
    label='top',
    fmt='-o',
    ms=ms)
  plt.errorbar(x=np.arange(nPoints),y=np.nanmean(all_choices_correct[bottom_subj,:,d],axis=0),
    yerr=scipy.stats.sem(all_choices_correct[bottom_subj,:,d],axis=0, nan_policy='omit'),
    color='k',alpha=alpha-0.1,lw=lw-1,label='bottom',fmt=':o',ms=ms)
  plt.xlabel('',fontsize=25)
  #plt.ylabel('area under -0.1')
  plt.xticks(np.arange(nPoints),fontsize=20)
  plt.yticks(np.arange(0,1.5,.5))
  plt.ylim([0,1])
  plt.plot([0,nStations-1],[0.5,0.5], '--', color='k', alpha=0.5, lw=lw-1)
  plt.title('')
  ax = plt.gca()
  ax.axes.xaxis.set_ticklabels([])
  ax.axes.yaxis.set_ticklabels([])

plt.subplot(1,4,1)
plt.ylabel('',fontsize=25)
# test significance across all points and do Bonferroni correction
cor = nStations*nRuns
for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[top_subj,st,0],all_choices_correct[bottom_subj,st,0])
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (0, st)
        printStatsResults(text, t, p/2,x,y)
plt.title('',fontsize=30)

plt.subplot(1,4,2)
for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[top_subj,st,1],all_choices_correct[bottom_subj,st,1])
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (1, st)
        printStatsResults(text, t, p/2,x,y)

plt.subplot(1,4,3)

for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[top_subj,st,2],all_choices_correct[bottom_subj,st,2])
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (2, st)
        printStatsResults(text, t, p/2,x,y)

plt.subplot(1,4,4)

for st in np.arange(nStations):
    x,y=nonNan(all_choices_correct[top_subj,st,3],all_choices_correct[bottom_subj,st,3])
    t,p = scipy.stats.ttest_ind(x,y)
    p =p * cor
    if np.mod(st,2):
        maxH = 1
    else:
        maxH = 1.05
    addComparisonStat_SYM(p/2,st,st,maxH,.05,0,text_above='')
    if p/2 < 0.1:
        text = '1-sided r %i station %i' % (3, st)
        printStatsResults(text, t, p/2,x,y)
plt.savefig('savedPlots_checked/choices_stations_correct_incor.pdf')
#plt.show()

# get average statistics by run
print('all_choices by run - CORRECT INCOR')
print('changing stats to be consistent')
cor=nRuns
print(f'cor is {cor}')
all_choices_correct_run = np.nanmean(all_choices_correct,axis=1)
for i in np.arange(4):
    x,y=nonNan(all_choices_correct_run[top_subj,i],all_choices_correct_run[bottom_subj,i])
    t,p = scipy.stats.ttest_ind(x,y)
    p = p * cor
    text = f'ALL CHOICES CORRECT INCOR - (1-sided) did they differ significantly for run {i}'
    printStatsResults(text,
                      t,
                      p/2,
                      x,
                      y)

print('END OF CHOICES STATIONS CORRECT INCOR')
# (3) plot comprehension differences across top and bottom performing classified subjects, collapsing across groups
maxH = 1.1
# first make data frame
data = {}
data_vector = all_story_scores
subject_vector = np.arange(nSubs)
group_str = [''] * nSubs
for s in np.arange(nSubs):
	if s in top_subj:
		group_str[s] = 'best'
	elif s in bottom_subj:
		group_str[s] = 'worst'
data['comprehension'] = data_vector
data['subject'] = subject_vector
data['group'] = group_str
df = pd.DataFrame.from_dict(data)

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='group',y='comprehension',ci=68,linewidth=lw,color='k', alpha=0.5, errcolor='k')#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='group',y='comprehension',split=True,color='k',size=ms, alpha=0.7)
maxH = 1.05
plt.ylim([0.5,1.05])
plt.ylabel('',fontsize=25)
plt.title('',fontsize=30)
plt.xlabel('',fontsize=25)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
x,y=nonNan(all_story_scores[top_subj],all_story_scores[bottom_subj])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p,0,1,maxH,.05,0,text_above=r'B\neqW')
printStatsResults('comprehension diff', t, p, x, y)
plt.yticks(np.array([0.5,0.75,1]))
plt.savefig('savedPlots_checked/comprehension_score_cor_incor.pdf')
#plt.show()

# (4) plot interpretation differences across top and bottom performing classified subjects, collapsing across groups
data = {}
data_vector = all_correct_context
subject_vector = np.arange(nSubs)
group_str = [''] * nSubs
for s in np.arange(nSubs):
	if s in top_subj:
		group_str[s] = 'best'
	elif s in bottom_subj:
		group_str[s] = 'worst'
data['comprehension'] = data_vector
data['subject'] = subject_vector
data['group'] = group_str
df = pd.DataFrame.from_dict(data)
fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='group',y='comprehension',ci=68,linewidth=lw,color='k', alpha=0.5, errcolor='k')#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='group',y='comprehension',split=True,color='k',size=ms, alpha=0.7)
maxH = 1.05
plt.ylim([-1.2,1.5])
plt.ylabel('',fontsize=25)
plt.title('', fontsize=30)
plt.yticks(np.array([-1,0,1]), fontsize=20,rotation=45) 
plt.xlabel('',fontsize=25)
x,y=nonNan(all_correct_context[top_subj],all_correct_context[bottom_subj])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='B>W')
printStatsResults('interpretation diff', t, p/2, x, y)
plt.plot([-2,2],[0,0], '--', color='k', alpha=0.5, linewidth=lw-1)
plt.yticks(np.array([-1, 0, 1]))
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.savefig('savedPlots_checked/context_score_cor_incor.pdf')
#plt.show()


# New - run t-test -- was correct context significantly above zero?
x,y=nonNan(all_correct_context[top_subj],[])
t,p = scipy.stats.ttest_1samp(x,0)
printStatsResults('is best subject correct context score > 0', t, p/2, x)


# (5) plot empathy differences for Arthur - Lee for the top and bottom groups, collapsing across interpretation groups
data = {}
data_vector = arthur_minus_lee_cor
subject_vector = np.arange(nSubs)
group_str = [''] * nSubs
for s in np.arange(nSubs):
	if s in top_subj:
		group_str[s] = 'best'
	elif s in bottom_subj:
		group_str[s] = 'worst'
data['comprehension'] = data_vector
data['subject'] = subject_vector
data['group'] = group_str
df = pd.DataFrame.from_dict(data)

fig,ax = plt.subplots(figsize=(12,9))
sns.despine()
sns.barplot(data=df,x='group',y='comprehension',ci=68,linewidth=lw,color='k', alpha=0.5, errcolor='k')#errcolor=".2", edgecolor=".2")
sns.swarmplot(data=df,x='group',y='comprehension',split=True,color='k',size=ms, alpha=0.7)
maxH = 5.1
#plt.ylim([-1.1,1.5])
plt.ylabel('',fontsize=25)
plt.title('', fontsize=30)
plt.xlabel('',fontsize=25)
plt.plot([-2,2],[0,0], '--', color='k', linewidth=lw-1, alpha=0.5)
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

x,y=nonNan(arthur_minus_lee_cor[top_subj],arthur_minus_lee_cor[bottom_subj])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0,1,maxH,.05,0,text_above='B>W')
printStatsResults('Arthur - Lee empathy diff ', t, p/2, x, y)
plt.yticks(np.array([-5, 0, 5]))
plt.savefig('savedPlots_checked/empathy_diff_cor_incor.pdf')
#plt.show()


# New - run t-test -- was correct empathy significantly above zero?
x,y=nonNan(arthur_minus_lee_cor[top_subj],[])
t,p = scipy.stats.ttest_1samp(x,0)
printStatsResults('is best subject empathy > 0', t, p/2, x)
