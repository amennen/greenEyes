# purpose: plot behavioral results for green eyes, behavior 
import os
import glob
import argparse
import numpy as np  # type: ignore
import sys
# Add current working dir so main can be run from the top level rtAttenPenn directory
sys.path.append(os.getcwd())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import nilearn.masking
# params = {'legend.fontsize': 'large',
#           'figure.figsize': (5, 3),
#           'axes.labelsize': 'x-large',
#           'axes.titlesize': 'x-large',
#           'xtick.labelsize': 'x-large',
#           'ytick.labelsize': 'x-large'}
# font = {'weight': 'normal',
#         'size': 22}
# plt.rc('font', **font)
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

projectDir = '/jukebox/norman/amennen/RT_prettymouth/data/laptopData/'
saveDir='/jukebox/norman/amennen/github/brainiak/rt-cloud/projects/greenEyes/savedPlots/'
interpretations = {}
subjects = np.array([25,26,28,29,30,31,32,33,35,36,37,38,39,41,40,42,43,44,45,46])
efficacy_FB = np.array([6,4,7,7,6,2,8,6,8,7,3,4,7,7,4,9,9,6,2,2])
nSub = len(subjects)
for s in np.arange(nSub):
    interpretations[s] = getSubjectInterpretation(subjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']

def get_behav_scores(subjects):

    all_context_scores = np.zeros((nSub,))*np.nan
    all_story_scores = np.zeros((nSub,))*np.nan
    nR = 9
    all_rating_scores = np.zeros((nSub,nR))*np.nan
    for s in np.arange(nSub):  
        subject = subjects[s]
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

    # change interpretation scores to account for the bad question
    z=all_context_scores*12
    new_context_scores = (z.copy() + 1)/11
    all_correct_context = new_context_scores.copy()
    all_correct_context[P_ind] = -1*new_context_scores[P_ind]
    all_context_scores = new_context_scores
    return all_story_scores, all_context_scores, all_correct_context, all_rating_scores

def get_empathy_diff(all_rating_scores):
    # subject empathy ratings
    arthur_minus_lee = all_rating_scores[:,0] - all_rating_scores[:,1]
    # convert arthur - lee empathy to the correct direction
    arthur_minus_lee_cor = arthur_minus_lee.copy()
    arthur_minus_lee_cor[P_ind] = -1*arthur_minus_lee[P_ind]
    return arthur_minus_lee, arthur_minus_lee_cor

if __name__ == '__main__':

    all_story_scores, all_context_scores, all_correct_context, all_rating_scores = get_behav_scores(subjects)
    scores = np.concatenate((all_story_scores[:,np.newaxis],all_context_scores[:,np.newaxis]),axis=1)
    arthur_minus_lee, arthur_minus_lee_cor = get_empathy_diff(all_rating_scores)


    # (1) plot comprehension scores by group
    maxH = 1.1
    fig,ax = plotPosterStyle_DF(scores[:,0],subjects)
    plt.xticks(np.array([-.2,.2]),fontsize=20) 
    plt.ylabel('',fontsize=25)
    plt.xlabel('',fontsize=25)
    plt.title('',fontsize=30)
    plt.ylim([0.5,1.05])
    plt.yticks(np.array([0.5,0.75,1]))
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.savefig('savedPlots_checked/comprehension_score.pdf')
    #plt.show()
    # t-test: did the groups differ in score
    x,y=nonNan(scores[P_ind,0],scores[C_ind,0])
    t,p = scipy.stats.ttest_ind(x,y)
    printStatsResults('comprehension group diff',t, p, x, y)

    # (2) plot interpretation scores by group
    fig,ax = plotPosterStyle_DF(scores[:,1],subjects)
    plt.xticks(np.array([-.2,.2]),fontsize=20) 
    plt.ylabel('',fontsize=25)
    plt.xlabel('',fontsize=25)
    plt.title('',fontsize=30)
    plt.plot([-2,2],[0,0], '--', color='k', linewidth=lw-1, alpha=0.5)
    plt.yticks(np.array([-1,0,1]),fontsize=20,rotation=45) 
    maxH=1.03
    x,y=nonNan(scores[C_ind,1],scores[P_ind,1])
    r,p = scipy.stats.ttest_ind(x,y)
    addComparisonStat_SYM(p/2,-.2,.2,maxH,.05,0,text_above='C > P')
    printStatsResults('interpretation group diff 1-tailed',r, p/2,x,y)
    plt.ylim([-1.2,1.2])
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.savefig('savedPlots_checked/context_score.pdf')


    #plt.show()
    # did either group show significant bias in the category?
    x,y=nonNan(scores[P_ind,1], [])
    r,p = scipy.stats.ttest_1samp(x,0)
    printStatsResults('did the paranoid group show bias?',r, p/2, x)
    x,y=nonNan(scores[C_ind,1], [])
    r,p = scipy.stats.ttest_1samp(x,0)
    printStatsResults('did the cheating group show bias?',r, p/2, x)

    # (3) plot empathy ratings, first for all characters
    subject_ratings_empathy = all_rating_scores[:,0:4]

    maxH=5.1
    fig,ax = plotPosterStyle_DF(subject_ratings_empathy,subjects)
    plt.title('', fontsize = 30)
    nq = 4
    labels=['Arthur','Lee','Joanie','the girl']
    plt.ylabel('', fontsize=25)
    plt.xticks(np.arange(nq),fontsize=25) 
    plt.yticks(np.arange(1,6),fontsize=20)
    plt.ylim([.1,6.5])
    x,y=nonNan(all_rating_scores[C_ind,0],all_rating_scores[P_ind,0])
    t,p = scipy.stats.ttest_ind(x,y)
    addComparisonStat_SYM(p/2,-.2,0.2,maxH,.05,0,text_above='C>P')
    printStatsResults('Arthur empathy diff',t, p/2, x, y)
    x,y=nonNan(all_rating_scores[C_ind,1],all_rating_scores[P_ind,1])
    t,p = scipy.stats.ttest_ind(x,y)
    addComparisonStat_SYM(p/2,0.8,1.2,maxH,.05,0,text_above='C<P')
    printStatsResults('Lee empathy diff',t, p/2, x, y)
    x,y=nonNan(all_rating_scores[C_ind,2],all_rating_scores[P_ind,2])
    t,p = scipy.stats.ttest_ind(x,y)
    addComparisonStat_SYM(p/2,1.8,2.2,maxH,.05,0,text_above='C<P')
    printStatsResults('Joanie empathy diff',t, p/2, x, y)
    x,y=nonNan(all_rating_scores[C_ind,3],all_rating_scores[P_ind,3])
    t,p = scipy.stats.ttest_ind(x,y)
    addComparisonStat_SYM(p/2,2.8,3.2,maxH,.05,0,text_above='')
    printStatsResults('the girl empathy diff',t, p/2, x, y)
    plt.xlabel('')
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.savefig('savedPlots_checked/all_ratings.pdf')
    #plt.show()

    # (4) plot empathy difference for Arthur - Lee
    fig,ax = plotPosterStyle_DF(arthur_minus_lee,subjects)
    plt.title('',fontsize=30)
    labels=['paranoid','cheating']
    plt.xticks(np.array([-.2,.2]), fontsize=20) 
    plt.yticks(np.array([-5,0,5]),fontsize=20)
    plt.ylim([-5,6.5])
    plt.xlabel('', fontsize=25)
    plt.ylabel('',fontsize=25)
    plt.plot([-2,2],[0,0], '--', color='k', linewidth=lw, alpha=0.5)
    x,y=nonNan(arthur_minus_lee[C_ind],arthur_minus_lee[P_ind])
    t,p = scipy.stats.ttest_ind(x,y)
    addComparisonStat_SYM(p/2,-.2,0.2,maxH,.05,0,text_above='C>P')
    printStatsResults('Arthur - Lee empathy diff',t, p/2, x, y)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.savefig('savedPlots_checked/arthur_minus_lee.pdf')
    #plt.show()


    # (5) linear relationship between empathy difference and intepretation score
    x = all_correct_context
    y = arthur_minus_lee_cor
    r,p=scipy.stats.pearsonr(x,y)
    printStatsResults('interpretation and empathy linear relationship', r, p, x, y)
    text_f = 'r = %2.2f\np = %2.2f' % (r,p)
    r,p=scipy.stats.pearsonr(all_context_scores,arthur_minus_lee)
    printStatsResults('interpretation and empathy linear relationship NOT CORRECTED', r, p, x, y)
