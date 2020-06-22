# purpose: plot behavioral results for green eyes
import os
import glob
import argparse
import numpy as np  # type: ignore
import sys
# Add current working dir so main can be run from the top level rtAttenPenn directory
sys.path.append(os.getcwd())
import matplotlib
import matplotlib.pyplot as plt
import scipy
import nilearn.masking
params = {'legend.fontsize': 'large',
          'figure.figsize': (5, 3),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
font = {'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
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

projectDir = '/jukebox/norman/amennen/RT_prettymouth/data/laptopData/'

interpretations = {}
#subjects = [2 ,3 ,4 ,5,6,7,8,9,10,11,12,13,14,16,18]
subjects = [25,26,28,29,30,31]
nSub = len(subjects)
for s in np.arange(nSub):
    interpretations[s] = getSubjectInterpretation(subjects[s])
C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']

all_context_scores = np.zeros((nSub,))
all_story_scores = np.zeros((nSub,))
nR = 9
all_rating_scores = np.zeros((nSub,nR))
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

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev * 4
#all_story_scores = rand_jitter(all_story_scores)
#all_context_scores = rand_jitter(all_context_scores)

# (1) plot story scores
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
colors = ['r', 'g']
labels = ['cheating', 'paranoid']
for s in np.arange(nSub):
    if interpretations[s] == 'C':
        index = 0 
    elif interpretations[s] == 'P':
        index = 1

    plt.plot(index + np.random.randn(1)*.1,all_story_scores[s] , '.', color=colors[index], ms=40,alpha=0.2)

# ADD MEAN OF GROUP
C_mean = np.mean(all_story_scores[C_ind])
P_mean = np.mean(all_story_scores[P_ind])
C_sem = scipy.stats.sem(all_story_scores[C_ind])
P_sem = scipy.stats.sem(all_story_scores[P_ind])

plt.errorbar(np.arange(2), np.array([C_mean,P_mean]), np.array([C_sem,P_sem]))

#plt.plot(0,C_mean,'.',color='k', ms = 25)
#plt.plot(1,P_mean,'.',color='k', ms = 25)
plt.xlim([-0.5,1.5])
plt.ylim([0.5,1.1])
plt.title('Story comprehension scores')
plt.xticks(np.array([0,1]), labels) 


# (2) plot context scores
plt.subplot(1,2,2)
for s in np.arange(nSub):
    if interpretations[s] == 'C':
        index = 0
    elif interpretations[s] == 'P':
        index = 1

    plt.plot(index+np.random.randn(1)*.1,all_context_scores[s], '.', color=colors[index], ms=40,alpha=0.2)
C_mean = np.mean(all_context_scores[C_ind])
P_mean = np.mean(all_context_scores[P_ind])
C_sem = scipy.stats.sem(all_context_scores[C_ind])
P_sem = scipy.stats.sem(all_context_scores[P_ind])
plt.errorbar(np.arange(2), np.array([C_mean,P_mean]), np.array([C_sem,P_sem]))

plt.xlim([-0.5,1.5])
plt.ylim([-1,1])
plt.title('Interpretation scores')
plt.xticks(np.array([0,1]), labels) 

plt.show()

# (3) plot ratings
RATINGS = {}
RATINGS[0] = 'Empathized with Arthur';
RATINGS[1] = 'Emphathized with Lee';
RATINGS[2] = 'Empathized with Joanie';
RATINGS[3] = 'Empathized with the girl';
RATINGS[4] = 'Enjoyed the story';
RATINGS[5] = 'Felt that you were engaged with the story';
RATINGS[6] = 'Felt that the neurofeedback helped you';
RATINGS[7] = 'Are certain that your interpretation is correct';
RATINGS[8] = 'Were sleepy in the scanner';
plt.figure(figsize=(20,20))
# make a 3 x 3 subplot
for r in np.arange(nR):
    plt.subplot(3,3,r+1)
    for s in np.arange(nSub):
        if interpretations[s] == 'C':
            index = 0 
        elif interpretations[s] == 'P':
            index = 1
        plt.plot(index+np.random.randn(1)*.1,all_rating_scores[s,r] , '.', color=colors[index], ms=30,alpha=0.3)
    C_mean = np.mean(all_rating_scores[C_ind,r])
    P_mean = np.mean(all_rating_scores[P_ind,r])
    C_sem = scipy.stats.sem(all_rating_scores[C_ind,r])
    P_sem = scipy.stats.sem(all_rating_scores[P_ind,r])
    plt.errorbar(np.arange(2), np.array([C_mean,P_mean]), np.array([C_sem,P_sem]))

    plt.title(RATINGS[r],fontsize=15)
    plt.xticks(np.array([0,1]), labels,fontsize=10) 
    plt.ylim([0.8,5.2])
plt.show()


# what about how much you felt nf helped you versus how you did
plt.figure(figsize=(20,20))
for s in np.arange(nSub):
    if interpretations[s] == 'C':
        index = 0 
        c_score = all_context_scores[s]
    elif interpretations[s] == 'P':
        index = 1
        c_score = all_context_scores[s] * -1
    plt.plot(all_rating_scores[s,6] , c_score,'.', color=colors[index], ms=30,alpha=0.3)
plt.xlabel('How much nf helped')
plt.ylabel('Correct context score')
plt.show()
# plot certainty, etc over other things


