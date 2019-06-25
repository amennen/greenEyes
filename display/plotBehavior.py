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

interpretations = {}
interpretations[101] = 'C'
interpretations[102] = 'P'
interpretations[1] =  'C'
interpretations[2] = 'P'
interpretations[3] = 'P'
interpretations[4] = 'C'
interpretations[5] = 'C'

projectDir = '/Users/amennen/greenEyes/display/data/'

subjects = [101 ,102 ,1 ,2 ,3 ,4 ,5]
nSub = len(subjects)

all_context_scores = np.zeros((nSub,))
all_story_scores = np.zeros((nSub,))
nR = 9
all_rating_scores = np.zeros((nSub,nR))
for s in np.arange(nSub):  
    subject = subjects[s]
    context = interpretations[subject]
    bids_id = 'sub-{0:03d}'.format(subject)
    response_mat = projectDir + bids_id + '/' + 'responses_scored.mat'
    z = scipy.io.loadmat(response_mat)
    ratings =  z['key_rating'][0]
    all_rating_scores[s,:] = ratings
    context_score =  z['mean_context_score'][0][0]
    all_context_scores[s] = context_score
    story_score = z['story_score'][0][0]
    all_story_scores[s] = story_score




# (1) plot story scores
plt.figure()
plt.subplot(1,2,1)
colors = ['r', 'g']
labels = ['cheating', 'paranoid']
for s in np.arange(nSub):
    if interpretations[subjects[s]] == 'C':
        index = 0 
    elif interpretations[subjects[s]] == 'P':
        index = 1

    plt.plot(index,all_story_scores[s] , '.', color=colors[index], ms=20,alpha=0.2)
plt.xlim([-0.5,1.5])
plt.ylim([0.5,1.1])
plt.title('Story comprehension scores')
plt.xticks(np.array([0,1]), labels) 


# (2) plot context scores
plt.subplot(1,2,2)
for s in np.arange(nSub):
    if interpretations[subjects[s]] == 'C':
        index = 0
    elif interpretations[subjects[s]] == 'P':
        index = 1

    plt.plot(index,all_context_scores[s], '.', color=colors[index], ms=20,alpha=0.2)
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
plt.figure()
# make a 3 x 3 subplot
for r in np.arange(nR):
    plt.subplot(3,3,r+1)
    for s in np.arange(nSub):
        if interpretations[subjects[s]] == 'C':
            index = 0 
        elif interpretations[subjects[s]] == 'P':
            index = 1
        plt.plot(index,all_rating_scores[s,r] , '.', color=colors[index], ms=10,alpha=0.3)
    plt.title(RATINGS[r],fontsize=8)
    plt.xticks(np.array([0,1]), labels,fontsize=5) 
    plt.ylim([0.8,5.2])
plt.show()

# plot certainty, etc over other things


