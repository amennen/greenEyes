# purpose: plot behavioral results for green eyes, behavior 
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
font = {'weight': 'normal',
        'size': 22}
plt.rc('font', **font)
from commonPlotting import *


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

# change interpretation scores to account for the bad question
z=all_context_scores*12
new_context_scores = (z.copy() + 1)/11
all_correct_context = new_context_scores.copy()
all_correct_context[P_ind] = -1*new_context_scores[P_ind]
all_context_scores = new_context_scores
scores = np.concatenate((all_story_scores[:,np.newaxis],all_context_scores[:,np.newaxis]),axis=1)

# subject empathy ratings
subject_ratings_empathy = all_rating_scores[:,0:4]
arthur_minus_lee = all_rating_scores[:,0] - all_rating_scores[:,1]
subject_ratings_empathy_diff = np.concatenate((all_rating_scores[:,0:2],arthur_minus_lee[:,np.newaxis],all_rating_scores[:,2:4]),axis=1)

# (1) plot comprehension scores by group
maxH = 1.1
fig,ax = plotPosterStyle_DF(scores[:,0],subjects)
plt.xticks(np.array([-.2,.2]), ['paranoid','cheating'],fontsize=20) 
plt.ylabel('comprehension score')
plt.xlabel('group')
plt.title('comprehension score')
# x,y=nonNan(scores[P_ind,0],[])
# t,p = scipy.stats.ttest_1samp(x,0.5)
# addComparisonStat_SYM(p/2,-.2,-0.2,maxH,.05,0,text_above='P>0.5')
# x,y=nonNan(scores[C_ind,0],[])
# t,p = scipy.stats.ttest_1samp(x,0.5)
# addComparisonStat_SYM(p/2,.2,0.2,maxH,.05,0,text_above='C>0.5')
plt.ylim([0.5,1.05])
plt.yticks(np.array([0.5,0.75,1]))
plt.savefig('savedPlots_checked/comprehension_score.pdf')
#plt.show()
# t-test: did the groups differ in score
x,y=nonNan(scores[P_ind,0],scores[C_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
printStatsResults('comprehension group diff',t, p)

# (2) plot interpretation scores by group
fig,ax = plotPosterStyle_DF(scores[:,1],subjects)
plt.xticks(np.array([-.2,.2]), ['paranoid','cheating'],fontsize=20) 
plt.ylabel('interpretation score')
plt.xlabel('group')
plt.title('interpretation score')
plt.plot([-2,2],[0,0], '--', color='k')
plt.yticks(np.array([-1,0,1]), ['paranoid','neutral','cheating'],fontsize=20,rotation=45) 
maxH=1.03
r,p = scipy.stats.ttest_ind(scores[P_ind,1],scores[C_ind,1])
addComparisonStat_SYM(p/2,-.2,.2,maxH,.05,0,text_above='C > P')
printStatsResults('interpretation group diff',r, p/2)
plt.ylim([-1.2,1.5])
plt.savefig('savedPlots_checked/context_score.pdf')
#plt.show()
# did either group show significant bias in the category?
r,p = scipy.stats.ttest_1samp(scores[P_ind,1],0)
printStatsResults('did the paranoid group show bias?',r, p/2)
r,p = scipy.stats.ttest_1samp(scores[C_ind,1],0)
printStatsResults('did the cheating group show bias?',r, p/2)

# (3) plot empathy ratings, first for all characters
maxH=5.1
fig,ax = plotPosterStyle_DF(subject_ratings_empathy,subjects)
plt.title('How much do you empathize with...')
nq = 4
labels=['Arthur','Lee','Joanie','the girl']
plt.ylabel('empathy (1-5)')
plt.xticks(np.arange(nq), labels,fontsize=20) 
plt.yticks(np.arange(1,6),fontsize=20)
plt.ylim([.1,6.5])
x,y=nonNan(all_rating_scores[C_ind,0],all_rating_scores[P_ind,0])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,0.2,maxH,.05,0,text_above='C>P')
printStatsResults('Arthur empathy diff',t, p/2)
x,y=nonNan(all_rating_scores[C_ind,1],all_rating_scores[P_ind,1])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,0.8,1.2,maxH,.05,0,text_above='C<P')
printStatsResults('Lee empathy diff',t, p/2)
x,y=nonNan(all_rating_scores[C_ind,2],all_rating_scores[P_ind,2])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,1.8,2.2,maxH,.05,0,text_above='C<P')
printStatsResults('Joanie empathy diff',t, p/2)
x,y=nonNan(all_rating_scores[C_ind,3],all_rating_scores[P_ind,3])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,2.8,3.2,maxH,.05,0,text_above='')
printStatsResults('the girl empathy diff',t, p)
plt.xlabel('')
plt.savefig('savedPlots_checked/all_ratings.pdf')
#plt.show()

# (4) plot empathy difference for Arthur - Lee
fig,ax = plotPosterStyle_DF(arthur_minus_lee,subjects)
plt.title('Arthur minus Lee')
labels=['paranoid','cheating']
plt.xticks(np.array([-.2,.2]), labels,fontsize=20) 
plt.yticks(np.array([-5,0,5]),fontsize=20)
plt.ylim([-5,6.5])
plt.xlabel('')
plt.ylabel('empathy difference')
plt.plot([-2,2],[0,0], '--', color='k')
x,y=nonNan(arthur_minus_lee[C_ind],arthur_minus_lee[P_ind])
t,p = scipy.stats.ttest_ind(x,y)
addComparisonStat_SYM(p/2,-.2,0.2,maxH,.05,0,text_above='C>P')
printStatsResults('Arthur - Lee empathy diff',t, p/2)
plt.savefig('savedPlots_checked/arthur_minus_lee.pdf')
#plt.show()


# plot interpretation score by whether it was correct or incorrect instead
# of by interpretation
# fig,ax = plotPosterStyle_DF(all_correct_context,subjects)
# plt.xticks(np.array([-.2,.2]), ['paranoid','cheating'],fontsize=20) 
# plt.ylabel('context score')
# plt.xlabel('group')
# plt.title('context score')
# plt.yticks(np.array([-1,1]), ['incorrect','correct'],fontsize=20,rotation=45) 

# #r,p = scipy.stats.ttest_1samp(all_correct_context[C_ind],0)
# #addComparisonStat_SYM(p,-.2,.2,0.9,.05,0,text_above='C P')
# plt.ylim([-1.2,1.2])
# plt.savefig('savedPlots/correct_context_score.pdf')
# plt.show()


# plotting all ratings by question
# RATINGS = {}
# RATINGS[0] = 'Empathized with Arthur';
# RATINGS[1] = 'Emphathized with Lee';
# RATINGS[2] = 'Empathized with Joanie';
# RATINGS[3] = 'Empathized with the girl';
# RATINGS[4] = 'Enjoyed the story';
# RATINGS[5] = 'Felt that you were engaged with the story';
# RATINGS[6] = 'Felt that the neurofeedback helped you';
# RATINGS[7] = 'Are certain that your interpretation is correct';
# RATINGS[8] = 'Were sleepy in the scanner';
# plt.figure(figsize=(20,20))
# # make a 3 x 3 subplot
# for r in np.arange(nR):
#     plt.subplot(3,3,r+1)
#     for s in np.arange(nSub):
#         if interpretations[s] == 'C':
#             index = 0 
#         elif interpretations[s] == 'P':
#             index = 1
#         plt.plot(index+np.random.randn(1)*.1,all_rating_scores[s,r] , '.', color=colors[index], ms=30,alpha=0.3)
#     C_mean = np.mean(all_rating_scores[C_ind,r])
#     P_mean = np.mean(all_rating_scores[P_ind,r])
#     C_sem = scipy.stats.sem(all_rating_scores[C_ind,r])
#     P_sem = scipy.stats.sem(all_rating_scores[P_ind,r])
#     plt.errorbar(np.arange(2), np.array([C_mean,P_mean]), np.array([C_sem,P_sem]))

#     plt.title(RATINGS[r],fontsize=15)
#     plt.xticks(np.array([0,1]), labels,fontsize=10) 
#     plt.ylim([0.8,5.2])
# plt.show()

# plotting efficacy ratings across groups
# fig,ax = plotPosterStyle_DF(efficacy_FB,subjects)
# plt.xticks(np.array([-.2,.2]), ['paranoid','cheating'],fontsize=20) 
# plt.ylabel('context score')
# plt.xlabel('group')
# plt.title('How much do you feel liek your lens controlled the NF score?')
# plt.yticks(np.array([1,10]), ['unrelated','determined'],fontsize=20,rotation=45) 
# #r,p = scipy.stats.ttest_1samp(all_correct_context[C_ind],0)
# #addComparisonStat_SYM(p,-.2,.2,0.9,.05,0,text_above='C P')
# #plt.ylim([-1.2,1.2])
# plt.show()

# plotting how much neurofeedback helps and interpretation scores
# arthur_minus_lee = all_rating_scores[:,0] - all_rating_scores[:,1]
# x,y=nonNan(arthur_minus_lee[C_ind],arthur_minus_lee[P_ind])
# t,p = scipy.stats.ttest_ind(x,y)
# # what about how much you felt nf helped you versus how you did
# plt.figure(figsize=(20,20))
# for s in np.arange(nSub):
#     if interpretations[s] == 'C':
#         index = 0 
#     elif interpretations[s] == 'P':
#         index = 1
#     plt.plot(all_rating_scores[s,6] , all_correct_context[s],'.', color=colors[index], ms=30,alpha=0.3)
# x,y=nonNan(all_rating_scores[:,6],all_correct_context)
# t,p = scipy.stats.pearsonr(x,y)
# plt.xlabel('How much nf helped')
# plt.ylabel('Correct context score')
# plt.show()
# # get diff
# x,y=nonNan(arthur_minus_lee[C_ind],arthur_minus_lee[P_ind]) # expect cheating to have more emphathy for arthur so cheating larger
# t,p = scipy.stats.ttest_ind(x,y)

# Plotting by one question at a time
# CONTEXT_QUESTIONS = [5, 9 ,27 ,28, 29, 30 ,34 ,35 ,36 ,37 ,38 ,39]
# ### NOW - PLOT EACH STORY ONE AT A TIME ###
# QUESTIONS = {}
# nContext = len(CONTEXT_QUESTIONS)
# for q in CONTEXT_QUESTIONS:
#     QUESTIONS[q] = {}
# QUESTIONS[5][0] = 'What was Lee''s girlfriend name?'
# QUESTIONS[5][1] = 'Joanie'
# QUESTIONS[5][2] = 'Rosie'
# QUESTIONS[9][0] = 'What did you think of Arthur when he said, "I have a feeling she went to work on some bastard in the kitchen."?'
# QUESTIONS[9][1] = 'He knew what he was talking about'
# QUESTIONS[9][2] = 'He was being paranoid'
# QUESTIONS[27][0] = 'What did you think - Why did Joanie come back home so late?'
# QUESTIONS[27][1] = 'She was with another man'
# QUESTIONS[27][2] = 'She went to drink and help her friends'
# QUESTIONS[28][0] = 'Why do you think Lee reacted that way?'
# QUESTIONS[28][1] = 'He thought Arthur was lying'
# QUESTIONS[28][2] = 'He realized Arthur was having one of his paranoid episodes again'
# QUESTIONS[29][0] = 'Did you believe Arthur? Did you think Joanie really came back home?'
# QUESTIONS[29][1] = 'No'
# QUESTIONS[29][2] = 'Yes'
# QUESTIONS[30][0] = 'If you didn''t believe Arthur, why do you think he lied about Joanie coming back home?'
# QUESTIONS[30][1] = 'He wanted to test Lee''s reaction'
# QUESTIONS[30][2] = 'He wanted to protect his image'
# QUESTIONS[34][0] = 'When you heard the phone conversation, did you think Arthur suspected Joanie was with Lee?'
# QUESTIONS[34][1] = 'Yes'
# QUESTIONS[34][2] = 'No'
# QUESTIONS[35][0] = 'Did you think Joanie was cheating on Arthur?'
# QUESTIONS[35][1]= 'Yes'
# QUESTIONS[35][2] = 'No'
# QUESTIONS[36][0] = 'If you did think she was cheating on him, with whom?'
# QUESTIONS[36][1] = 'Lee'
# QUESTIONS[36][2] = 'Another man'
# QUESTIONS[37][0]  = 'When the phone rang at the first time, why did you think the gray-haired man asked the girl if she would rather he didn''t answer it?'
# QUESTIONS[37][1]  = 'Because they were afraid it was her husband'
# QUESTIONS[37][2] = 'Because they were desperate to go to sleep'
# QUESTIONS[38][0] = 'Why do you think Lee didn''t tell Arthur that there was a girl at his place?'
# QUESTIONS[38][1] = 'He didn''t want Arthur to suspect anything'
# QUESTIONS[38][2] = 'He didn''t want Arthur to feel that he is interrupting'
# QUESTIONS[39][0] = 'Why do you think Lee didn''t want Arthur to come over?'
# QUESTIONS[39][1]  = 'Because Joanie was there'
# QUESTIONS[39][2]  = 'Because he was with his girlfriend, and he didn''t want to be interrupted'

# # get all context_responses
# all_context_resp = np.zeros((nSub,nContext))
# for s in np.arange(nSub):
#     for q in np.arange(nContext):
#         all_context_resp[s,q] = getContextResponse(subjects[s],CONTEXT_QUESTIONS[q]-1)

# plt.figure(figsize=(20,20))
# # make a 3 x 3 subplot
# for q in np.arange(nContext):
#     plt.subplot(3,4,q+1)
#     for s in np.arange(nSub):
#         if interpretations[s] == 'C':
#             index = 0 
#         elif interpretations[s] == 'P':
#             index = 1
#         plt.plot(index+np.random.randn(1)*.1,all_context_resp[s,q] , '.', color=colors[index], ms=15,alpha=0.3)
#     C_mean = np.mean(all_context_resp[C_ind,q])
#     P_mean = np.mean(all_context_resp[P_ind,q])
#     C_sem = scipy.stats.sem(all_context_resp[C_ind,q])
#     P_sem = scipy.stats.sem(all_context_resp[P_ind,q])
#     plt.errorbar(np.arange(2), np.array([C_mean,P_mean]), np.array([C_sem,P_sem]),color='k',lw=5)

#     plt.title(QUESTIONS[CONTEXT_QUESTIONS[q]][0],fontsize=5)
#     plt.xticks(np.array([0,1]), labels,fontsize=5) 
#     labels_answer = [QUESTIONS[CONTEXT_QUESTIONS[q]][1],QUESTIONS[CONTEXT_QUESTIONS[q]][2]]
#     plt.yticks(np.array([1,2]), labels_answer,fontsize=5) 
#     #plt.ylim([0.8,5.2])
# plt.show()


# # subject_ratings 
# for q in np.arange(nContext):
#     fig,ax = plotPosterStyle_DF(all_context_resp[:,q],subjects)
#     plt.xticks(np.array([-.2,.2]), ['paranoid','cheating'],fontsize=15) 
#     plt.title(QUESTIONS[CONTEXT_QUESTIONS[q]][0],fontsize=10)
#     labels_answer = [QUESTIONS[CONTEXT_QUESTIONS[q]][1],QUESTIONS[CONTEXT_QUESTIONS[q]][2]]
#     plt.yticks(np.array([1.5,2.5]), labels_answer,fontsize=10,rotation=70) 
#     x,y=nonNan(all_context_resp[C_ind,q],all_context_resp[P_ind,q])
#     t,p = scipy.stats.ttest_ind(x,y)
#     maxH=2.2
#     if np.isnan(p):
#         pass
#     else:
#         addComparisonStat_SYM(p/2,-.2,0.2,maxH,.05,0,text_above='C<P')
#     plt.ylim([0.5,2.5])
#     plt.ylabel('')
#     #plt.show()
#     fig_name = '{}/resp_{}.pdf'.format(saveDir,q)
#     plt.savefig(fig_name)


# plt.plot(all_context_scores,all_context_resp[:,7], '.')
# plt.show()