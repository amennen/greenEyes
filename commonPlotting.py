import numpy as np
import glob
import sys
import os
import scipy
sys.path.append(os.getcwd())
import matplotlib
import matplotlib.pyplot as plt
font = {'weight':'normal',
'size':22}
plt.rc('axes',linewidth=5)
import pandas as pd
import seaborn as sns
matplotlib.rc('font',**font)


def printStatsResults(text, t, p):
  print('****************************')
  print(text)
  print('t value is : %0.3f' % t)
  print('p value is : %0.3f' % p)
  return

def getSubjectInterpretation(subject_num):
    # load interpretation file and get it
    # will be saved in subject full day path
    bids_id = 'sub-{0:03d}'.format(subject_num)
    ses_id = 'ses-{0:02d}'.format(2)
    filename = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/' + bids_id + '/' + ses_id + '/' + bids_id + '_' + ses_id + '_' + 'intepretation.txt'
    z = open(filename, "r")
    temp_interpretation = z.read()
    if 'C' in temp_interpretation:
        interpretation = 'C'
    elif 'P' in temp_interpretation:
        interpretation = 'P'
    return interpretation

def getContextResponse(subject_num,question):
    # load interpretation file and get it
    # will be saved in subject full day path
    bids_id = 'sub-{0:03d}'.format(subject_num)
    ses_id = 'ses-{0:02d}'.format(2)
    projectDir = '/jukebox/norman/amennen/RT_prettymouth/data/laptopData/' 
    response_mat = glob.glob(projectDir + bids_id + '/' + 'responses_20*.mat')[0]
    z = scipy.io.loadmat(response_mat)
    s = z['stim']
    responses = s['actual_resp'][0][0][0]
    answer = responses[question]
    return answer


def convertMatToDF(all_data,subjects):
    data = {}
    data_vector = all_data.flatten() # now it's in subj day 1, subj day 2, etc.
    n_subs = len(subjects)
    if np.ndim(all_data)==1:
      nquestions=1
    else:
      nquestions = np.shape(all_data)[1] # data n_subjects x nquestions for the score for each question
    interpretations = {}
    for s in np.arange(n_subs):
      interpretations[s] = getSubjectInterpretation(subjects[s])
    C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
    P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
    group_str = [''] * n_subs
    for s in np.arange(n_subs):
      if s in C_ind:
        group_str[s] = 'C'
      elif s in P_ind:
        group_str[s] = 'P'
    group_vector = np.repeat(group_str,nquestions)
    question_vector = np.tile(np.arange(nquestions),n_subs)
    subject_vector = np.repeat(np.arange(n_subs),nquestions)
    data['data'] = data_vector
    data['subjects'] = subject_vector
    data['group'] = group_vector
    data['q'] = question_vector
    df = pd.DataFrame.from_dict(data)
    return df

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev * 4

def makeColorPalette(colors):
  # Create an array with the colors you want to use
  # Set your custom color palette
  customPalette = sns.color_palette(colors)
  return customPalette

def makeCustomLegend(color_1,color_2,lw):
    custom_lines = [Line2D([0], [0], color=color_1, lw=lw),
                Line2D([0], [0], color=color_2, lw=lw)]
    return custom_lines

def plotPosterStyle_DF(all_data,subjects):
  df = convertMatToDF(all_data,subjects)
  fig,ax = plt.subplots(figsize=(12,9))
  sns.despine()
  P1 = makeColorPalette(['#2ca25f','#de2d26']) # COLORS ARE PARANOID THEN CHEATING
  P2 = makeColorPalette(['#99d8c9','#fc9272'])
  P3 = makeColorPalette(['#e5f5f9','#fee0d2'])
  #sns.set_palette(sns.color_palette(colors))
  sns.barplot(data=df,x='q',y='data',hue='group',ci=68,linewidth=2.5,palette=P2)#errcolor=".2", edgecolor=".2")
  #sns.barplot(data=df,x='day',y='data',hue='group',ci=68,linewidth=2.5,palette=P1,errcolor=".2", edgecolor=".2")
  sns.swarmplot(data=df,x='q',y='data',hue='group',split=True,color='k',size=8,alpha=0.5)
  ax.get_legend().remove()
  #plt.show()
  return fig,ax

def plotPosterStyle_DF_valence(all_data,subjects,ylabel):
  df,df2 = convertMatToDF_valence(all_data,subjects)
  df = df.rename(columns={'data':ylabel})
  df2 = df2.rename(columns={'data':ylabel})
  fig,ax = plt.subplots(figsize=(12,9))
  sns.despine()
  #P1 = makeColorPalette(['#636363','#de2d26'])
  #P2 = makeColorPalette(['#f0f0f0','#fee0d2'])
  #P3 = makeColorPalette(['#bdbdbd','#fc9272'])
  #sns.set_palette(sns.color_palette(colors))
  P2 = makeColorPalette(['#99d8c9','#fc9272'])
  sns.barplot(data=df,x='run',y=ylabel,hue='group',ci=68,linewidth=2.5,palette=P2)#errcolor=".2", edgecolor=".2")
  sns.swarmplot(data=df2,x='run',y=ylabel,hue='group',split=True,color='k',size=8,alpha=0.5)
  ax.legend().remove()
  return fig,ax

def plotPosterStyle_multiplePTS(all_data,subjects):
    """Assume data is in subject x PTS x day """
    colors_dark = ['#2ca25f','#de2d26']
    colors_light = ['#99d8c9','#fc9272']
    n_subs = len(subjects)
    interpretations = {}
    for s in np.arange(n_subs):
      interpretations[s] = getSubjectInterpretation(subjects[s])
    C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
    P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
    C_data = all_data[C_ind,:,:]
    P_data = all_data[P_ind,:,:]
    n_subs = len(subjects)
    C_mean = np.nanmean(C_data,axis=0)
    P_mean = np.nanmean(P_data,axis=0)
    C_err = scipy.stats.sem(C_data,axis=0,nan_policy='omit')
    P_err = scipy.stats.sem(P_data,axis=0,nan_policy='omit')
    alpha=0.2
    nDays = np.shape(all_data)[2]
    nPoints = np.shape(all_data)[1]
    fig,ax = plt.subplots(figsize=(20,9))
    for d in np.arange(nDays):
      plt.subplot(1,nDays,d+1)
      sns.despine()
      for s in np.arange(n_subs):
          if s in P_ind:
              color = 0
          elif s in C_ind:
              color = 1
          #plt.plot(all_data[s,:,d],'-',ms=10,color=colors_light[color],alpha=alpha,lw=2)
      plt.errorbar(x=np.arange(nPoints),y=C_mean[:,d],yerr=C_err[:,d],color=colors_light[1],lw=5,label='C',fmt='-o',ms=10)
      plt.errorbar(x=np.arange(nPoints),y=P_mean[:,d],yerr=P_err[:,d],color=colors_light[0],lw=5,label='P',fmt='-o',ms=10)
      plt.xlabel('point')
      #plt.ylabel('area under -0.1')
      plt.xticks(np.arange(nPoints))
    #  plt.legend()
    #ax.get_legend().remove()
    #plt.show()
    return fig

def addComparisonStat(score,x1,x2,maxHeight,heightAbove):
  y,h,col = maxHeight + heightAbove, heightAbove, 'k'
  plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
  text_stat = "1-sided p = {0:2.4f}".format(score)
  plt.text((x1+x2)*.5, y+h, text_stat, ha='center', va='bottom', color=col,fontsize=25)
  return

def addComparisonStat_SYM(score,x1,x2,maxHeight,heightAbove,fontH,text_above=[]):
  y,h,col = maxHeight, heightAbove, 'k'
  y2 = y+h
  h2 = y+h+h
  if score < 0.0001:
    text = '****'
  elif score < 0.001:
    text = '***'
  elif score < 0.01:
    text = '**'
  elif score < 0.05:
    text = '*'
  elif score < 0.1:
    text = '+'
  elif score > 0.1:
    text = 'ns'
  if score < 0.1:
    if x1 != x2:
      plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    if len(text_above) > 0:
      text_stat = "{}\n{}".format(text_above,text)
    else:
      text_stat = text
    plt.text((x1+x2)*.5, y+h+(fontH), text_stat, ha='center', va='bottom', color=col,fontsize=22,fontstyle='normal')
  return

def addSingleStat(score,x,maxHeight,heightAbove):
  y,h,col = maxHeight + heightAbove, heightAbove, 'k'
  plt.plot([x, x], [y, y+h], lw=1.5, c=col)
  text_stat = "1-sided p = {0:2.4f}".format(score)
  plt.text(x, y+h, text_stat, ha='center', va='bottom', color=col,fontsize=15)
  return

def addSingleStat_ax(score,x,maxHeight,heightAbove,ax):
  y,h,col = maxHeight + heightAbove, heightAbove, 'k'
  ax.plot([x, x], [y, y+h], lw=1.5, c=col)
  text_stat = "p = {0:2.4f}".format(score)
  ax.text(x, y+h, text_stat, ha='center', va='bottom', color=col,fontsize=15)
  return

def nonNan(x,y):
  if len(x) == len(y): # then treat as paired case
    if any(np.isnan(x)) and not any(np.isnan(y)):
        xnew = x[~np.isnan(x)]
        ynew = y[~np.isnan(x)]
    elif any(np.isnan(y)) and not any(np.isnan(x)):
        ynew = y[~np.isnan(y)]
        xnew = x[~np.isnan(y)]
    elif any(np.isnan(x)) and any(np.isnan(y)):
        bad_x = np.argwhere(np.isnan(x))
        bad_y = np.argwhere(np.isnan(y))
        all_bad = np.unique(np.concatenate((bad_x,bad_y)))
        all_indices = np.arange(len(x))
        keep_indices = np.delete(all_indices,all_bad)
        xnew = x[keep_indices]
        ynew=y[keep_indices]
    elif not any(np.isnan(x)) and not any(np.isnan(y)):
        print('NO NANS!')
        xnew=x
        ynew=y
  else:
    # if they are different numbers just remove what is nan
    if any(np.isnan(x)):
      xnew = x[~np.isnan(x)]
    else:
      xnew = x
    if any(np.isnan(y)):
      ynew = y[~np.isnan(y)]
    else:
      ynew = y
  return xnew,ynew


def convertMatToDF_valence(all_data,subjects):
    # let's assume all_data - in this case it's n_subs x n_stations x n_runs
    data = {}
    data_vector = all_data.flatten() # now it's in subj day 1, subj day 2, etc.
    n_runs = np.shape(all_data)[2]
    n_subs = len(subjects)
    n_stations = np.shape(all_data)[1]
    subject_vector = np.repeat(subjects,n_runs*n_stations)
    station_vector = np.tile(np.repeat(np.arange(n_stations),n_runs),n_subs)
    run_vector = np.tile(np.arange(n_runs),n_stations*n_subs)
    interpretations = {}
    for s in np.arange(n_subs):
      interpretations[s] = getSubjectInterpretation(subjects[s])
    C_ind = [sub for sub, interp in interpretations.items() if interp == 'C']
    P_ind = [sub for sub, interp in interpretations.items() if interp == 'P']
    group_str = [''] * n_subs
    for s in np.arange(n_subs):
      if s in C_ind:
        group_str[s] = 'cheating'
      elif s in P_ind:
        group_str[s] = 'paranoid'
    group_vector = np.repeat(group_str,n_stations*n_runs)

    data['data'] = data_vector
    data['subjects'] = subject_vector
    data['group'] = group_vector
    data['run'] = run_vector
    data['station'] = station_vector
    df = pd.DataFrame.from_dict(data)

    # make new dataframe by computing subject average within each run
    data_averaged_over_station = np.nanmean(all_data,axis=1)
    data2 = {}
    data_vector = data_averaged_over_station.flatten()
    subject_vector = np.repeat(subjects,n_runs)
    group_vector = np.repeat(group_str,n_runs)
    run_vector = np.tile(np.arange(n_runs),n_subs)
    data2['data'] = data_vector
    data2['subjects'] = subject_vector
    data2['group'] = group_vector
    data2['run'] = run_vector
    df2 = pd.DataFrame.from_dict(data2)

    return df,df2