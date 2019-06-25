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

## IF RUNNING AS SCRIPT
currPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(os.path.dirname(currPath))
sys.path.append(rootPath)
# when running not in file: sys.path.append(os.getcwd())


#WHEN TESTING
sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
# OR FOR VM
sys.path.append('/home/amennen/code/rt-cloud')
# OR FOR INTELRT
sys.path.append('/Data1/code/rt-cloud/')
from rtCommon.utils import loadConfigFile, dateStr30, DebugLevels, writeFile, loadMatFile
from rtCommon.readDicom import readDicomFromBuffer
from rtCommon.fileClient import FileInterface
import rtCommon.webClientUtils as wcutils
from rtCommon.structDict import StructDict
import rtCommon.dicomNiftiHandler as dnh
import greenEyes

subject=102
#conf='/home/amennen/code/rt-cloud/projects/greenEyes/conf/greenEyes_organized.local.toml'
conf = '/Data1/code/rt-cloud/projects/greenEyes/conf/greenEyes_organized.toml'
args = StructDict()
args.config=conf
args.runs = '1'
args.scans = '5'
args.webpipe = None
args.filesremote = False
cfg = greenEyes.initializeGreenEyes(args.config,args)

r = 0
fileStr = '{0}/patternsData_r{1}*'.format(cfg.subject_full_day_path,r+1)
run_pat = glob.glob(fileStr)[-1]
run_data = loadMatFile(run_pat)

# check classifier
modelfn = '/home/amennen/utils/greenEyes_clf/UPPERRIGHT_stationInd_0_ROI_1_AVGREMOVE_1_filter_0_k1_0_k2_25.sav'
modelfn = '/Data1/code/utils_greenEyes/greenEyes_clf/UPPERRIGHT_stationInd_0_ROI_1_AVGREMOVE_1_filter_0_k1_0_k2_25.sav''
loaded_model = pickle.load(open(modelfn, 'rb'))