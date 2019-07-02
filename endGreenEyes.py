# purpose: get files from cloud server, then delete files

import os
import glob
import numpy as np
from subprocess import call
import time
import nilearn
from scipy import stats
import scipy.io as sio
import pickle
import nibabel as nib
import argparse
import random
import sys
from datetime import datetime
currPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(os.path.dirname(currPath))
sys.path.append(rootPath)
#WHEN TESTING
#sys.path.append('/jukebox/norman/amennen/github/br
#sys.path.append('/Data1/code/rt-cloud/')
from rtCommon.utils import loadConfigFile
from rtCommon.structDict import StructDict

def main():
    # MAKES STRUCT WITH ALL PARAMETERS IN IT
    defaultConfig = os.path.join(currPath , 'conf/greenEyes_organized.toml')
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig,type=str,
                   help='experiment config file (.json or .toml)')
    argParser.add_argument('--addr', '-a', default='localhost', type=str, 
                   help='server ip address')
    argParser.add_argument('--syncCloud', '-s',default=False, action='store_true',
                   help='get data from cloud')
    argParser.add_argument('--syncCluster', '-k', default=False, action='store_true',
                   help='whether or not to move to cluster too')
    argParser.add_argument('--subjectNumber', '-i', default=None, type=str,
                   help='enter specific subject number if you want, otherwise will go with config')
    args = argParser.parse_args()
    params = StructDict({'config': args.config})

    cfg = loadConfigFile(params.config)
    if args.subjectNumber is None:
        cfg.bids_id = 'sub-{0:03d}'.format(cfg.subjectNum)
    else:
        cfg.bids_id = 'sub-{0:03d}'.format(int(args.subjectNumber))
    # get subj
    if cfg.machine == 'intel':
        if args.syncCloud:
            print('transferring from cloud to intelrt for subject %s' % cfg.bids_id)
            intel_subject_full_path = '{0}/data/{1}/'.format(cfg.intelrt.codeDir,cfg.bids_id)
            cloud_subject_full_path = '{0}/data/{1}/'.format(cfg.cloud.codeDir,cfg.bids_id)
            # now see if you need to randomly draw the intepretation
            if args.addr is not 'localhost':
                command = 'rsync -e "ssh -i ~/.ssh/azure_id_rsa" -av --remove-source-files amennen@{0}:{1} {2}'.format(args.addr,cloud_subject_full_path,intel_subject_full_path)
                call(command,shell=True)
            else:
                logging.warning('YOU NEED TO INPUT CLOUD IP ADDR!!')
                print('YOU NEED TO INPUT CLOUD IP ADDR!!')
        if args.syncCluster:
            print('transferring to cluster for subject %s' % cfg.bids_id)
            intel_subject_full_path = '{0}/data/{1}'.format(cfg.intelrt.codeDir,cfg.bids_id)
            cluster_subject_full_path = '/jukebox/norman/amennen/RT_prettymouth/data/intelData/'
            command = 'rsync  -av {0} amennen@scotty:{1} '.format(intel_subject_full_path,cluster_subject_full_path)
            call(command,shell=True)

if __name__ == "__main__":
    # execute only if run as a script
    main()
