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
    args = argParser.parse_args()
    params = StructDict({'config': args.config})

    cfg = loadConfigFile(params.config)
    cfg.bids_id = 'sub-{0:03d}'.format(cfg.subjectNum)
    cfg.ses_id = 'ses-{0:02d}'.format(cfg.subjectDay)
    # get subj
    if cfg.machine == 'intel':

        intel_subject_full_path = '{0}/data/{1}/'.format(cfg.intelrt.codeDir,cfg.bids_id)
        cloud_subject_full_path = '{0}/data/{1}/'.format(cfg.cloud.codeDir,cfg.bids_id)
        # now see if you need to randomly draw the intepretation
        if args.addr is not 'localhost':
            command = 'rsync -e "ssh -i ~/.ssh/azure_id_rsa" -av --remove-source-files amennen@{0}:{1} {2}'.format(args.addr,cloud_subject_full_path,intel_subject_full_path)
            call(command,shell=True)
        else:
            logging.warning('YOU NEED TO INPUT CLOUD IP ADDR!!')
            print('YOU NEED TO INPUT CLOUD IP ADDR!!')
if __name__ == "__main__":
    # execute only if run as a script
    main()
