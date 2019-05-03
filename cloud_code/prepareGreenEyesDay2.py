# PURPOSE: get registration transferred and ready on intel linux

import os
import glob
import numpy as np
from subprocess import call
from rtfMRI.RtfMRIClient import loadConfigFile
from rtfMRI.StructDict import StructDict
import time
import nilearn
from scipy import stats
import scipy.io as sio
import pickle
import nibabel as nib
import argparse


# subject path from previous day
def copyClusterFileToIntel(fileOnCluster,pathOnLinux):
	"""This copies a file from the cluster to intel, assuming you're on the intel linux calling the function"""
	command = 'scp amennen@apps.pni.princeton.edu:{0} {1} '.format(fileOnCluster,pathOnLinux)
	call(command,shell=True)
	#return command


def main():
	
	# MAKES STRUCT WITH ALL PARAMETERS IN IT
	argParser = argparse.ArgumentParser()
	argParser.add_argument('--config', '-c', default='greenEyes_organized.toml', type=str,
	                   help='experiment config file (.json or .toml)')
	args = argParser.parse_args()
	params = StructDict({'config': args.config})

	cfg = loadConfigFile(params.config)
	cfg.bids_id = 'sub-{0:03d}'.format(cfg.subjectNum)
	cfg.ses_id = 'ses-{0:02d}'.format(1)
	cluster_wf_dir='{0}/derivatives/work/fmriprep_wf/single_subject_{1:03d}_wf'.format(cfg.clusterBidsDir,cfg.subjectNum)
	cluster_BOLD_to_T1= cluster_wf_dir + '/func_preproc_ses_01_task_story_run_01_wf/bold_reg_wf/bbreg_wf/fsl2itk_fwd/affine.txt'
	cluster_T1_to_MNI= cluster_wf_dir + '/anat_preproc_wf/t1_2_mni/ants_t1_to_mniComposite.h5'
	cluster_ref_BOLD=glob.glob(cluster_wf_dir + '/func_preproc_ses_01_task_story_run_01_wf/bold_reference_wf/gen_ref/ref_image.nii.gz')[0]
	local_subject_registration  = cfg.day1Dir.format(cfg.bids_id,cfg.ses_id)
	if not os.path.exists(local_subject_registration):
		os.makedirs(local_subject_registration)
	
	copyClusterFileToIntel(cluster_BOLD_to_T1,local_subject_registration)
	copyClusterFileToIntel(cluster_T1_to_MNI,local_subject_registration)
	copyClusterFileToIntel(clutser_ref_BOLD,local_subject_registration)




if __name__ == "__main__":
    # execute only if run as a script
    main()
