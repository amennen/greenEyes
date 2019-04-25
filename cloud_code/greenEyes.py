import os
import glob
from shutil import copyfile
import pandas as pd
import json
import numpy as np
from subprocess import call
from rtfMRI.RtfMRIClient import loadConfigFile
import time
from nilearn.masking import apply_mask

# conda activate /usr/people/amennen/miniconda3/envs/rtAtten/




def initializeGreenEyes(configFile):
	# load subject information
	# create directories for new niftis
	# randomize which category they'll be attending to and save that
	cfg = loadConfigFile(configFile)
	subjectNum = cfg.session.subjectNum
	subjectName = cfg.session.subjectName
	subjectDay = cfg.session.subjectDay
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	ses_id = 'ses-{0:02d}'.format(subjectDay)
	bidsDir = cfg.computer.bidsDir
	codeDir = cfg.computer.codeDir
	dataDir = cfg.computer.dataDir
	dcmDir = cfg.computer.imgDir
	regDir = cfg.computer.registrationDir
	subjectDcmDir = glob.glob(dcmDir.format(subjectName))[0]
	expectedFileName = cfg.session.dicomNamePattern
	classifierDir = codeDir + cfg.session.classifierDir
	###### STATION PARAMETERS #######
	station_FN = classifierDir + cfg.session.stationDict
	stationDict = np.load(station_FN).item()
	nStations = len(stationDict)
	classifiers_FN = classifierDir  + cfg.session.classifierNamePattern
	last_tr_in_station = np.zeros((nStations,))
	for st in np.arange(nStations):
 		last_tr_in_station[st] = stationDict[st][-1]
	###### BUILD SUBJECT FOLDERS #######
	temp_nifti_dir,subject_reg_dir = buildSubjectFolders(subjectNum,subjectDay,dataDir,regDir)
	###### REGISTRATION PARAMETERS #######
	wf_dir='{0}/derivatives/work/fmriprep_wf/single_subject_{1:03d}_wf'.format(bidsDir,subjectNum)
	BOLD_to_T1=wf_dir + '/func_preproc_ses_01_task_story_run_01_wf/bold_reg_wf/bbreg_wf/fsl2itk_fwd/affine.txt'
	T1_to_MNI= wf_dir + '/anat_preproc_wf/t1_2_mni/ants_t1_to_mniComposite.h5'
	ref_BOLD=glob.glob(wf_dir + '/func_preproc_ses_01_task_story_run_01_wf/bold_reference_wf/gen_ref/ref_image.nii.gz')[0]
	MNI_ref_BOLD = cfg.computer.MNI_ref_BOLD


def buildSubjectFolders(subjectNum,subjectDay,dataDir,regDir):
	# make subject directories for each subject number/day/directory
	# day 1: registration and functional
	# day 2: this is where neurofeedback is
	bids_id = 'sub-{0:03d}'.format(subjectNum)
	ses_id = 'ses-{0:02d}'.format(subjectDay)
	subject_full_day_path = '{0}/{1}/{2}'.format(dataDir,bids_id,ses_id)
	command = 'mkdir -pv {0}'.format(subject_full_day_path)
	call(command,shell=True)
	if subjectDay == 1:
		# nothing else here yet, make all data directory
		# nothing really needed for day 1
		pass
	elif subjectDay == 2:
		# make real-time folders ready
		# make temporary nifti folders
		temp_nifti_dir = '{0}/converted_niftis/'.format(subject_full_day_path)
		command = 'mkdir -pv {0}'.format(temp_nifti_dir)
		call(command,shell=True)
		subject_reg_dir = '{0}/registration_outputs/'.format(subject_full_day_path)
		command = 'mkdir -pv {0}'.format(subject_reg_dir)
		call(command,shell=True)
		return temp_nifti_dir, subject_reg_dir


def convertToNifti(full_dicom_name,output_nifti_path,compress):
	# uses dcm2niix to convert incoming dicom to nifti
	# needs to know where to save nifti file output
	# take the dicom file name without the .dcm at the end, and just save that as a .nii
	base_dicom_name = full_dicom_name.split('/')[-1].split('.')[0]
	nameToSaveNifti = '{0}'.format(base_dicom_name)
	if compress:
		command = 'dcm2niix -s y -b n -f {0} -o {1} -z y {2}'.format(nameToSaveNifti,temp_nifti_dir,full_dicom_name)
	else:
		command = 'dcm2niix -s y -b n -f {0} -o {1} -z n {2}'.format(nameToSaveNifti,temp_nifti_dir,full_dicom_name)
	A = time.time()
	call(command,shell=True)
	B = time.time()
	print(B-A)
	if compress:
		new_nifti_name = '{0}{1}.nii.gz'.format(temp_nifti_dir,nameToSaveNifti)
	else:
		new_nifti_name = '{0}{1}.nii'.format(temp_nifti_dir,nameToSaveNifti)
	return new_nifti_name
	# ask about nifti conversion or not

def registerNewNiftiToMNI(full_nifti_name,subject_reg_dir,BOLD_to_T1,T1_to_MNI,ref_BOLD):
	# should operate over each TR
	# needs full path of nifti file to register
	base_nifti_name = full_nifti_name.split('/')[-1].split('.')[0]
	# (1) run mcflirt with motion correction to align to bold reference
	command = 'mcflirt -in {0} -reffile {1} -out {2}{3}_MC -mats'.format(full_nifti_name,ref_BOLD,subject_reg_dir,base_nifti_name)
	A = time.time()
	call(command,shell=True)
	B = time.time()
	print(B-A)
	# (2) run c3daffine tool to convert .mat to .txt
	command = 'c3d_affine_tool -ref {0} -src {1} {2}{3}_MC.mat/MAT_0000 -fsl2ras -oitk {4}{5}_2ref.txt'.format(ref_BOLD,full_nifti_name,subject_reg_dir,base_nifti_name,subject_reg_dir,base_nifti_name)
	A = time.time()
	call(command,shell=True)
	B = time.time()
	print(B-A)
	# (3) combine everything with ANTs call
	command = 'antsApplyTransforms --default-value 0 --float 1 --interpolation LanczosWindowedSinc -d 3 -e 3 --input {0} --reference-image {1} --output {2}{3}_space-MNI.nii.gz --transform {4}{5}_2ref.txt --transform {6} --transform {7} -v 1'.format(full_nifti_name,MNI_ref_BOLD,subject_reg_dir,base_nifti_name,subject_reg_dir,base_nifti_name,BOLD_to_T1,T1_to_MNI)
	A = time.time()
	call(command,shell=True)
	B = time.time()
	print(B-A)
	output_nifti_name = '{0}{1}_space-MNI.nii.gz'.format(subject_reg_dir,base_nifti_name)
	return output_nifti_name 




# FOR TESTING: in cloud_code dir
configFile = 'princetonCfg_display.toml'

#### MAIN PROCESSING ###
compress = 0
## FUNCTION TO OPERATE OVER ALL SCANNING RUNS
SERIES = 9

## FUNCTION TO OPERATE OVER ALL TRS ## -- either cycle through all TRs > certain number or remove afterwards
TR = 0

dicomFileName=expectedFileName.format(SERIES,TR+1)
full_dicom_name = '{0}{1}'.format(subjectDcmDir,dicomFileName)
full_nifti_name = convertToNifti(full_dicom_name,temp_nifti_dir,compress)
# next set to registration pipelien
registeredFileName = registerNewNiftiToMNI(full_nifti_name,BOLD_to_T1,T1_to_MNI,bold_ref_BOLD)

# TO DO: HANDLE ALL BAD VOXELS!!, save timing structures (check for when doing for non-compressed vs. compressed images)
# TO DO: have other functions that are masking, then load, add mask as a filename
# TO DO: ask Grant about .nii vs. .nii.gz
# put this info in config file
nTRs = 485
nVox = 2414
all_current_run_data[:,TR] = apply_mask(registeredFileName,MASK)
if np.any(TR == last_tr_in_station.astype(int)):
	station_ind = np.argwhere(TR == last_tr_in_station.astype(int))[0][0]
	# preprocess data if this is the end of the station

	# classify
else:
	pass # keep just storying data



