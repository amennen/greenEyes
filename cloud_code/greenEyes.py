import os
import glob
from shutil import copyfile
import pandas as pd
import json
import numpy as np
from subprocess import call
from rtfMRI.RtfMRIClient import loadConfigFile
from rtfMRI.StructDict import StructDict
import time
import nilearn
from nilearn.masking import apply_mask
from scipy import stats
import scipy.io as sio
import pickle
from rtfMRI.utils import dateStr30
# conda activate /usr/people/amennen/miniconda3/envs/rtAtten/




def initializeGreenEyes(configFile):
	# load subject information
	# create directories for new niftis
	# randomize which category they'll be attending to and save that
	# purpose: load information and add to configuration things that you won't want to do each time a new file comes in
	# TO RUN AT THE START OF EACH RUN
	cfg = loadConfigFile(configFile)
	if cfg.sessionId in (None, '') or cfg.useSessionTimestamp is True:
		cfg.useSessionTimestamp = True
		cfg.sessionId = dateStr30(time.localtime())
	else:
		cfg.useSessionTimestamp = False
	cfg.bids_id = 'sub-{0:03d}'.format(cfg.subjectNum)
	cfg.ses_id = 'ses-{0:02d}'.format(cfg.subjectDay)
	cfg.dataDir = cfg.codeDir + 'data'
	cfg.subjectDcmDir = glob.glob(cfg.imgDir.format(cfg.subjectName))[0]
	cfg.classifierDir = cfg.codeDir + cfg.classifierDir
	cfg.subject_full_day_path = '{0}/{1}/{2}'.format(cfg.dataDir,cfg.bids_id,cfg.ses_id)
	cfg.temp_nifti_dir = '{0}/converted_niftis/'.format(cfg.subject_full_day_path)
	cfg.subject_reg_dir = '{0}/registration_outputs/'.format(cfg.subject_full_day_path)
	cfg.nStations, cfg.stationsDict, cfg.last_tr_in_station = getStationInformation(cfg)

	# REGISTRATION THINGS
	cfg.wf_dir='{0}/derivatives/work/fmriprep_wf/single_subject_{1:03d}_wf'.format(cfg.bidsDir,cfg.subjectNum)
	cfg.BOLD_to_T1= cfg.wf_dir + '/func_preproc_ses_01_task_story_run_01_wf/bold_reg_wf/bbreg_wf/fsl2itk_fwd/affine.txt'
	cfg.T1_to_MNI= cfg.wf_dir + '/anat_preproc_wf/t1_2_mni/ants_t1_to_mniComposite.h5'
	cfg.ref_BOLD=glob.glob(cfg.wf_dir + '/func_preproc_ses_01_task_story_run_01_wf/bold_reference_wf/gen_ref/ref_image.nii.gz')[0]

	###### BUILD SUBJECT FOLDERS #######
	buildSubjectFolders(cfg)
	###### REGISTRATION PARAMETERS #######
	return cfg

def getSubjectInterpretation(cfg):
	# options
	# 1 - set by experimenter
	# 2 - alternated every other subject
	# 3 - randomized so it's double blind
	interpretation = 'C'
	return interpretation	

def buildSubjectFolders(cfg):
	# make subject directories for each subject number/day/directory
	# day 1: registration and functional
	# day 2: this is where neurofeedback is
	command = 'mkdir -pv {0}'.format(cfg.subject_full_day_path)
	call(command,shell=True)
	if cfg.subjectDay == 1:
		# nothing else here yet, make all data directory
		# nothing really needed for day 1
		pass
	elif cfg.subjectDay == 2:
		# make real-time folders ready
		# make temporary nifti folders
		command = 'mkdir -pv {0}'.format(cfg.temp_nifti_dir)
		call(command,shell=True)
		command = 'mkdir -pv {0}'.format(cfg.subject_reg_dir)
		call(command,shell=True)


def convertToNifti(TRnum,scanNum,cfg):
	# uses dcm2niix to convert incoming dicom to nifti
	# needs to know where to save nifti file output
	# take the dicom file name without the .dcm at the end, and just save that as a .nii
	expected_dicom_name = cfg.dicomNamePattern.format(scanNum,TRnum)
	nameToSaveNifti = expected_dicom_name.split('.')[0]
	#base_dicom_name = full_dicom_name.split('/')[-1].split('.')[0]
	full_dicom_name = '{0}{1}'.format(cfg.subjectDcmDir,expected_dicom_name)
	if cfg.compress:
		command = 'dcm2niix -s y -b n -f {0} -o {1} -z y {2}'.format(nameToSaveNifti,cfg.temp_nifti_dir,full_dicom_name)
	else:
		command = 'dcm2niix -s y -b n -f {0} -o {1} -z n {2}'.format(nameToSaveNifti,cfg.temp_nifti_dir,full_dicom_name)
	A = time.time()
	call(command,shell=True)
	B = time.time()
	print(B-A)
	if cfg.compress:
		new_nifti_name = '{0}{1}.nii.gz'.format(cfg.temp_nifti_dir,nameToSaveNifti)
	else:
		new_nifti_name = '{0}{1}.nii'.format(cfg.temp_nifti_dir,nameToSaveNifti)
	return new_nifti_name
	# ask about nifti conversion or not

def registerNewNiftiToMNI(cfg,full_nifti_name):
	# should operate over each TR
	# needs full path of nifti file to register

	base_nifti_name = full_nifti_name.split('/')[-1].split('.')[0]
	# (1) run mcflirt with motion correction to align to bold reference
	command = 'mcflirt -in {0} -reffile {1} -out {2}{3}_MC -mats'.format(full_nifti_name,cfg.ref_BOLD,cfg.subject_reg_dir,base_nifti_name)
	A = time.time()
	call(command,shell=True)
	B = time.time()
	print(B-A)

	# (2) run c3daffine tool to convert .mat to .txt
	command = 'c3d_affine_tool -ref {0} -src {1} {2}{3}_MC.mat/MAT_0000 -fsl2ras -oitk {4}{5}_2ref.txt'.format(cfg.ref_BOLD,full_nifti_name,cfg.subject_reg_dir,base_nifti_name,cfg.subject_reg_dir,base_nifti_name)
	A = time.time()
	call(command,shell=True)
	B = time.time()
	print(B-A)

	# (3) combine everything with ANTs call
	command = 'antsApplyTransforms --default-value 0 --float 1 --interpolation LanczosWindowedSinc -d 3 -e 3 --input {0} --reference-image {1} --output {2}{3}_space-MNI.nii.gz --transform {4}{5}_2ref.txt --transform {6} --transform {7} -v 1'.format(full_nifti_name,cfg.MNI_ref_BOLD,cfg.subject_reg_dir,base_nifti_name,cfg.subject_reg_dir,base_nifti_name,cfg.BOLD_to_T1,cfg.T1_to_MNI)
	A = time.time()
	call(command,shell=True)
	B = time.time()
	print(B-A)
	output_nifti_name = '{0}{1}_space-MNI.nii.gz'.format(cfg.subject_reg_dir,base_nifti_name)
	return output_nifti_name 



def preprocessData(cfg,dataMatrix,previous_badVoxels=None):
	# steps: zscore///check bad voxels//remove signal average
	# remove bad voxels
	# bad voxel criteria: (1) if raw signal < 100 OR std is < 1E-3 ( I think we're going to set it equal to 0 anyway)

	t_end = np.shape(dataMatrix)[1]
	zscoredData = stats.zscore(dataMatrix,axis=1,ddof = 1)
	zscoredData = np.nan_to_num(dataMatrix)
	# remove story TRs
	# remove story average
	std = np.std(dataMatrix,axis=1,ddof=1)
	non_changing_voxels = np.argwhere(std < 1E-3)
	low_value_voxels = np.argwhere(np.min(dataMatrix,axis=1) < 100)
	badVoxels = np.unique(np.concatenate((non_changing_voxels,low_value_voxels)))
	# now combine with previously made badvoxels
	if previous_badVoxels is not None:
		updated_badVoxels = np.unique(np.concatenate((previous_badVoxels,badVoxels)))
	else:
		updated_badVoxels = badVoxels
	signalAvg = getAvgSignal(cfg) 
	preprocessedData = zscoredData - signalAvg[:,0:t_end]
	return preprocessedData,updated_badVoxels


def loadClassifier(cfg,station):
	thisClassifierFileName = cfg.classifierDir + cfg.classifierNamePattern.format(station)
	loaded_model = pickle.load(open(thisClassifierFileName, 'rb'))
	return loaded_model

def getAvgSignal(cfg):
	averageSignal = np.load(cfg.classifierDir + '/' + cfg.averageSignal)
	return averageSignal

def getStationInformation(cfg):
	allinfo = {}
	station_FN = cfg.classifierDir + '/' + cfg.stationDict
	stationDict = np.load(station_FN).item()
	nStations = len(stationDict)
	last_tr_in_station = np.zeros((nStations,))
	for st in np.arange(nStations):
		last_tr_in_station[st] = stationDict[st][-1]
	return nStations, stationDict, last_tr_in_station


def getRunFilename(sessionId, runId):
	"""Return run filename given session and run"""
	filename = "patternsData_r{}_{}_py.mat".format(runId,sessionId)
	return filename

def preprocessAndPredict(cfg,runData,TRindex_story):
	"""Predict cheating vs. paranoid probability at given station"""
	stationInd = np.argwhere(TRindex_story == cfg.last_tr_in_station.astype(int))[0][0]
	print('this station is %i' % stationInd)
	print('this story TR is %i' % TRindex_story)
	# indexing for data goes to +1 because we want the index to include the last station TR
	if stationInd == 0 or len(runData.badVoxels) == 0:
		runData.dataForClassification[stationInd],runData.badVoxels[stationInd] = preprocessData(cfg,runData.story_data[:,0:TRindex_story+1])
	else:
		runData.dataForClassification[stationInd],runData.badVoxels[stationInd] = preprocessData(cfg,runData.story_data[:,0:TRindex_story+1],runData.badVoxels[stationInd-1])
	loaded_model = loadClassifier(cfg,stationInd)
	this_station_TRs = np.array(cfg.stationsDict[stationInd])
	n_station_TRs = len(this_station_TRs)
	if len(runData.badVoxels[stationInd]) > 0:
		voxelsToExclude = runData.badVoxels[stationInd]
		runData.dataForClassification[stationInd][voxelsToExclude,:] = 0
	thisStationData = runData.dataForClassification[stationInd][:,this_station_TRs]
	dataForClassification_reshaped = np.reshape(thisStationData,(1,cfg.nVox*n_station_TRs))
	runData.cheating_probability[stationInd] = loaded_model.predict_proba(dataForClassification_reshaped)[0][1]
	if runData.interpretation == 'C':
		runData.correct_prob[stationInd] = runData.cheating_probability[stationInd]
	elif runData.interpretation == 'P':
		runData.correct_prob[stationInd] = 1 - runData.cheating_probability[stationInd]
	return runData

# FOR TESTING: in cloud_code dir
def main():
	configFile = 'greenEyes.toml'
	cfg = initializeGreenEyes(configFile)
	runData = StructDict()
	runData.cheating_probability = np.zeros((cfg.nStations,))
	runData.correct_prob = np.zeros((cfg.nStations,))
	runData.interpretation = getSubjectInterpretation(cfg)
	runData.badVoxels = {}
	runData.dataForClassification = {}
	story_TRs = cfg.story_TR_2 - cfg.story_TR_1
	SKIP = 10
	all_data = np.zeros((cfg.nVox,cfg.nTR_run - SKIP)) # don't need to save
	runData.story_data = np.zeros((cfg.nVox,story_TRs))
	#### MAIN PROCESSING ###
	## FUNCTION TO OPERATE OVER ALL SCANNING RUNS
	scanNum = 9
	for TRindex in np.arange(cfg.nTR_run-SKIP):
		print('TRindex')
		#for TRindex in np.arange(44):
		TRnum = TRindex + 1 + SKIP # actual file number to look for
		TRindex_story = TRindex - cfg.story_TR_1
		full_nifti_name = convertToNifti(TRnum,scanNum,cfg)
		registeredFileName = registerNewNiftiToMNI(cfg,full_nifti_name)
		maskedData = apply_mask(registeredFileName,cfg.MASK)
		all_data[:,TRindex] = maskedData
		if TRindex_story >= 0: # we're at a story TR now
			runData.story_data[:,TRindex_story] = maskedData
			if np.any(TRindex_story == cfg.last_tr_in_station.astype(int)):
				# NOW PREPROCESS AND CLASSIFY
				runData = preprocessAndPredict(cfg,runData,TRindex_story)
		else:
			pass
	# now save run data

	#filename = getRunFilename(cfg.sessionId,runId)
	#runFilename = os.path.join(cfg.dataDir,filename)
 	#sio.savemat(runFilename, runId,appendmat=False)
 #    try:
 #    	sio.savemat(runFilename, self.blkGrp, appendmat=False)
	# except Exception as err:
 #    	errorReply = self.createReplyMessage(msg, MsgResult.Error)
 #    	errorReply.data = "Error: Unable to save blkGrpFile %s: %r" % (blkGrpFilename, err)
 #    return errorReply

	
# TO DO:
# check with inputs of scan num, TR number for rtAtten
# put in timing checks--maybe check with grant on that
# go through saving data like Rt ATtten does

if __name__ == "__main__":
    # execute only if run as a script
    main()


