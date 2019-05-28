# any file from intel RT will have to be called from file interface

import os
import glob
import numpy as np
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
import shutil
currPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(os.path.dirname(currPath))
sys.path.append(rootPath)

#WHEN TESTING
#sys.path.append('/jukebox/norman/amennen/github/brainiak/rt-cloud')
from rtCommon.utils import loadConfigFile, dateStr30, DebugLevels, writeFile
from rtCommon.readDicom import readDicomFromBuffer
from rtCommon.fileClient import FileInterface
import rtCommon.webClientUtils as wcutils
from rtCommon.structDict import StructDict
import rtCommon.dicomNiftiHandler as dnh
# in tests directory can see test script

defaultConfig = os.path.join(currPath, 'conf/greenEyes_organized.toml')


def initializeGreenEyes(configFile,params):
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
    # MERGE WITH PARAMS
    if params.runs != '' and params.scans != '':
        # use the run and scan numbers passed in as parameters
        cfg.runs = [int(x) for x in params.runs.split(',')]
        cfg.scanNums = [int(x) for x in params.scans.split(',')]

    # GET DICOM DIRECTORY
    if cfg.mode != 'debug':
        if cfg.buildImgPath:
            imgDirDate = datetime.datetime.now()
            dateStr = cfg.date.lower()
            if dateStr != 'now' and dateStr != 'today':
                try:
                    imgDirDate = parser.parse(cfg.date)
                except ValueError as err:
                    raise RequestError('Unable to parse date string {} {}'.format(cfg.date, err))
            datestr = imgDirDate.strftime("%Y%m%d")
            imgDirName = "{}.{}.{}".format(datestr, cfg.subjectName, cfg.subjectName)
            cfg.dicomDir = os.path.join(cfg.intelrt.imgDir, imgDirName)
        else:
            cfg.dicomDir = cfg.intelrt.imgDir # then the whole path was supplied
        cfg.dicomNamePattern = cfg.intelrt.dicomNamePattern
    else:
        cfg.dicomDir = glob.glob(cfg.cluster.imgDir.format(cfg.subjectName))[0]
        cfg.dicomNamePattern = cfg.cluster.dicomNamePattern
    cfg.webpipe = params.webpipe
    cfg.webfilesremote = params.filesremote # FLAG FOR REMOTE OR LOCAL
	########
    cfg.bids_id = 'sub-{0:03d}'.format(cfg.subjectNum)
    cfg.ses_id = 'ses-{0:02d}'.format(cfg.subjectDay)
    if cfg.mode == 'local':
        # then all processing is happening on linux too
        cfg.dataDir = cfg.intelrt.codeDir + 'data'
        cfg.classifierDir = cfg.intelrt.classifierDir
        cfg.mask_filename = cfg.intelrt.maskDir + cfg.MASK
        cfg.MNI_ref_filename = cfg.intelrt.maskDir + cfg.MNI_ref_BOLD
    elif cfg.mode == 'cloud':
        cfg.dataDir = cfg.cloud.codeDir + 'data'
        cfg.classifierDir = cfg.cloud.classifierDir
        cfg.mask_filename = cfg.cloud.maskDir + cfg.MASK
        cfg.MNI_ref_filename = cfg.intelrt.maskDir + cfg.MNI_ref_BOLD
        cfg.intelrt.subject_full_day_path = '{0}/data/{1}/{2}'.format(cfg.intelrt.codeDir,dataDir,cfg.bids_id,cfg.ses_id)
    elif cfg.mode == 'debug':
        cfg.dataDir = cfg.cluster.codeDir + 'data'
        cfg.classifierDir = cfg.cluster.classifierDir
        cfg.mask_filename = cfg.cluster.maskDir + cfg.MASK
        cfg.MNI_ref_filename = cfg.cluster.maskDir + cfg.MNI_ref_BOLD

	
    cfg.subject_full_day_path = '{0}/{1}/{2}'.format(cfg.dataDir,cfg.bids_id,cfg.ses_id)
    cfg.temp_nifti_dir = '{0}/converted_niftis/'.format(cfg.subject_full_day_path)
    cfg.subject_reg_dir = '{0}/registration_outputs/'.format(cfg.subject_full_day_path)
    cfg.nStations, cfg.stationsDict, cfg.last_tr_in_station, cfg.all_station_TRs = getStationInformation(cfg)

	# REGISTRATION THINGS
    cfg.wf_dir = '{0}/{1}/ses-{2:02d}/registration/'.format(cfg.dataDir,cfg.bids_id,1)
    cfg.BOLD_to_T1= cfg.wf_dir + 'affine.txt'
    cfg.T1_to_MNI= cfg.wf_dir + 'ants_t1_to_mniComposite.h5'
    cfg.ref_BOLD=cfg.wf_dir + 'ref_image.nii.gz'

    # GET CONVERSION FOR HOW TO FLIP MATRICES
    cfg.axesTransform = getTransform()
    ###### BUILD SUBJECT FOLDERS #######
    return cfg

def getSubjectInterpretation(cfg):
	# load interpretation file and get it
    # will be saved in subject full day path
    filename = cfg.bids_id + '_' + cfg.ses_id + '_' + 'intepretation.txt'
    full_path_filename = cfg.subject_full_day_path + '/' + filename
    z = open(full_path_filename, "r")
    temp_interpretation = z.read()
    if 'C' in temp_interpretation:
        interpretation = 'C'
    elif 'P' in temp_interpretation:
        interpretation = 'P'
    return interpretation

def getTransform():
	target_orientation = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
	dicom_orientation = nib.orientations.axcodes2ornt(('P', 'L', 'S'))
	transform = nib.orientations.ornt_transform(dicom_orientation,target_orientation)
	return transform


def convertToNifti(TRnum,scanNum,cfg,dicomData):
    #anonymizedDicom = anonymizeDicom(dicomData) # should be anonymized already
    expected_dicom_name = cfg.dicomNamePattern.format(scanNum,TRnum)
    tempNiftiDir = os.path.join(cfg.dataDir, 'tmp/convertedNiftis/')
    nameToSaveNifti = expected_dicom_name.split('.')[0] + '.nii.gz'
    fullNiftiFilename = os.path.join(tempNiftiDir, nameToSaveNifti)
    if not os.path.isfile(fullNiftiFilename): # only convert if haven't done so yet (check if doesn't exist)
       new_nifti_name = dnh.saveAsNiftiImage(dicomData,expected_dicom_name,cfg)
    return new_nifti_name
    # ask about nifti conversion or not

def registerNewNiftiToMNI(cfg,full_nifti_name):
    # should operate over each TR
    # needs full path of nifti file to register
    base_nifti_name = full_nifti_name.split('/')[-1].split('.')[0]
    output_nifti_name = '{0}{1}_space-MNI.nii.gz'.format(cfg.subject_reg_dir,base_nifti_name)
    if not os.path.isfile(output_nifti_name): # only run this code if the file doesn't exist already
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
        command = 'antsApplyTransforms --default-value 0 --float 1 --interpolation LanczosWindowedSinc -d 3 -e 3 --input {0} --reference-image {1} --output {2}{3}_space-MNI.nii.gz --transform {4}{5}_2ref.txt --transform {6} --transform {7} -v 1'.format(full_nifti_name,cfg.MNI_ref_filename,cfg.subject_reg_dir,base_nifti_name,cfg.subject_reg_dir,base_nifti_name,cfg.BOLD_to_T1,cfg.T1_to_MNI)
        A = time.time()
        call(command,shell=True)
        B = time.time()
        print(B-A)

    return output_nifti_name 

def getDicomFileName(cfg, scanNum, fileNum):
    if cfg.mode != 'debug':
        if scanNum < 0:
            raise ValidationError("ScanNumber not supplied of invalid {}".format(scanNum))
        scanNumStr = str(scanNum).zfill(2)
        fileNumStr = str(fileNum).zfill(3)
        if cfg.dicomNamePattern is None:
            raise InvocationError("Missing config settings dicomNamePattern")
        fileName = cfg.dicomNamePattern.format(scanNumStr, fileNumStr)
    else:
        fileName = cfg.dicomNamePattern.format(scanNum,fileNum)
    fullFileName = os.path.join(cfg.dicomDir, fileName)
    return fullFileName

def loadClassifier(cfg,station):
	thisClassifierFileName = cfg.classifierDir + cfg.classifierNamePattern.format(station)
	loaded_model = pickle.load(open(thisClassifierFileName, 'rb'))
	return loaded_model

def getAvgSignal(cfg):
	averageSignal = np.load(cfg.classifierDir +  cfg.averageSignal)
	return averageSignal

def getStationInformation(cfg):
    allinfo = {}
    station_FN = cfg.classifierDir + '/' + cfg.stationDict
    stationDict = np.load(station_FN,allow_pickle=True).item()
    nStations = len(stationDict)
    last_tr_in_station = np.zeros((nStations,))
    allTR = list(stationDict.values())
    all_station_TRs = [item for sublist in allTR for item in sublist]
    for st in np.arange(nStations):
        last_tr_in_station[st] = stationDict[st][-1]
    return nStations, stationDict, last_tr_in_station, all_station_TRs

def getStationClassoutputFilename(runId, stationId):
	""""Return station classification filename"""
	filename = "classOutput_r{}_st{}_py.txt".format(runId,stationId)
	return filename

def getRunFilename(sessionId, runId):
	"""Return run filename given session and run"""
	filename = "patternsData_r{}_{}_py.mat".format(runId,sessionId)
	return filename

def retrieveIntelFileAndSaveToCloud(intelFilePath,pathToSaveOnCloud,fileInterface):
	data = fileInterface.getFile(intelFilePath)
	writeFile(pathToSaveOnCloud,data)

def preprocessData(cfg,dataMatrix,previous_badVoxels=None):
    # steps: zscore///check bad voxels//remove signal average
    # remove bad voxels
    # bad voxel criteria: (1) if raw signal < 100 OR std is < 1E-3 ( I think we're going to set it equal to 0 anyway)

    t_end = np.shape(dataMatrix)[1]
    zscoredData = stats.zscore(dataMatrix,axis=1,ddof = 1)
    zscoredData = np.nan_to_num(zscoredData)
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

def preprocessAndPredict(cfg,runData,TRindex_story):
    """Predict cheating vs. paranoid probability at given station"""
    stationInd = np.argwhere(TRindex_story == cfg.last_tr_in_station.astype(int))[0][0]
    stationKey = 'station' + str(stationInd)
    prevStationKey = 'station' + str(stationInd - 1)
    print('this station is %i' % stationInd)
    print('this story TR is %i' % TRindex_story)
    # indexing for data goes to +1 because we want the index to include the last station TR
    if stationInd == 0 or len(runData.badVoxels) == 0:
    	runData.dataForClassification[stationKey],runData.badVoxels[stationKey] = preprocessData(cfg,runData.story_data[:,0:TRindex_story+1])
    else:
        runData.dataForClassification[stationKey],runData.badVoxels[stationKey] = preprocessData(cfg,runData.story_data[:,0:TRindex_story+1],runData.badVoxels[prevStationkey])
    loaded_model = loadClassifier(cfg,stationInd)
    this_station_TRs = np.array(cfg.stationsDict[stationInd])
    n_station_TRs = len(this_station_TRs)
    if len(runData.badVoxels[stationKey]) > 0:
        voxelsToExclude = runData.badVoxels[stationKey]
        runData.dataForClassification[stationKey][voxelsToExclude,:] = 0
    thisStationData = runData.dataForClassification[stationKey][:,this_station_TRs]
    dataForClassification_reshaped = np.reshape(thisStationData,(1,cfg.nVox*n_station_TRs))
    runData.cheating_probability[stationInd] = loaded_model.predict_proba(dataForClassification_reshaped)[0][1]
    if runData.interpretation == 'C':
        runData.correct_prob[stationInd] = runData.cheating_probability[stationInd]
    elif runData.interpretation == 'P':
        runData.correct_prob[stationInd] = 1 - runData.cheating_probability[stationInd]
    return runData

def makeRunHeader(cfg,runIndex): 
    # Output header 
    now = datetime.datetime.now() 
    print('**************************************************************************************************')
    print('* greenEyes v.1.0') 
    print('* Date/Time: ' + now.isoformat()) 
    print('* Subject Number: ' + str(cfg.subjectNum)) 
    print('* Subject Name: ' + str(cfg.subjectName)) 
    print('* Run Number: ' + str(cfg.runs[runIndex])) 
    print('* Scan Number: ' + str(cfg.scanNums[runIndex])) 
    print('* Real-Time Data: ' + str(cfg.rtData))     
    print('* Mode: ' + str(cfg.mode)) 
    print('* Machine: ' + str(cfg.machine)) 
    print('* Dicom directory: ' + str(cfg.dicomDir)) 
    print('**************************************************************************************************')
    # prepare for TR sequence 
    print('{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}'.format('run','filenum','storyInd', 'taskInfo', 'station', 'p(correct)')) 
    return  

def makeTRHeader(cfg,runIndex,TRFilenum,storyTRCount,stationInd,correct_prob):
    isStation = storyTRCount in cfg.all_station_TRs
    if isStation:
        stStr = 'station'
    else:
        stStr = 'listen'
    print('{:<10.0f}{:<10d}{:<10d}{:<10s}{:<10d}{:<10.3f}'.format(
        cfg.runs[runIndex],TRFilenum,storyTRCount,stStr,stationInd,correct_prob))
    return

def deleteTmpFiles(cfg):
    tempNiftiDir = os.path.join(cfg.dataDir, 'tmp/convertedNiftis/')
    if os.path.exists(tempNiftiDir):
        shutil.rmtree(tempNiftiDir)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('DELETING ALL NIFTIS IN tmp/convertedNiftis')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    return

# testing code--debug mode -- run in greenEyes directory
from greenEyes import *
params = StructDict({'config':defaultConfig, 'runs': '1', 'scans': '9', 'webpipe': 'None', 'webfilesremote': False})
cfg = initializeGreenEyes(params.config,params)
args = StructDict()
args.filesremote = False
webComm = None

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                       help='experiment config file (.json or .toml)')
    argParser.add_argument('--runs', '-r', default='', type=str,
                       help='Comma separated list of run numbers')
    argParser.add_argument('--scans', '-s', default='', type=str,
                       help='Comma separated list of scan number')
    argParser.add_argument('--deleteTmpNifti', '-d', default='1', type=str,
                       help='DO NOT RUN IF RERUNNING WITHIN RUN')
    # creates web pipe communication link to send/request responses through web pipe
    argParser.add_argument('--webpipe', '-w', default=None, type=str,
                       help='Named pipe to communicate with webServer')
    argParser.add_argument('--filesremote', '-x', default=False, action='store_true',
                       help='dicom files retrieved from remote server')

    args = argParser.parse_args()
    print(args)
    cfg = initializeGreenEyes(args.config,args)

    # DELETE ALL FILES IF FLAGGED TO # 
    if args.deleteTmpNifti == '1':
        deleteTmpFiles(cfg)
    # DELETE ALL FILES IF FLAGGED TO # 

    # webpipe
    webComm = None
    if args.webpipe:
        webComm = wcutils.openWebServerConnection(args.webpipe)
        wcutils.watchForExit()
    # initialize file interface class -- for now only local
    fileInterface = FileInterface(filesremote=args.filesremote, webpipes=webComm)
    # intialize watching in particular directory
    fileInterface.initWatch(cfg.dicomDir, cfg.dicomNamePattern, cfg.minExpectedDicomSize) 
    story_TRs = cfg.story_TR_2 - cfg.story_TR_1 + 1
    #### MAIN PROCESSING ###
    nRuns = len(cfg.runs)
    for runIndex in np.arange(nRuns):
        runData = StructDict()
        runData.cheating_probability = np.zeros((cfg.nStations,))
        runData.correct_prob = np.zeros((cfg.nStations,))
        runData.interpretation = getSubjectInterpretation(cfg)
        runData.badVoxels = {}
        runData.dataForClassification = {}
        all_data = np.zeros((cfg.nVox,cfg.nTR_run + 1)) # adding 1 because we're not starting at 0 with the indexing
        runData.story_data = np.zeros((cfg.nVox,story_TRs))

        makeRunHeader(cfg,runIndex)
        run = cfg.runs[runIndex]
        scanNum = cfg.scanNums[runIndex]
        storyTRCount = 0
        stationInd=0
        for TRFilenum in np.arange(cfg.nTR_skip+1,cfg.nTR_run+1):
        # for TRFilenum in np.arange(11,54):
            if TRFilenum == cfg.nTR_skip+1: # wait until run starts
                timeout_file = 180
            else:
                timeout_file = 5
            dicomData = readRetryDicomFromFileInterface(fileInterface, getDicomFileName(cfg, scanNum, TRFilenum), timeout=timeout_file)
            full_nifti_name = convertToNifti(TRFilenum,scanNum,cfg,dicomData)
            registeredFileName = registerNewNiftiToMNI(cfg,full_nifti_name)
            maskedData = apply_mask(registeredFileName,cfg.mask_filename)
            all_data[:,TRFilenum] = maskedData
            if TRFilenum >= cfg.fileNum_story_TR_1 and TRFilenum <= cfg.fileNum_story_TR_2: # we're at a story TR now
                runData.story_data[:,storyTRCount] = maskedData
                if np.any(storyTRCount == cfg.last_tr_in_station.astype(int)):
                    # NOW PREPROCESS AND CLASSIFY
                    stationInd = np.argwhere(storyTRCount == cfg.last_tr_in_station.astype(int))[0][0]
                    runData = preprocessAndPredict(cfg,runData,storyTRCount)
                    text_to_save = '{0:05f}'.format(runData.correct_prob[stationInd])
                    file_name_to_save = getStationClassoutputFilename(run, stationInd)
                    if cfg.mode == 'cloud':
                        full_filename_to_save = os.path.join(cfg.intelrt.subject_full_day_path,file_name_to_save) 
                    else:
                        full_filename_to_save = os.path.join(cfg.subject_full_day_path,file_name_to_save) 
                    fileInterface.putTextFile(full_filename_to_save,text_to_save)
                    if args.webpipe:    
                        wcutils.sendClassicationResult(webComm, run, TRFilenum, val)
                storyTRCount += 1
            TRheader = makeTRHeader(cfg,runIndex,TRFilenum,storyTRCount-1,stationInd,runData.correct_prob[stationInd])

        # SAVE OVER RUN NP FILE
        run_filename = getRunFilename(cfg.sessionId, run)
        full_run_filename_to_save = os.path.join(cfg.subject_full_day_path,run_filename)
        #try:
        sio.savemat(full_run_filename_to_save, runData, appendmat=False)
        #except Exception as err:
        #    errorReply = self.createReplyMessage(msg, MsgResult.Errsor)
        #    errorReply.data = "Error: Unable to save blkGrpFile %s: %r" % (blkGrpFilename, err)
        #    return errorReply
    sys.exit(0)

if __name__ == "__main__":
    # execute only if run as a script
    main()


