
plt.subplot(331)
plt.imshow(d_1[:,:,15])
plt.title('from dicom OG')
plt.subplot(332)
plt.imshow(d_2[:,:,15])
plt.title('from dicom')
plt.subplot(333)
plt.imshow(a2[:,:,15])
plt.title('from nifti')
plt.subplot(334)
plt.imshow(d_1[:,30,:])
plt.title('from dicom OG')
plt.subplot(335)
plt.imshow(d_2[:,30,:])
plt.title('from dicom')
plt.subplot(336)
plt.imshow(a2[:,30,:])
plt.title('from nifti')
plt.subplot(337)
plt.imshow(d_1[30,:,:])
plt.title('from dicom OG')
plt.subplot(338)
plt.imshow(d_2[30,:,:])
plt.title('from dicom')
plt.subplot(339)
plt.imshow(a2[30,:,:])
plt.title('from nifti')
plt.show()


t = dicomreaders.mosaic_to_nii(dicomImg)
a = t.get_fdata()
output_image_correct = nib.orientations.apply_orientation(a,cfg.axesTransform)
correct_object = new_img_like(testingImg,output_image_correct,copy_header=True)
correct_object.to_filename()


# this means that BEFORE the image is saved out it's already rotated incorrectly

target_orientation = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
dicom_orientation = nib.orientations.axcodes2ornt(('P', 'L', 'S'))
transform = nib.orientations.ornt_transform(dicom_orientation,target_orientation)


plt.subplot(331)
plt.imshow(a[:,:,15])
plt.title('from dicom OG')
plt.subplot(332)
plt.imshow(z[:,:,15])
plt.title('from dicom')
plt.subplot(333)
plt.imshow(a2[:,:,15])
plt.title('from nifti')
plt.subplot(334)
plt.imshow(a[:,30,:])
plt.title('from dicom OG')
plt.subplot(335)
plt.imshow(z[:,30,:])
plt.title('from dicom')
plt.subplot(336)
plt.imshow(a2[:,30,:])
plt.title('from nifti')
plt.subplot(337)
plt.imshow(a[30,:,:])
plt.title('from dicom OG')
plt.subplot(338)
plt.imshow(z[30,:,:])
plt.title('from dicom')
plt.subplot(339)
plt.imshow(a2[30,:,:])
plt.title('from nifti')
plt.show()


#test other image here 
testingImg = '/jukebox/norman/amennen/github/brainiak/rtAttenPenn/greenEyes/data/sub-102/ses-02/converted_niftis/9-11-1.nii.gz'

t2 = nib.load(testingImg)
a2 = t2.get_fdata()
niftiObject = dicomreaders.mosaic_to_nii(dicomImg)
niftiObject.to_filename(full_nifti_output)
    nameToSaveNifti = expected_dicom_name.split('.')[0]
    #base_dicom_name = full_dicom_name.split('/')[-1].split('.')[0]
canonical_img = nib.as_closest_canonical(t)
canonical_img.affine




# PURPOSE: anonmyize dicom data

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
from nilearn.image import new_img_like
from scipy import stats
import scipy.io as sio
import pickle
from rtfMRI.utils import dateStr30
#import greenEyes_043019
from greenEyes_043019 import initializeGreenEyes
#from greenEyes import *
#from greenEyes_043019 import initializeGreenEyes
from rtfMRI.ReadDicom import *
import numpy as np  # type: ignore
from rtfMRI.Errors import StateError
try:
    import pydicom as dicom  # type: ignore
except ModuleNotFoundError:
    import dicom  # type: ignore
from nibabel.nicom import dicomreaders
from nilearn import image
import nibabel as nib
import matplotlib.pyplot as plt




def readDicomFromBuffer(data):
    dataBytesIO = dicom.filebase.DicomBytesIO(data)
    dicomImg = dicom.dcmread(dataBytesIO)
    return dicomImg


def readDicomFromFile(filename):
    dicomImg = dicom.read_file(filename)
    return dicomImg

def anonymizeDicom(dicomFilePath):
    """Read dicom + header, anonymize header"""
    dicomImg = readDicomFromFile(dicomFilePath)
    del dicomImg.PatientID
    del dicomImg.PatientAge
    del dicomImg.PatientBirthDate
    del dicomImg.PatientName
    del dicomImg.PatientSex
    del dicomImg.PatientSize
    del dicomImg.PatientWeight
    del dicomImg.PatientPosition
    return dicomImg

def saveAsNiftiImage(dicomDataObject,expected_dicom_name,cfg):
    A = time.time()
    nameToSaveNifti = expected_dicom_name.split('.')[0] + '.nii.gz'
    tempNiftiDir = os.path.join(cfg.codeDir, 'tmp/convertedNiftis/')
    if not os.path.exists(tempNiftiDir):
        command = 'mkdir -pv {0}'.format(tempNiftiDir)
        call(command,shell=True)
    fullNiftiFilename = os.path.join(tempNiftiDir,nameToSaveNifti)
    niftiObject = dicomreaders.mosaic_to_nii(dicomDataObject)
    temp_data = niftiObject.get_fdata()
    rounded_temp_data = np.round(temp_data)
    output_image_correct = nib.orientations.apply_orientation(temp_data,cfg.axesTransform)
    correct_object = new_img_like(cfg.ref_BOLD,output_image_correct,copy_header=True)
    correct_object.to_filename(fullNiftiFilename)
    B = time.time()
    print(B-A)

configFile = 'greenEyes.toml'
cfg = initializeGreenEyes(configFile)
scanNum = 9
TRnum = 11
expected_dicom_name = cfg.dicomNamePattern.format(scanNum,TRnum)
full_dicom_name = '{0}{1}'.format(cfg.subjectDcmDir,expected_dicom_name)

dicomImg = anonymizeDicom(full_dicom_name)
saveAsNiftiImage(dicomImg,expected_dicom_name,cfg)


# TEST EXACTLY THE SAME  
f1 = '/jukebox/norman/amennen/github/brainiak/rtAttenPenn/greenEyes/tmp/convertedNiftis/9-11-1.nii.gz'
f2 = '/jukebox/norman/amennen/github/brainiak/rtAttenPenn/greenEyes/data/sub-102/ses-02/converted_niftis/9-11-1.nii.gz'

obj_1 = nib.load(f1)
obj_2 = nib.load(f2)
d_1 = obj_1.get_fdata()
d_2 = obj_2.get_fdata()



plt.subplot(331)
plt.imshow(d_1[:,:,15])
plt.title('from dicom OG')
plt.subplot(332)
plt.imshow(d_2[:,:,15])
plt.title('from dicom')
plt.subplot(333)
plt.imshow(a2[:,:,15])
plt.title('from nifti')
plt.subplot(334)
plt.imshow(d_1[:,30,:])
plt.title('from dicom OG')
plt.subplot(335)
plt.imshow(d_2[:,30,:])
plt.title('from dicom')
plt.subplot(336)
plt.imshow(a2[:,30,:])
plt.title('from nifti')
plt.subplot(337)
plt.imshow(d_1[30,:,:])
plt.title('from dicom OG')
plt.subplot(338)
plt.imshow(d_2[30,:,:])
plt.title('from dicom')
plt.subplot(339)
plt.imshow(a2[30,:,:])
plt.title('from nifti')
plt.show()


t = dicomreaders.mosaic_to_nii(dicomImg)
a = t.get_fdata()
output_image_correct = nib.orientations.apply_orientation(a,cfg.axesTransform)
correct_object = new_img_like(testingImg,output_image_correct,copy_header=True)
correct_object.to_filename()


# this means that BEFORE the image is saved out it's already rotated incorrectly

target_orientation = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
dicom_orientation = nib.orientations.axcodes2ornt(('P', 'L', 'S'))
transform = nib.orientations.ornt_transform(dicom_orientation,target_orientation)


plt.subplot(331)
plt.imshow(a[:,:,15])
plt.title('from dicom OG')
plt.subplot(332)
plt.imshow(z[:,:,15])
plt.title('from dicom')
plt.subplot(333)
plt.imshow(a2[:,:,15])
plt.title('from nifti')
plt.subplot(334)
plt.imshow(a[:,30,:])
plt.title('from dicom OG')
plt.subplot(335)
plt.imshow(z[:,30,:])
plt.title('from dicom')
plt.subplot(336)
plt.imshow(a2[:,30,:])
plt.title('from nifti')
plt.subplot(337)
plt.imshow(a[30,:,:])
plt.title('from dicom OG')
plt.subplot(338)
plt.imshow(z[30,:,:])
plt.title('from dicom')
plt.subplot(339)
plt.imshow(a2[30,:,:])
plt.title('from nifti')
plt.show()


#test other image here 
testingImg = '/jukebox/norman/amennen/github/brainiak/rtAttenPenn/greenEyes/data/sub-102/ses-02/converted_niftis/9-11-1.nii.gz'

t2 = nib.load(testingImg)
a2 = t2.get_fdata()
niftiObject = dicomreaders.mosaic_to_nii(dicomImg)
niftiObject.to_filename(full_nifti_output)
    nameToSaveNifti = expected_dicom_name.split('.')[0]
    #base_dicom_name = full_dicom_name.split('/')[-1].split('.')[0]
canonical_img = nib.as_closest_canonical(t)
canonical_img.affine





