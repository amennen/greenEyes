
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


