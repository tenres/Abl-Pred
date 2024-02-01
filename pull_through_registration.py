# This file pulls necessary objects through a registration transformation function

# Imports
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob
import re


def get_files(subject_folder, name, ext):
    """Get files with names that match a particular wild card
        name = substring that identifies the file
        ext = File extension to search
    """
    file_card = os.path.join(subject_folder, "*" + name + "*" + ext)
    try:
        return glob.glob(file_card)
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))



if __name__ == '__main__':

    # NOTE: We follow the same style of code we used in "align_antenna_and_rotation.py"    
    #=======================================================================================================
    
    # Set the data base folder
    data_base_folder = r'/home/path_to_registered_data'

    # The output path 
    output_data_base_path = os.path.join(r'/home/path_to_output_data')


    # Some settings that are used below   
    crop_followup_post = True
    crop_pre = True
    ref_reg_space = 'followup_post'
    
    # List all the valid cases with the relevant files with full paths
    #===============================================================================================
    # Initialize the valid case list, file path dictionary and invalid list
    invalid_cases_list = []
    failed_cases_error_msg = []
    # Iterate through registered case folders
    for subject_folder in glob.glob(os.path.join(data_base_folder, "*")):
        
        # Assuming all directories in this folder are case directories
        if os.path.isdir(subject_folder):
            
            # Get name of the case
            case_name = os.path.basename(subject_folder)

            # Initialize empty strings for the various case files
            followup_post_scan_file = ''
            followup_post_seg_file = ''
            pre_scan_file = ''
            pre_seg_file = ''

            if crop_pre and crop_followup_post:
                if ref_reg_space == 'followup_post':

                    # Make sure the case is valid, i.e., it has all the necessary files.
                    # READ THE PRE FILE POST REGISTRATION
                    # Find pre registered to the followup space
                    pre_scan = get_files(subject_folder, 'pre_in_post_nonRigid_*' + 'Cropped*' 'mask30Both_ablRigidPenaltyPre*', '.nrrd') # Note that the function adds additional wild cards between the inputs
                    #pre_scan = get_files(subject_folder, 'pre_in_post_nonRigid_*' + 'Cropped*' 'mask50Both_ablRigidPenaltyPre*', '.nrrd') # Note that the function adds additional wild cards between the inputs
                    # Check validity of the query result. There should be exactly one such file
                    if len(pre_scan) != 1:
                        invalid_cases_list.append(case_name)
                        continue
                    # Set full file path string
                    pre_scan_file = pre_scan[0]

                    # Get the pre segmentation file PRIOR TO REGISTRATION
                    #--------------------------------------------
                    # Find the pre segmentation file. This could give us multiple files since the segmentation files are 
                    # saved at the beginning of every registration attempt (i.e., with decreasing mask sizes).    
                    pre_seg_files = get_files(subject_folder, 'pre_' + '*' + '*seg*' + '*cropped*', '.nrrd') # Note that the function adds additional wild cards between the inputs
                    if not pre_seg_files:
                        invalid_cases_list.append(case_name)
                        continue
                    # Find the segmentation file with the smallest crop radius. This is the one for which the registration succeeded.
                    # Sort in increasing order and take the first seg file.
                    pre_seg_files.sort()
                    pre_seg_file = pre_seg_files[0]       


                    # Followup post files
                    #-------------------------------------------------------------------------------------
                    # The followup post segmentations are cropped in each registration attempt with decreasing cropped radii.
                    # Hence there may be multiple followup post segmentation files. Use the one that has the smallest radius in 
                    # its name since that is the one for which the registration succeeded. But the followup post scan itself 
                    # should be only 1 since it is saved only when the registration succeeds.
                    #-------------------------------------------------
                    # Find all post cropped files. This should give back at least 3 files, the scan, regmask and seg. There may be multiple seg files.   
                    followup_post_files = get_files(subject_folder, 'post_' + '*' + '*cropped*', '.nrrd') # Note that the function adds additional wild cards between the inputs
                    # Get the followup post scan file
                    #---------------------------------
                    followup_post_scan_file = []
                    for followup_post_file in followup_post_files:
                        if ('seg' in os.path.basename(followup_post_file)) or ('regmask' in os.path.basename(followup_post_file)):
                            continue
                        followup_post_scan_file.append(followup_post_file)                     
                    # There should be only 1 followup post scan file. Check and store it. If not found, the registration must have 
                    # failed. Store it as an invalid file and continue to the next case.  
                    if len(followup_post_scan_file) != 1:
                        invalid_cases_list.append(case_name)
                        continue
                    # Followup post scan file
                    followup_post_scan_file = followup_post_scan_file[0]                    
                    # Get the follow up post segmentation file
                    #--------------------------------------------
                    # Find the followup post segmentation file. This could give us multiple files since the segmentation files are 
                    # saved at the beginning of every registration attempt (i.e., with decreasing mask sizes).    
                    followup_post_seg_files = get_files(subject_folder, 'post_' + '*' + '*seg*' + '*cropped*', '.nrrd') # Note that the function adds additional wild cards between the inputs
                    if not followup_post_seg_files:
                        invalid_cases_list.append(case_name)
                        continue
                    # Find the segmentation file with the smallest crop radius. This is the one for which the registration succeeded.
                    # Sort in increasing order and take the first seg file.
                    followup_post_seg_files.sort()
                    followup_post_seg_file = followup_post_seg_files[0]       
                    
                    # Check valid cropped scan and seg files were found. If not, count this case as failure and continue to the next one.
                    if (not followup_post_scan_file) or (not followup_post_seg_file):
                        invalid_cases_list.append(case_name)
                        continue

                    # Make sure the mask radius is the same for both pre and post files
                    # For pre files, we need to use regex to extract crop radius
                    crop_pattern = 'imgCropped(.+?)Both'
                    pre_scan_file_crop_radius = re.search(crop_pattern, os.path.basename(pre_scan_file)).group(1)
                    assert os.path.splitext(os.path.basename(followup_post_scan_file))[0].rsplit('_', 1)[1] \
                                == os.path.splitext(os.path.basename(followup_post_seg_file))[0].rsplit('_', 1)[1] \
                                    == pre_scan_file_crop_radius \
                                        == os.path.splitext(os.path.basename(pre_seg_file))[0].rsplit('_', 1)[1] 

                    #--------------------------------------------------------------------------------------------------------
                    # All the tests above was to make sure the case is valid. Now, we can read the transform and pass the 
                    # pre segmentation through the transform:            
                    # Read back the saved transforms, setup transformix, apply transform 
                    # to image and sved the results. 
                    try:
                        # Iterate over all the param files and read them back
                        # Find all the pre to post transform parameter files 
                        pre_to_post_trans_param_files = get_files(subject_folder, 'pre_in_post' + '*' + 'transParam', '.txt') # Note that the function adds additional wild cards between the inputs
                        pre_to_post_trans_param_files.sort() # This will work since the transform parameter files for various transforms only differ by the transform number at the end and they are in the order in which transforms were added to the registration. 
                        # Iterate through the list of files and read the transform parameters in 
                        transformParamMapReadFromFiles = []
                        for transformParamFile in pre_to_post_trans_param_files:
                            # Read transform file
                            transformParam = sitk.ReadParameterFile(transformParamFile)
                            # 1) Change the interpolation to nearest neighbour since we are pulling
                            # a segmentation through the transform. It is otherwise set to
                            # 'FinalBSplineInterpolator' for the registration.
                            # See:
                            # https://github.com/SuperElastix/SimpleElastix/issues/409
                            # https://github.com/SuperElastix/SimpleElastix/issues/255
                            transformParam["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
                            # 2) Set final bspline interpolation order to 0 (I think this is
                            # redundant since we are already doing nearest neighbour interpolation,
                            # and this was supposed help produce a binary image.)
                            # See: https://github.com/SuperElastix/SimpleElastix/issues/419 
                            transformParam["FinalBSplineInterpolationOrder"] = ["0"]
                            # Save transform
                            transformParamMapReadFromFiles.append(transformParam)
                        # Apply transform to moving image, in this case binary label of tumor in pre.
                        # Setup transformix
                        transformixImageFilter = sitk.TransformixImageFilter()
                        transformixImageFilter.SetTransformParameterMap(tuple(transformParamMapReadFromFiles))
                        transformixImageFilter.SetMovingImage(sitk.ReadImage(pre_seg_file))
                        transformixImageFilter.Execute()
                        # Create output file names and save the transformed segmentation file
                        # Save to local storage
                        #output_file_name = 'preseg' + os.path.basename(pre_scan_file)[3:] 
                        output_file_name = 'presegbinary' + os.path.basename(pre_scan_file)[3:] 
                        output_file_path = os.path.join(subject_folder, output_file_name)
                        sitk.WriteImage(transformixImageFilter.GetResultImage(), output_file_path)

                    except Exception as error_output:
                        # Print and store the exception message for debugging. There is no code for python exceptions unless explicitly
                        # created that way. Store the exception info.
                        error_msg = 'Applying reg. transform on segmentation failed for case: {a}! ErrorName: {c}, Message: {m}'.format(a = case_name, c = type(error_output).__name__, m = str(error_output))
                        print(error_msg)
                        failed_cases_error_msg.append(error_msg)

