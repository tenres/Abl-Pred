# Novel deformable registration methodology optimized for ablation workflows. 
# --------------------------------------------------------------------------------------------------------------- 
# This code registers scans in the ablation workflow. E.g., the pre scans to the followup scan. 
# We use the Simple Elastix image registration library 
# This reads all necessary data including
# applicator power and time information, pre and followup scans, and the followup post ablation segmentation.
# The pre and followup excel sheets are read for the ablation treatment information and the scans are copied from the "path_to_ablation_folder" folder to 
# the output folder. we don't register the full images; we start with 100mm cropped images. 
#

# Imports
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob

def compute_antenna_tail_coords(antenna_tip_coords=[], antenna_back_coords=[], dist_from_tip=[]):
    '''
    This function takes in the antenna tip co-ordinates, antenna back co-ordinates and distance from
    tip (along the antenna), all in mm, and returns tha tail point. Note that all computation assumes
    LPS co-ordinates. 
    '''
    antenna_tip_to_tail_vec = np.array(antenna_back_coords) - np.array(antenna_tip_coords)
    antenna_tip_to_tail_unit_vec =  antenna_tip_to_tail_vec / np.linalg.norm(antenna_tip_to_tail_vec)
    antenna_tail_coords =  np.array(antenna_tip_coords) + dist_from_tip * antenna_tip_to_tail_unit_vec
    return antenna_tail_coords[0], antenna_tail_coords[1], antenna_tail_coords[2] 


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


def get_segment_center(seg_img_file):
    '''Compute the center of a binary segmentations in the world space'''

    # Debug
    #seg_img_file = followup_post_seg_file

    # Read image using SimpleITK 
    input_image = sitk.ReadImage(seg_img_file)
    
    # # Method 1: Convert all grid indices to physical space and then take the average.
    # # This seems a little slow. Better to first compute the average in the grid space 
    # # and then transform it to the physical space.  
    # #------------------------------------------------------------------------------------
    # # Get the size of the image
    # img_size = input_image.GetSize()
    # # Average co-ordinates
    # x_coords = []
    # y_coords = []
    # z_coords = []
    # # Check if the point is within the sphere with radius radius and center center
    # for i in range(img_size[0]):
    #     for j in range(img_size[1]):
    #         for k in range(img_size[2]):
    #             # Check if the index position is inside the segmentation and if yes, add the physical co-ordinates to the array
    #             if input_image[i,j,k] != 0:
    #                 # Get physical co-ordinates of index i, j, k
    #                 x, y, z = input_image.TransformIndexToPhysicalPoint((i,j,k))
    #                 # Store the x, y, z
    #                 x_coords.append(x)
    #                 y_coords.append(y)
    #                 z_coords.append(z)

    # # Return the centroid of the ablation zone
    # return sum(x_coords)/float(len(x_coords)), sum(y_coords)/float(len(y_coords)), sum(z_coords)/float(len(z_coords)) 


    # # Method 2: First compute the average in the grid space co-ordinates 
    # # and then transform it to the physical space.  
    # #------------------------------------------------------------------------------------
    # # Get the size of the image
    # img_size = input_image.GetSize()
    # # Average co-ordinates
    # i_seg_coords = []
    # j_seg_coords = []
    # k_seg_coords = []
    # # Check if the point is within the sphere with radius radius and center center
    # for i in range(img_size[0]):
    #     for j in range(img_size[1]):
    #         for k in range(img_size[2]):
    #             # Check if the index position is inside the segmentation and if yes, add the grid co-ordinates to the array
    #             if input_image[i,j,k] != 0:
    #                 # Store the i, j, k grid co-ordinates
    #                 i_seg_coords.append(i)
    #                 j_seg_coords.append(j)
    #                 k_seg_coords.append(k)

    # # Compute mean i,j,k grid co-ordinates, the center of the segmentation in grid space
    # i_seg_coords_mean, j_seg_coords_mean, k_seg_coords_mean = sum(i_seg_coords)/float(len(i_seg_coords)), sum(j_seg_coords)/float(len(j_seg_coords)), sum(k_seg_coords)/float(len(k_seg_coords))
    # # Get physical co-ordinates of the mean i, j, k
    # #x_mean, y_mean, z_mean = input_image.TransformIndexToPhysicalPoint((i_seg_coords_mean, j_seg_coords_mean, k_seg_coords_mean)) #Note that grid indices have to be integers, else an error is thrown.
    # x_mean, y_mean, z_mean = input_image.TransformIndexToPhysicalPoint((round(i_seg_coords_mean), round(j_seg_coords_mean), round(k_seg_coords_mean)))

    # # Return the centroid of the ablation zone in the physical space
    # return x_mean, y_mean, z_mean 


    # Method 3: First compute the average in the grid space co-ordinates using np.nonzero. 
    # This increases the speed manifold. Then transform it to the physical space.  
    #---------------------------------------------------------------------------------------
    # Get the pixel value as a numpy array
    pixel_array = sitk.GetArrayFromImage(input_image)
    # Swap the first and last axis since numpy array index has these two axis swapped compared to the original image (ITK stores in the same order).
    pixel_array = np.swapaxes(pixel_array, 0, 2) 
    # Get non zero voxel indices (i.e., those of the segmentation)
    pixel_array_nonzero_ind = np.nonzero(pixel_array)
    # Get means of each axis of the non-zero indices 
    i_seg_coords_mean = np.mean(pixel_array_nonzero_ind[0])
    j_seg_coords_mean = np.mean(pixel_array_nonzero_ind[1])
    k_seg_coords_mean = np.mean(pixel_array_nonzero_ind[2])
    # Convert the mean non-zero index in grid space to physical space
    x_seg_coords_mean, y_seg_coords_mean, z_seg_coords_mean = input_image.TransformIndexToPhysicalPoint((round(i_seg_coords_mean), round(j_seg_coords_mean), round(k_seg_coords_mean)))

    return x_seg_coords_mean, y_seg_coords_mean, z_seg_coords_mean


def create_mask_img(image_file, center=(0,0,0), shape='sphere', radius=1, subtraction_mask_file=[]):
    '''
    This function creates a binary image file (an ITK image type) in the same space as the input_image (an ITK image type)
    with 1s inside the specified shape around center and 0s outside it
    '''

    # Read image
    input_image = sitk.ReadImage(image_file)

    # Get the size of the image
    img_size = input_image.GetSize()

    # Create a numpy array the size of the original image
    mask_array = np.zeros(img_size, dtype=np.uint8)

    # Loop over the image index co-ordinates, convert them to physical co-ordinates and check if they are within the shape
    if shape == 'sphere':
        # Check if the point is within the sphere with radius radius and center center
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                for k in range(img_size[2]):
                    # Get physical co-ordinates of index i, j, k
                    x, y, z = input_image.TransformIndexToPhysicalPoint((i,j,k))
                    # Check if the physical co-ordinates are inside the sphere and set the label to 1 if so
                    if (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 <= radius**2:
                        mask_array[i,j,k] = 1
        
        # Check if there is a subtraction mask and account for it if present
        # Check subtraction mask and prepare it
        if subtraction_mask_file:
            # Read the subtraction mask image
            subtraction_mask_image = sitk.ReadImage(subtraction_mask_file)
            # Convert the subtraction mask to a binary mask
            subtraction_mask_array = (sitk.GetArrayFromImage(subtraction_mask_image)).astype(np.uint8)
            subtraction_mask_array = np.where(subtraction_mask_array != 0, 1, 0)
            # Subtract
            print(subtraction_mask_array.shape) 
            print(mask_array.shape) 
            mask_array = mask_array - subtraction_mask_array.T
            # Sometimes the subtraction mask is bigger than the shape mask. In such case, the 
            # subtraction mask pixels go beyond the shape and hence the subtractino will result 
            # in -1. When this is case to sitk.sitkUInt8, it is converted to 255, and hence 
            # changing the mask values. This was observed in at least 10 cases, whose post reg masks
            # were incorrect, resulting in incorrect registrations. A simple solution is just to
            # set all intensities < 0 to 0.
            mask_array[mask_array < 0] = 0 # np.where doesn't provide functionality to alter only the pixels that satisfy the condition. It requires user to specify the value to set pixels when the condition is not satisifed as well, and not just let it be as is.

    # Construct an ITK image from the numpy mask array and write as output
    mask_img = sitk.GetImageFromArray(mask_array.T)
    mask_img.CopyInformation(input_image)

    return sitk.Cast(mask_img, sitk.sitkUInt8)



def create_mask_img_input_img(input_image, center=(0,0,0), shape='sphere', radius=1, subtraction_mask_image=[]):
    '''
    This function creates a binary image file (an ITK image type) in the same space as the input_image (an ITK image type)
    with 1s inside the specified shape around center and 0s outside it
    '''

    ## Read image
    #input_image = sitk.ReadImage(image_file)

    # Get the size of the image
    img_size = input_image.GetSize()

    # Create a numpy array the size of the original image
    mask_array = np.zeros(img_size, dtype=np.uint8)

    # Loop over the image index co-ordinates, convert them to physical co-ordinates and check if they are within the shape
    if shape == 'sphere':
        # Check if the point is within the sphere with radius radius and center center
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                for k in range(img_size[2]):
                    # Get physical co-ordinates of index i, j, k
                    x, y, z = input_image.TransformIndexToPhysicalPoint((i,j,k))
                    # Check if the physical co-ordinates are inside the sphere and set the label to 1 if so
                    if (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 <= radius**2:
                        mask_array[i,j,k] = 1
        
        # Check if there is a subtraction mask and account for it if present
        # Check subtraction mask and prepare it
        if subtraction_mask_image:
            ## Read the subtraction mask image
            #subtraction_mask_image = sitk.ReadImage(subtraction_mask_file)
            # Convert the subtraction mask to a binary mask
            subtraction_mask_array = (sitk.GetArrayFromImage(subtraction_mask_image)).astype(np.uint8)
            subtraction_mask_array = np.where(subtraction_mask_array != 0, 1, 0)
            # Subtract
            print(subtraction_mask_array.shape) 
            print(mask_array.shape) 
            mask_array = mask_array - subtraction_mask_array.T
            # Sometimes the subtraction mask is bigger than the shape mask. In such case, the 
            # subtraction mask pixels go beyond the shape and hence the subtractino will result 
            # in -1. When this is case to sitk.sitkUInt8, it is converted to 255, and hence 
            # changing the mask values. This was observed in at least 10 cases, whose post reg masks
            # were incorrect, resulting in incorrect registrations. A simple solution is just to
            # set all intensities < 0 to 0.
            mask_array[mask_array < 0] = 0 # np.where doesn't provide functionality to alter only the pixels that satisfy the condition. It requires user to specify the value to set pixels even when the condition is not satisifed as well, and not just let it be as is.


    # Construct an ITK image from the numpy mask array and write as output
    mask_img = sitk.GetImageFromArray(mask_array.T)
    mask_img.CopyInformation(input_image)

    print(mask_img.GetDirection())
    print(mask_img.GetSpacing())
    print(mask_img.GetOrigin())

    return sitk.Cast(mask_img, sitk.sitkUInt8)
    #sitk.WriteImage(mask_img, mask_file)



def create_coords_pts_file(point_coords=[], pts_full_file_name=[]):
    '''
    Function takes in the co-ordinates of points as a list of tuples and create a '.pts' file pts_full_file_name
    '''

    # Check the input
    if (not point_coords) or (not pts_full_file_name):  
        raise ValueError('Invalid input. Exiting...')
        
    # The points file gets overwrite each time. Be careful!
    with open(pts_full_file_name, 'w') as pts_file:
        pts_file.write('point\n')
        pts_file.write('{}\n'.format(len(point_coords)))
        # Iterate over al points and write them to file
        for point in point_coords:
            pts_file.write('{} {} {}\n'.format(point[0], point[1], point[2]))


def pre_to_post_reg_error_sol_crop_img(followup_post_scan_file = [], pre_scan_file = [],
        followup_post_abl_center_lps = [], pre_tumor_center_lps=[],
            fixed_image=[], moving_image=[], output_case_folder_path=[],
                is_pre_matching_coords_valid=[], is_followup_post_matching_coords_valid=[],
                    fixed_img_matching_coords_file=[], moving_img_matching_coords_file=[], 
                        fixed_img_abl_center_coord_file=[], moving_img_tumor_center_coord_file=[],
                            rigidMap=[], bsplineMap=[], crop_radius=[], requiredRatioOfValidSamples = [], followup_post_seg_file = [], pre_seg_file = [], use_pre_seg_rigidity_mask=True, use_followup_seg_rigidity_mask=False, affineMap=[], use_affine_map=False, reg_mask_radius=[]):

    # Debug
    #crop_radius = 70

    # Create binary masks to crop moving and fixed images
    #------------------------------------------------------
    fixed_image_crop_mask = create_mask_img(followup_post_scan_file, center = followup_post_abl_center_lps, shape='sphere', radius=crop_radius)
    moving_image_crop_mask = create_mask_img(pre_scan_file, center = pre_tumor_center_lps, shape='sphere', radius=crop_radius)
    # # Debug
    # followup_post_crop_mask_file = os.path.join(output_case_folder_path, os.path.splitext(os.path.basename(followup_post_seg_file))[0] + '_cropMask_' + str(crop_radius) + '.nrrd')
    # sitk.WriteImage(fixed_image_crop_mask, followup_post_crop_mask_file)
    # pre_crop_mask_file = os.path.join(output_case_folder_path, os.path.splitext(os.path.basename(pre_seg_file))[0] + '_cropMask_' + str(crop_radius) + '.nrrd')
    # sitk.WriteImage(moving_image_crop_mask, pre_crop_mask_file)

    # Crop the scans using the masks
    #-----------------------------------
    fixed_image_cropped = sitk.LabelMapMask(sitk.LabelImageToLabelMap(fixed_image_crop_mask), fixed_image, crop=True)
    moving_image_cropped = sitk.LabelMapMask(sitk.LabelImageToLabelMap(moving_image_crop_mask), moving_image, crop=True)

    # Crop the segmentations as well
    #-----------------------------------
    # Post
    followup_post_seg_image = sitk.ReadImage(followup_post_seg_file)
    followup_post_seg_image_cropped = sitk.LabelMapMask(sitk.LabelImageToLabelMap(fixed_image_crop_mask), followup_post_seg_image, crop=True)
    # Save cropped followup segmentation mask. Note that we save this since it may be useful to set the followup post registration mask. 
    followup_post_seg_image_cropped_file = os.path.join(output_case_folder_path, os.path.splitext(os.path.basename(followup_post_seg_file))[0] + '_cropped_' + str(crop_radius) + '.nrrd')
    sitk.WriteImage(followup_post_seg_image_cropped, followup_post_seg_image_cropped_file)
    # Pre
    pre_seg_image = sitk.ReadImage(pre_seg_file)
    pre_seg_image_cropped = sitk.LabelMapMask(sitk.LabelImageToLabelMap(moving_image_crop_mask), pre_seg_image, crop=True)
    # Save cropped pre segmentation mask as file since it could be useful for setting the pre rigidity mask and registration msk
    # during registration. 
    pre_seg_image_cropped_file = os.path.join(output_case_folder_path, os.path.splitext(os.path.basename(pre_seg_file))[0] + '_cropped_' + str(crop_radius) + '.nrrd')
    sitk.WriteImage(pre_seg_image_cropped, pre_seg_image_cropped_file)


    # Create and crop rigidity mask based on the new cropped images
    #----------------------------------------------------------------
    # Create a rigidity mask for the pre images. 
    # Two options: 2) Use the tumor segmentation itself or 1) a 20mm rigid sphere. 
    # Pre/moving
    #--------------
    if use_pre_seg_rigidity_mask:
        # Option 2
        moving_image_rigidity_mask_file = pre_seg_image_cropped_file # This is required to later set the registration parameters
    else:
        # Option 1
        pre_rigidity_mask_20 = create_mask_img_input_img(moving_image_cropped, center = pre_tumor_center_lps, shape='sphere', radius=10)
        # The directional cosines, spacing and origin differed very very slightly from moving_image_cropped for some reason even when we used CopyInformation. So, setting it explicitly here.
        pre_rigidity_mask_20.SetDirection(moving_image_cropped.GetDirection())
        pre_rigidity_mask_20.SetSpacing(moving_image_cropped.GetSpacing())
        pre_rigidity_mask_20.SetOrigin(moving_image_cropped.GetOrigin())
        # Save to file, required for registration
        pre_rigidity_mask_20_file = os.path.join(output_case_folder_path, 'pre_rigidity_mask_10_file_cropped_' + str(crop_radius) + '.nrrd')
        sitk.WriteImage(sitk.Cast(pre_rigidity_mask_20, sitk.sitkUInt8), pre_rigidity_mask_20_file)
        moving_image_rigidity_mask_file = pre_rigidity_mask_20_file # This is required to later set the registration parameters

    # Followup post
    #----------------------------------------------------------
    # Based on the elastix documentation advice, this should not be on, i.e. the followup post rigidity is not advised.
    if use_followup_seg_rigidity_mask:
        fixed_image_rigidity_mask_file = followup_post_seg_image_cropped_file # This is required to later set the registration parameters


    # Create new 50 mm masks to focus the registration. These are computed from the cropped images.
    # We did not include the tumor and ablation zone for registration since they do not need to match. 
    #-----------------------------------------------------------------------------------------------------
    # Fixed
    #fixed_image_mask = create_mask_img_input_img(fixed_image_cropped, center = followup_post_abl_center_lps, shape='sphere', radius=50)
    fixed_image_mask = create_mask_img_input_img(fixed_image_cropped, center = followup_post_abl_center_lps, shape='sphere', radius=reg_mask_radius, subtraction_mask_image = followup_post_seg_image_cropped)
    # The directional cosines, spacing and origin differed very very slightly from fixed_image_cropped for some reason even when used CopyInformation. So, setting it explicitly
    fixed_image_mask.SetDirection(fixed_image_cropped.GetDirection())
    fixed_image_mask.SetSpacing(fixed_image_cropped.GetSpacing())
    fixed_image_mask.SetOrigin(fixed_image_cropped.GetOrigin())
    # Moving
    #moving_image_mask = create_mask_img_input_img(moving_image_cropped, center = pre_tumor_center_lps, shape='sphere', radius=50)
    moving_image_mask = create_mask_img_input_img(moving_image_cropped, center = pre_tumor_center_lps, shape='sphere', radius=reg_mask_radius, subtraction_mask_image = pre_seg_image_cropped)
    # The directional cosines differed from movingImage_cropped for some reason even when used CopyInformation. So, setting it explicitly
    moving_image_mask.SetDirection(moving_image_cropped.GetDirection())
    moving_image_mask.SetSpacing(moving_image_cropped.GetSpacing())
    moving_image_mask.SetOrigin(moving_image_cropped.GetOrigin())
    # # Debug
    # # Test if they look ok
    # fixed_image_mask_cropped_file = os.path.join(output_case_folder_path, 'fixed_image_reg_mask_cropped' + '.nrrd')
    # sitk.WriteImage(sitk.Cast(fixed_image_mask, sitk.sitkUInt8), fixed_image_mask_cropped_file)
    # moving_image_mask_cropped_file = os.path.join(output_case_folder_path, 'moving_image_reg_mask_cropped' + '.nrrd')
    # sitk.WriteImage(sitk.Cast(moving_image_mask, sitk.sitkUInt8), moving_image_mask_cropped_file)



    # Setup registration parameters
    #--------------------------------
    elastixImageFilter = sitk.ElastixImageFilter()
    # Set fixed and moving images
    elastixImageFilter.SetFixedImage(fixed_image_cropped)
    elastixImageFilter.SetMovingImage(moving_image_cropped)
    # Set fixed and moving image masks
    elastixImageFilter.SetFixedMask(sitk.Cast(fixed_image_mask, sitk.sitkUInt8))
    elastixImageFilter.SetMovingMask(sitk.Cast(moving_image_mask, sitk.sitkUInt8))
    # Set fixed and moving image matching point sets
    # Check if there are additional points that need to be matched for both images, and if so, use it for matching, 
    # and add the appropriate setup
    # NOTE: Though we had already set these in the main function, we are re-initializing the elasticImageFilter here since we
    # are setting new cropped images. Hence we have to re-set the matching points too. 
    if is_pre_matching_coords_valid and is_followup_post_matching_coords_valid:
        elastixImageFilter.SetFixedPointSetFileName(fixed_img_matching_coords_file)
        elastixImageFilter.SetMovingPointSetFileName(moving_img_matching_coords_file)
    else:
        # If extra points not present for both both, set to match the ablation centers as an initial rigid pre step. 
        # This will be used only for initialization and not for non-rigid registration.
        elastixImageFilter.SetFixedPointSetFileName(fixed_img_abl_center_coord_file)
        elastixImageFilter.SetMovingPointSetFileName(moving_img_tumor_center_coord_file)
    # Edit the transform and create a new parameterMapVector: rigidity mask, automatic initialization and maximum number of attempts (only if relevant)
    rigidMap["AutomaticTransformInitialization"] = ("true",) # When the images are cropped around the ablation center and tumor, automatic intialization, which is just aligning the means, should work well.
    rigidMap["MaximumNumberOfSamplingAttempts"] = ("10",) # Attemps to draw sampled 10 times instead of failing when sufficient samples are not found in the first attempt.
    # Affine commented out since it was not helping, rather spoiling
    if use_affine_map and bool(affineMap):
        affineMap["AutomaticTransformInitialization"] = ("true",) # When the images are cropped around the ablation center and tumor, automatic intialization, which is just aligning the means, should work well.
        affineMap["MaximumNumberOfSamplingAttempts"] = ("10",) # Attemps to draw sampled 10 times instead of failing when sufficient samples are not found in the first attempt.
    bsplineMap["AutomaticTransformInitialization"] = ("true",)
    bsplineMap["MaximumNumberOfSamplingAttempts"] = ("10",)
    bsplineMap["MovingRigidityImageName"] = (moving_image_rigidity_mask_file,) # Specify the rigidity mask
    # Based on the elastix documentation advice, this should not be on, i.e. the followup post rigidity is not advised.
    if use_followup_seg_rigidity_mask:
        bsplineMap["FixedRigidityImageName"] = (fixed_image_rigidity_mask_file,) # Specify the rigidity mask
    if bool(requiredRatioOfValidSamples):
        # The default is 0.25. This is what has been seen to work well by elastix authors. Hence
        # it is good to set this to default for pre to post registration. 
        # We may have had to set it to slightly lower values like 0.10 in some cases, but let it be not the default
        # here.         
        rigidMap["RequiredRatioOfValidSamples"] = (requiredRatioOfValidSamples,)
        bsplineMap["RequiredRatioOfValidSamples"] = (requiredRatioOfValidSamples,)

    # Combine all transforms
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(rigidMap)
    if use_affine_map and bool(affineMap):
        parameterMapVector.append(affineMap)
    parameterMapVector.append(bsplineMap)
    # Set transform
    elastixImageFilter.SetParameterMap(parameterMapVector)
    # Execute registration
    # # Debug
    # elastixImageFilter.LogToFileOn()
    # elastixImageFilter.SetOutputDirectory(subject_folder)
    elastixImageFilter.Execute()

    # If successful, save all relevant images and registration parameters.
    # Save registered image
    if is_pre_matching_coords_valid and is_followup_post_matching_coords_valid:
        # Note that the loss function weighs are only for the bspline/deformable registration part.
        #output_file_name = 'pre_in_post_nonRigid_imgCropped' + str(crop_radius) + 'Both_mask50Both_ablRigidPenaltyPre_correspondingPointsMatching_weights1111.nrrd'
        output_file_name = 'pre_in_post_nonRigid_imgCropped' + str(crop_radius) + 'Both_mask' + str(reg_mask_radius) + 'Both_ablRigidPenaltyPre_correspondingPointsMatching_weights1111.nrrd'
    else:
        #output_file_name = 'pre_in_post_nonRigid_imgCropped' + str(crop_radius) + 'Both_mask50Both_ablRigidPenaltyPre_centerMatchInitialOnly_weights111.nrrd'
        output_file_name = 'pre_in_post_nonRigid_imgCropped' + str(crop_radius) + 'Both_mask' + str(reg_mask_radius) + 'Both_ablRigidPenaltyPre_centerMatchInitialOnly_weights111.nrrd'

    output_file = os.path.join(output_case_folder_path, output_file_name)
    sitk.WriteImage(elastixImageFilter.GetResultImage(), output_file)
    # Save cropped images            
    fixed_image_cropped_file = os.path.join(output_case_folder_path, os.path.splitext(os.path.basename(followup_post_scan_file))[0] + '_cropped_' + str(crop_radius) + '.nrrd')
    sitk.WriteImage(fixed_image_cropped, fixed_image_cropped_file)
    moving_image_cropped_file = os.path.join(output_case_folder_path, os.path.splitext(os.path.basename(pre_scan_file))[0] + '_cropped_' + str(crop_radius) + '.nrrd')
    sitk.WriteImage(moving_image_cropped, moving_image_cropped_file)
    # Save the fixed and moving image masks
    fixed_image_mask_file = os.path.join(output_case_folder_path, os.path.splitext(os.path.basename(followup_post_scan_file))[0] + '_cropped_' + str(crop_radius) + '_regmask_' + str(reg_mask_radius) + '.nrrd')
    sitk.WriteImage(sitk.Cast(fixed_image_mask, sitk.sitkUInt8), fixed_image_mask_file)
    moving_image_mask_file = os.path.join(output_case_folder_path, os.path.splitext(os.path.basename(pre_scan_file))[0] + '_cropped_' + str(crop_radius) + '_regmask_' + str(reg_mask_radius) + '.nrrd')
    sitk.WriteImage(sitk.Cast(moving_image_mask, sitk.sitkUInt8), moving_image_mask_file)


    # Iterate over all the transform params and write them to files
    # Example:
    # transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    # WriteParameterFile(transformationMap, filename)' to save it to disk and 'ReadParameterFile(filename)' to read them back
    # Extract the transform parameters
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    # Note that we obtain a list of all the transforms parameter maps in the order in which they were applied. e.g. transformParameterMap[0] = initial rigid transform
    # transformParameterMap[1] = non-rigid bspline transform. Iterate over all the params and write them to separate files. These would have to be read in the same
    # order to reconstruct the transform back. An example is provided below. 
    param_index = 0
    for transformParam in transformParameterMap:
        # Create names for the transforms that will be saved
        if is_pre_matching_coords_valid and is_followup_post_matching_coords_valid:
            # If matching points present in both scans
            # Note that the loss function weighs are only for the bspline/deformable registration part.
            output_transform_file_name = 'pre_in_post_nonRigid_imgCropped' + str(crop_radius) + 'Both_mask' + str(reg_mask_radius) + 'Both_ablRigidPenaltyPre_correspondingPointsMatching_weights1111_transParam' + str(param_index) + '.txt'
        else:
            # If only matching the centers, which is used only during the initial rigid/affine transform
            output_transform_file_name = 'pre_in_post_nonRigid_imgCropped' + str(crop_radius) + 'Both_mask' + str(reg_mask_radius) + 'Both_ablRigidPenaltyPre_centerMatchInitialOnly_weights111_transParam' + str(param_index) + '.txt'
        # Output file
        output_file = os.path.join(output_case_folder_path, output_transform_file_name)
        # Save transform
        sitk.WriteParameterFile(transformParam, output_file)  
        # Increase param index
        param_index = param_index + 1



if __name__ == '__main__':


    # Set paths and initialize variables
    #----------------------------------------------------------------------------------------------
    # Input data folder
    data_base_folder = r'/home/path_to_ablation_data'
    # Registered output data folder
    registered_data_base_folder = r'/home/path_to_ablation_data/registration_output'
    # Read in the ablation information dictionary. 
    # There was an issue with allow_pickle set to false by default in the later versions. Check here: https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa
    # Work around:
    #---------------------
    # save current np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # Load ablation information 
    ablation_info_all = np.load(os.path.join(data_base_folder, 'pre_post_ablation_info_all.npz'))
    # Restore np.load for future normal usage
    np.load = np_load_old 
    #---------------------
    # Get the ablation info dictionary. Note that we need the [()] at the end since numpy saves the dictionary as a numpy array. Check here: https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa
    ablation_info_all = ablation_info_all['ablation_info_all'][()]
    # # Debug
    # for key, value in ablation_info_all.items():
    #     print(key)


    # Read all the cases from the data folder and register them one by one.
    #-------------------------------------------------------------------------------------------------------------------------------------
    # Cases that required various solutions of registration. Note that the observations after running cross all cases are noted on the right.
    succeeded_first_attempt = []    
    succeeded_second_attempt = []   
    succeeded_third_attempt = []    
    succeeded_fourth_attempt = []   
    succeeded_fifth_attempt = []    
    succeeded_sixth_attempt = []      
    registation_failed_cases = []   
    imageread_failed_cases = []         
    # Keep a count of cases
    processed_case_count = 0
    # Iterate through all the cases
    for subject_folder in glob.glob(os.path.join(data_base_folder, "*")):

        # Assuming all directories in this folder are valid case directories
        if os.path.isdir(subject_folder):

            # Get name of the case
            case_name = os.path.basename(subject_folder)
            processed_case_count += 1 
            print('Processing case number {}: {}'.format(processed_case_count, case_name))


            # Find the pre files (scan + segmentation file)
            #---------------------------------------------------
            pre_files = get_files(subject_folder, 'pre_' + '*' + case_name, '.nrrd') # Note that the function adds additional wild cards between the inputs
            # If the size of the list is not 2, try the base name
            if len(pre_files) != 2:
                # Some times the sub case name like a, b, c is not there in the scan file name. Hence try again with the last character removed.
                base_case_name = case_name[:-1] 
                pre_files = get_files(subject_folder, 'pre_' + '*' + base_case_name, '.nrrd') # Note that the function adds additional wild cards between the inputs
            # Check again now, and if no 2 this time, raise exception
            if len(pre_files) != 2:
                raise ValueError('Pre files not found. Exiting...')
            else:    
                # If exactly two pre files found, find which one is the scan and which one the segmentation label.
                if 'seg' in os.path.basename(pre_files[0]).lower():
                    pre_seg_file = pre_files[0]
                    pre_scan_file = pre_files[1]
                elif 'seg' in os.path.basename(pre_files[1]).lower():
                    pre_seg_file = pre_files[1]
                    pre_scan_file = pre_files[0]
                else:
                    # Segmentation file not found. This is an incomplete case that cannot be used. 
                    # TODO This doesn't catch the case where the segmentation is a valid file, but the scan is not. The scan is assumed to be valid for now and this needs to be fixed.
                    raise ValueError('Pre files not found. Exiting...')


            # Find the followup post files (scan + segmentation file)
            #----------------------------------------------------------
            followup_post_files = get_files(subject_folder, 'post_' + '*' + case_name, '.nrrd') # Note that the function adds additional wild cards between the inputs
            # If the size of the list is not 2, try the base name
            if len(followup_post_files) != 2:
                # Some times the sub case name like a, b, c is not there in the scan file name. Hence try again with the last character remvoed.
                base_case_name = case_name[:-1] 
                followup_post_files = get_files(subject_folder, 'post_' + '*' + base_case_name, '.nrrd') # Note that the function adds additional wild cards between the inputs
            # Check again now, and if no 2 this time, raise exception
            if len(followup_post_files) != 2:
                raise ValueError('Followup post files not found. Exiting...')
            else:    
                # If exactly two post files found, find which one is the scan and which one the segmentation label.
                if 'seg' in os.path.basename(followup_post_files[0]).lower():
                    followup_post_seg_file = followup_post_files[0]
                    followup_post_scan_file = followup_post_files[1]
                elif 'seg' in os.path.basename(followup_post_files[1]).lower():
                    followup_post_seg_file = followup_post_files[1]
                    followup_post_scan_file = followup_post_files[0]
                else:
                    # Segmentation file not found. This is an incomplete case that cannot be used. 
                    # TODO This doesn't catch the case where the segmentation is a valid file, but the scan is not. The scan is assumed to be valid for now and this needs to be fixed.
                    raise ValueError('Followup post files not found. Exiting...')


            # Read ablation information of the case from the ablation dictionary
            ablation_info = ablation_info_all[case_name]

            # Create subject directory in the registered folder. We assume that the case is valid and all the necessary ablation information exists.
            output_case_folder_path = os.path.join(registered_data_base_folder, case_name)
            if not os.path.exists(output_case_folder_path):
                os.makedirs(output_case_folder_path)
            # THIS WAS COMMENTED OUT TEMPORARILY FOR PROCESSING SPECIFIC CASE THAT HAD ERRORED DURING REGISTRATION AND MISSING MASKS!!!.
            # WE WANT TO PROCESS THESE CASES AGAIN, EVEN THOUGH THE FOLDERS WERE ALREADY CREATED.
            # UNCOMMENTED AFTER REGISTRATION
            else:
                # Assume that the case had already been processed. Move to next case.
                # TODO: We assume that if the folder exists, the case is already registered and move on to the next one. So, the folder needs to be deleted if the case needs to be processed again.
                continue

            # Copy all the original image and segmentation files over to the registered folder. This is just for completeness/independence of the registered folder
            # from the original folder. 
            # Pre
            shutil.copyfile(pre_scan_file, os.path.join(output_case_folder_path, os.path.basename(pre_scan_file)))
            shutil.copyfile(pre_seg_file, os.path.join(output_case_folder_path, os.path.basename(pre_seg_file)))
            # Post
            shutil.copyfile(followup_post_scan_file, os.path.join(output_case_folder_path, os.path.basename(followup_post_scan_file)))
            shutil.copyfile(followup_post_seg_file, os.path.join(output_case_folder_path, os.path.basename(followup_post_seg_file)))

            print('Found fixed and moving image files. Starting registration...')

            # Pre to post ablation registration:
            #-----------------------------------------------------------------
            # Fixed image = post scan; Moving image = Pre scan
            # Registration steps:
            #----------------------------
            # Latest modifications to the steps:
            # 1. Rigidity constraint: Having the tumor segmentation itself as the rigidity constraint is useful 
            #    for registration. Having 20mm sphereimposes rigidity on a large 
            #    area surrounding the tumor and is not ideal for pre. The tumors are usually much smaller. 
            #    We want everything outside the tumor area to match to the post if the region 
            #    exists. Hence maybe it is better to use only the tumor seg as rigidity mask and leave everything 
            #    else to the algorithm. We tried with a smaller sphere like 10mm, but it didn't seem to make much of 
            #    a difference from just using the segmentation mask. The 10mm sphere may potentially be useful, 
            #    forcing the region outside the immediate tumor surrounding area to match. BUt all tumors will not be 
            #    the same size and hence 10mm may not be a good idea. Maybe a bigger tumor dilated area is more 
            #    useful, but feels like all this may not make much of a difference. Hence just going with the 
            #    tumor segmentation and the default dilation (I think 1 mm).
            #
            # 2. Registration mask: Using a 50mm mask. But in some cases the closest to ablation zone area was not 
            #    registered well. Hence, experimented with decreasing to a 30mm radius mask. This seemed to work 
            #    okayish, not always better than 50mm, but sometimes. Also, in some cases, the ablation zone is 
            #    larger than 30mm sphere, in which case, there will be no features surrounding the ablation zone 
            #    that the registration will be able to use to match. The conclusion is, try both, but first try 
            #    50mm mask.
            #
            # 3. Rigid vs affine initialization: Rigid is predictable always and yields sufficiently close 
            #    initialization. Affine works better some times, but throws the registration totally off in some 
            #    cases, e.g., unatural scaling and shears. The conclusion is to try both, but first use rigid.
            #
            # Steps:
            #  1. Use 50 mm radius spherical masks on both images to focus the registration (size found to be enough for matching in many cases)
            #  2. Remove ablation zone segmentations (if we only followup abl zone segmentation, remove that from the above masks). 
            #     This is because we don't want the intensities inside the post ablation zone to influence the registation. They may cause 
            #     the ablation zone itself to change shape/size and stretch to match the pre ablation zone, defeating the 
            #     purpose of pre to post prediction. We should retain all the differences between them. For the pre's, we can subtract 
            #     the tumor segmentation or possibly 
            #     use a 20mm radius sphere as a proxy for pre ablation segmentation and remove that from the mask.
            #     The current version uses the pre tumor segmentation.
            #     NOTE: Later we found that including the mask intensities was not harming the registration and infact helping 
            #     some times. The above caveat is still true, this may influence the match around ablation zone, inspite of the rigidity penalty.  
            #     Because, the intensity match and the rigidity penalty are equally weighted during the registration.
            #     CONCLUSION: We will try both including them and removing them.    
            #  3. Apply rigidity penalty on the pre tumor region, so it doesn't deform too much. We use a sphere of radius 
            #     20mm as proxy for the pre tumor region (not fully sure). This will make sure the 
            #     shape and size characteristics of the tumor are maintained to some extent. If allowed to fully deform, 
            #     it may take the shape of the followup post ablation? Is this convincing? Is it enough to do step 2? Test in future versions.
            #  4. Use matching points if present. If not use centers of the tumor and post ablation zone during rigid initialization and 
            #     not during non-rigid phase. Using them during non-rigid registration results in poor registrations in some cases. Also, 
            #     the tumor center doesn;t necessarily have to match the post ablation center.
            #  5. For the registration itself, perform an initial rigid intialization followed by bspline deformable registration. 
            #     Initially, we were using affine for initialization, but observed that it was causing the alignment to be worse than rigid in some 
            #     cases and changing the size/scale of the ablation zone. Hence, we use rigid now. But, we don't know if it is indeed better! 

            # Set fixed and moving images
            try:
                fixed_image = sitk.ReadImage(followup_post_scan_file)
                moving_image = sitk.ReadImage(pre_scan_file)
            except Exception as e:
                # Some times, the image reading itself fails, e.g., multiframe subj62. We store such cases and process later.
                imageread_failed_cases.append(case_name)
                continue


            # 1,2. Create registration masks and remove ablation zone segmentations (we tried both removing and not removing).
            # In the case of pre to post registration, it makes more sense to remove it since the tumor doesn't have to match
            # the ablation zone in shape/intensities. Hence removing it.
            #----------------------------------------------------------------------------------------------------------------
            # Find centers of post ablation zone
            # Followup ablation zone center is computed as the center of the ablation segmentation
            followup_post_abl_center_lps = get_segment_center(followup_post_seg_file)
            # Find center of tumor segmentation
            pre_tumor_center_lps = get_segment_center(pre_seg_file)

            # Compute registration masks
            # Subtracting the follow-up segmentation mask and tumor segmentation mask. The same is not true of pre to post since tumor will not match 
            # with ablation zone. 
            #-------------------------------------------------------------------------------
            # Followup abl. 50mm radius mask. Note that we subtract the ablation zone segmentation itself from the mask.
            reg_mask_radius = 30 #50
            fixed_image_mask = create_mask_img(followup_post_scan_file, center = followup_post_abl_center_lps, shape='sphere', radius=reg_mask_radius, subtraction_mask_file = followup_post_seg_file)
            # Pre tumor centric 50mm radius mask. Note that we subtract the pre tumor segmentation itself from the mask.
            moving_image_mask = create_mask_img(pre_scan_file, center = pre_tumor_center_lps, shape='sphere', radius=reg_mask_radius, subtraction_mask_file = pre_seg_file)
            # # Test if they look ok
            # fixed_image_mask_file = os.path.join(output_case_folder_path, 'fixed_image_reg_mask' + '.nrrd')
            # sitk.WriteImage(sitk.Cast(fixed_image_mask, sitk.sitkUInt8), fixed_image_mask_file)
            # moving_image_mask_file = os.path.join(output_case_folder_path, 'moving_image_reg_mask' + '.nrrd')
            # sitk.WriteImage(sitk.Cast(moving_image_mask, sitk.sitkUInt8), moving_image_mask_file)

            
            # 3. Compute rigidity mask for moving image (rigidity mask for fixed image not that important compared to the moving image,
            # but we try it on the fixed image as well)
            # Two options for the rigidity mask. We expriment with both:
            # 1) A 20mm radius sphere around the tumor
            # 2) The tumor segmentation. This is much smaller than the rigidity mask.
            #----------------------------------------------------------------------------------------------------------------------------
            # Moving
            use_pre_seg_rigidity_mask = True
            if use_pre_seg_rigidity_mask:
                # Option 2
                # We set the pre tumor segmentation as the rigidity mask
                moving_image_rigidity_mask_file = pre_seg_file # This is required to later set the registration parameters
            else:
                # Option 1
                # Create a rigidity mask for the pre images. We use a sphere of radius 20mm centered around the center of the abl zone. 
                pre_rigidity_mask_20 = create_mask_img(pre_scan_file, center = pre_tumor_center_lps, shape='sphere', radius=10)
                pre_rigidity_mask_20_file = os.path.join(output_case_folder_path, 'pre_rigidity_mask_10mm_sphere_file.nrrd')
                sitk.WriteImage(sitk.Cast(pre_rigidity_mask_20, sitk.sitkUInt8), pre_rigidity_mask_20_file)
                moving_image_rigidity_mask_file = pre_rigidity_mask_20_file # This is required to later set the registration parameters

            # Followup (Not sure if this will create any issues or worsen the registration. Check!)
            # Keeping this off in the latest runs as this was not recommended by the elastix documentation.
            use_followup_seg_rigidity_mask = False
            if use_followup_seg_rigidity_mask:
                fixed_image_rigidity_mask_file = followup_post_seg_file # This is required to later set the registration parameters


            # Create .pts files for the tumor and ablation center co-ordinates. We use these for the
            # initial rigid match only.   
            #---------------------------------------------------------------------------------------------------------------------
            # Pre center coords
            pre_tumor_center_coord_file = os.path.join(output_case_folder_path, 'pre_tumor_center_coord.pts')
            #if not os.path.isfile(pre_tumor_center_coord_file): We want to write even if a file exists, just to be sure.
            with open(pre_tumor_center_coord_file, 'w') as pts_file:
                pts_file.write('point\n')
                pts_file.write('1\n')
                pts_file.write('{} {} {}\n'.format(pre_tumor_center_lps[0], pre_tumor_center_lps[1], pre_tumor_center_lps[2]))
            moving_img_tumor_center_coord_file = pre_tumor_center_coord_file

            # Followup post_center_coords
            followup_post_abl_center_coord_file = os.path.join(output_case_folder_path, 'followup_post_abl_center_coord.pts')
            #if not os.path.isfile(followup_post_abl_center_coord_file): We want to write even if a file exists, just to be sure.
            with open(followup_post_abl_center_coord_file, 'w') as pts_file:
                pts_file.write('point\n')
                pts_file.write('1\n')
                pts_file.write('{} {} {}\n'.format(followup_post_abl_center_lps[0], followup_post_abl_center_lps[1], followup_post_abl_center_lps[2]))
                #pts_file.write('{} {} {}\n'.format(followup_post_abl_tip[0], followup_post_abl_tip[1], followup_post_abl_tip[2]))
            fixed_img_abl_center_coord_file = followup_post_abl_center_coord_file


            # 5. Initialize parameters for registration           
            #------------------------------------------------------
            parameterMapVector = sitk.VectorOfParameterMap()

            # Setup registration transformations using both intensity and landmark match as similarity functions
            # Set initial rigid registration. 
            # Note that point match criterion is anyway included. If both scans have
            # other labeled matching points, they will be matched. The center match is used
            # for an initial rigid alignment.  
            rigidMap = sitk.GetDefaultParameterMap("rigid")
            prev_map = list(rigidMap["Metric"])
            prev_map.append("CorrespondingPointsEuclideanDistanceMetric") # This is the point matching metric, which always needs to be the last.
            rigidMap["Metric"] = tuple(prev_map)
            # Assigning equal weight to all loss functions.
            rigidMap["Metric0Weight"] = ("1.0",) # Image similarity. 
            rigidMap["Metric1Weight"] = ("1.0",) # Point(s) match
            rigidMap["Registration"] = ("MultiMetricMultiResolutionRegistration",)
            rigidMap["ImageSampler"] = ("RandomSparseMask",) # This is important to avoid too few samples inside mask error. This essentially only samples from inside the mask. 
            rigidMap["ErodeMask"] = ("false",) # This is important to avoid too few samples inside mask error. This essentially only samples from inside the mask.
            # We didn't turn the automatic initialization on since in many cases, the scan are already aligned better than auto initialization.

            #We didn't use affine since it was throwing off some registrations, making it harder for non-rigid reg. Also,
            #it was affecting the size and shape of the ablation zone.
            # Set initial affine transform
            use_affine_map = False
            affineMap = []
            if use_affine_map:
                affineMap = sitk.GetDefaultParameterMap("affine")
                prev_map = list(affineMap["Metric"])
                prev_map.append("CorrespondingPointsEuclideanDistanceMetric")
                affineMap["Metric"] = tuple(prev_map)
                affineMap["Metric0Weight"] = ("1.0",)
                affineMap["Metric1Weight"] = ("1.0",)
                affineMap["Registration"] = ("MultiMetricMultiResolutionRegistration",)
                affineMap["ImageSampler"] = ("RandomSparseMask",)
                affineMap["ErodeMask"] = ("false",)
                #We didn't turn the automatic initialization on since in many cases, the scan are already aligned better than auto initialization.

            # Non-rigid B-spline transform
            bsplineMap = sitk.GetDefaultParameterMap("bspline")
            prev_map = list(bsplineMap["Metric"])
            prev_map.append("TransformRigidityPenalty")
            bsplineMap["Metric"] = tuple(prev_map)
            # Assigning equal weight to all loss functions.
            bsplineMap["Metric0Weight"] = ("1.0",) # Image similarity
            bsplineMap["Metric1Weight"] = ("1.0",) # Transform bend regularization
            bsplineMap["Metric2Weight"] = ("1.0",) # Local rigidity constraint
            bsplineMap["Registration"] = ("MultiMetricMultiResolutionRegistration",)
            bsplineMap["MovingRigidityImageName"] = (moving_image_rigidity_mask_file,) # Specify the rigidity mask
            if use_followup_seg_rigidity_mask:
                bsplineMap["FixedRigidityImageName"] = (fixed_image_rigidity_mask_file,) # Specify the rigidity mask
            bsplineMap["ImageSampler"] = ("RandomSparseMask",)
            bsplineMap["ErodeMask"] = ("false",)
            # Combine all transforms
            parameterMapVector.append(rigidMap)
            if use_affine_map:
                parameterMapVector.append(affineMap)
            parameterMapVector.append(bsplineMap)

            # 6. Setup the registration
            #------------------------------------------------------------------------
            elastixImageFilter = sitk.ElastixImageFilter()
            # Set images
            elastixImageFilter.SetFixedImage(fixed_image)
            elastixImageFilter.SetMovingImage(moving_image)
            # Set masks
            elastixImageFilter.SetFixedMask(sitk.Cast(fixed_image_mask, sitk.sitkUInt8))
            elastixImageFilter.SetMovingMask(sitk.Cast(moving_image_mask, sitk.sitkUInt8))
            # Set landmark matching
            elastixImageFilter.SetFixedPointSetFileName(fixed_img_abl_center_coord_file)
            elastixImageFilter.SetMovingPointSetFileName(moving_img_tumor_center_coord_file)

            # Set the transforms to the elastix image filters
            elastixImageFilter.SetParameterMap(parameterMapVector)


            # If there are matching co-ordinates in the fixed and moving images, set the below 
            # appropriately so they can be passed as input to the registation function below.
            #----------------------------------------------------------------------------------------
            # For calling the functions below, we need to set these variables
            # # Set matching co-ordinates
            # is_pre_matching_coords_valid = is_pre_matching_coords_valid
            # is_followup_post_matching_coords_valid=is_followup_post_matching_coords_valid
            # fixed_img_matching_coords_file = fixed_img_matching_coords_file
            # moving_img_matching_coords_file=moving_img_matching_coords_file
            # If no matching co-ordinates, set them to empty lists
            is_pre_matching_coords_valid = []
            is_followup_post_matching_coords_valid=[]
            fixed_img_matching_coords_file = []
            moving_img_matching_coords_file=[]


            # 7. Execute registration
            #------------------------------------------------------------------------------------------
            try:
                # We have commented this out since we don't need to register images at their full resolution. 
                # We only register the cropped images!
                #--------------------------------------------------------------------------------------
                #elastixImageFilter.Execute()
                #succeeded_first_attempt.append(case_name)
                raise ValueError('Nothing')
            except Exception as e:
                # Handle registration exception:
                # It is likely "Too many samples map outside moving image buffer". Modify some registration parameters and 
                # re-attempt registration.
                #
                # Solution 1: Crop the images to a sphere of 100mm radius around the ablation center. The masks also have 
                # to be cropped. But retain everything else about the registration as is. Finally, turn on automatic initialization, 
                # which will place the images on top of each other, aligned at the center of the ablation zones.
                try: 
                    print('First registration attempt failed. Trying again with 100mm radius crop...')
                    # We have commented this out since we don't need to register images cropped to 100mm 
                    # spherical regions. Thats is still a large region. We register a further smaller crop! 
                    #---------------------------------------------------------------------------------------
                    #succeeded_second_attempt.append(case_name)
                    #continue
                    raise ValueError('Nothing')
                except Exception as e:
                    # Solution 2: Crop the images to a sphere of 70mm radius around the ablation center. The masks also have 
                    # to be cropped. But retain everything else about the registration as is. Finally, turn on automatic initialization, 
                    # which will place the images on top of each other, aligned at the center of the ablation zones. Note that even though a 
                    # 50mm cropped with a 70mm registration mask appears to make one of them redundant, some cases didn't work if we don't include 
                    # both. Hence doing both.
                    try:
                        print('Second registration attempt failed. Trying again with 70mm radius crop...') 
                        pre_to_post_reg_error_sol_crop_img(followup_post_scan_file = followup_post_scan_file, pre_scan_file = pre_scan_file,
                            followup_post_abl_center_lps = followup_post_abl_center_lps, pre_tumor_center_lps=pre_tumor_center_lps,
                                fixed_image=fixed_image, moving_image=moving_image, output_case_folder_path=output_case_folder_path,
                                    is_pre_matching_coords_valid=is_pre_matching_coords_valid, is_followup_post_matching_coords_valid=is_followup_post_matching_coords_valid,
                                        fixed_img_matching_coords_file=fixed_img_matching_coords_file, moving_img_matching_coords_file=moving_img_matching_coords_file,
                                            fixed_img_abl_center_coord_file=fixed_img_abl_center_coord_file, moving_img_tumor_center_coord_file=moving_img_tumor_center_coord_file,
                                                rigidMap=rigidMap, bsplineMap=bsplineMap, crop_radius=70, followup_post_seg_file = followup_post_seg_file, pre_seg_file = pre_seg_file, use_pre_seg_rigidity_mask=use_pre_seg_rigidity_mask, use_followup_seg_rigidity_mask=use_followup_seg_rigidity_mask, affineMap=affineMap, use_affine_map=use_affine_map, reg_mask_radius=reg_mask_radius)
                        succeeded_third_attempt.append(case_name)
                        continue
                    except Exception as e:
                        # Solution 3: Crop the images to a sphere of 50mm radius around the ablation center. The masks also have 
                        # to be cropped. But retain everything else about the registration as is. Finally, turn on automatic initialization, 
                        # which will place the images on top of each other, aligned at the center of the ablation zones. Note that even though a 
                        # 50mm cropped with a 50mm registration mask appears to make one of them redundant, some cases didn't work if we don't include 
                        # both, e.g. subj65. Hence doing both.
                        try:
                            print('Third registration attempt failed. Trying again with 50mm radius crop...') 
                            pre_to_post_reg_error_sol_crop_img(followup_post_scan_file = followup_post_scan_file, pre_scan_file = pre_scan_file,
                                followup_post_abl_center_lps = followup_post_abl_center_lps, pre_tumor_center_lps=pre_tumor_center_lps,
                                    fixed_image=fixed_image, moving_image=moving_image, output_case_folder_path=output_case_folder_path,
                                        is_pre_matching_coords_valid=is_pre_matching_coords_valid, is_followup_post_matching_coords_valid=is_followup_post_matching_coords_valid,
                                            fixed_img_matching_coords_file=fixed_img_matching_coords_file, moving_img_matching_coords_file=moving_img_matching_coords_file,
                                                fixed_img_abl_center_coord_file=fixed_img_abl_center_coord_file, moving_img_tumor_center_coord_file=moving_img_tumor_center_coord_file,
                                                    rigidMap=rigidMap, bsplineMap=bsplineMap, crop_radius=50, followup_post_seg_file = followup_post_seg_file, pre_seg_file = pre_seg_file, use_pre_seg_rigidity_mask=use_pre_seg_rigidity_mask, use_followup_seg_rigidity_mask=use_followup_seg_rigidity_mask, affineMap=affineMap, use_affine_map=use_affine_map, reg_mask_radius=reg_mask_radius)
                            succeeded_fourth_attempt.append(case_name)
                            continue
                        except Exception as e:
                            # Solution 4: Crop the images to a sphere of 50mm radius around the ablation center. The masks also have 
                            # to be cropped. But retain everything else about the registration as is. Finally, turn on automatic initialization, 
                            # which will place the images on top of each other, aligned at the center of the ablation zones. Note that even though a 
                            # 50mm cropped with a 50mm registration mask appears to make one of them redundant, some cases didn't work if we don't include 
                            # both, e.g. subj65. Hence doing both. Additionally decrease the "RequiredRatioOfValidSamples" to 0.20"                            
                            try:
                                print('Fourth registration attempt failed. Trying again with 50mm radius crop and RequiredRatioOfValidSamples = 0.20...') 
                                pre_to_post_reg_error_sol_crop_img(followup_post_scan_file = followup_post_scan_file, pre_scan_file = pre_scan_file,
                                    followup_post_abl_center_lps = followup_post_abl_center_lps, pre_tumor_center_lps=pre_tumor_center_lps,
                                        fixed_image=fixed_image, moving_image=moving_image, output_case_folder_path=output_case_folder_path,
                                            is_pre_matching_coords_valid=is_pre_matching_coords_valid, is_followup_post_matching_coords_valid=is_followup_post_matching_coords_valid,
                                                fixed_img_matching_coords_file=fixed_img_matching_coords_file, moving_img_matching_coords_file=moving_img_matching_coords_file,
                                                    fixed_img_abl_center_coord_file=fixed_img_abl_center_coord_file, moving_img_tumor_center_coord_file=moving_img_tumor_center_coord_file,
                                                        rigidMap=rigidMap, bsplineMap=bsplineMap, crop_radius=50, requiredRatioOfValidSamples="0.20", followup_post_seg_file = followup_post_seg_file, pre_seg_file = pre_seg_file, use_pre_seg_rigidity_mask=use_pre_seg_rigidity_mask, use_followup_seg_rigidity_mask=use_followup_seg_rigidity_mask, affineMap=affineMap, use_affine_map=use_affine_map, reg_mask_radius=reg_mask_radius)
                                succeeded_fifth_attempt.append(case_name)
                                continue
                            except Exception as e:
                                # Solution 5: Crop the images to a sphere of 50mm radius around the ablation center. The masks also have 
                                # to be cropped. But retain everything else about the registration as is. Finally, turn on automatic initialization, 
                                # which will place the images on top of each other, aligned at the center of the ablation zones. Note that even though a 
                                # 50mm cropped with a 50mm registration mask appears to make one of them redundant, some cases didn't work if we don't include 
                                # both, e.g. subj65. Hence doing both. Additionally decrease the "RequiredRatioOfValidSamples" to 0.15"                            
                                try:
                                    print('Fifth registration attempt failed. Trying again with 50mm radius crop and RequiredRatioOfValidSamples = 0.15...') 
                                    pre_to_post_reg_error_sol_crop_img(followup_post_scan_file = followup_post_scan_file, pre_scan_file = pre_scan_file,
                                        followup_post_abl_center_lps = followup_post_abl_center_lps, pre_tumor_center_lps=pre_tumor_center_lps,
                                            fixed_image=fixed_image, moving_image=moving_image, output_case_folder_path=output_case_folder_path,
                                                is_pre_matching_coords_valid=is_pre_matching_coords_valid, is_followup_post_matching_coords_valid=is_followup_post_matching_coords_valid,
                                                    fixed_img_matching_coords_file=fixed_img_matching_coords_file, moving_img_matching_coords_file=moving_img_matching_coords_file,
                                                        fixed_img_abl_center_coord_file=fixed_img_abl_center_coord_file, moving_img_tumor_center_coord_file=moving_img_tumor_center_coord_file,
                                                            rigidMap=rigidMap, bsplineMap=bsplineMap, crop_radius=50, requiredRatioOfValidSamples="0.15", followup_post_seg_file = followup_post_seg_file, pre_seg_file = pre_seg_file, use_pre_seg_rigidity_mask=use_pre_seg_rigidity_mask, use_followup_seg_rigidity_mask=use_followup_seg_rigidity_mask, affineMap=affineMap, use_affine_map=use_affine_map, reg_mask_radius=reg_mask_radius)
                                    succeeded_sixth_attempt.append(case_name)
                                    continue
                                except Exception as e:            
                                    print('All attempts to register failed. Adding the case to a list of failures.')
                                    registation_failed_cases.append(case_name)
                                    continue 
