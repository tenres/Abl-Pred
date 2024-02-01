# Organizes data from ablation workflows   
# --------------------------------------------------------------------------------------------------------------------------------------------
# This code reads all the necessary data for each subject, i.e., CT scans and ablation treatment parameters and writes them to appropriate folders,
# files and formats. E.g., pre and followup post scans, applicator position, power and duration, 
# the pre tumor segmentation and followup post ablation segmentation. All this is written as into output directory for the subject.
#
# NOTES:
# 1. The code assumes a directory structure where the scans and segmentations are organized as: subject_folder -> pre_folder and post_folder. 
# 2. The code also assumes the ablation applicator position, power and duration are in an excel sheet, which is read and organized.
# 3. That if the case directory already exists in the output path, it is complete and hence no files are copied over. If you want to 
# re-run some cases and want them to be copied over again. Delete the cases directories in the output path and process again.
# 2. The followup post antenna co-ordinates are read from fcsv files in the latest runs. But for the ones that do not have the fcsv file (vast majority), we read the co-ordinates from 
# the excel file. They were assumed to be good enough. 
#

# Imports
import pandas as pd
import os
import glob
import numpy as np
import shutil
import SimpleITK as sitk


def read_ant_coords_fcsv_file(ant_coords_fcsv_file):
    '''
    Read antenna co-ordinates from the fcsv file and return them in LPS co-ordinates as dicts. The passed in 
    antenna coords file is read and the co-ordinates are converted to LPS from whatever format they were in 
    (# CoordinateSystem = 0 is RAS and # CoordinateSystem = 1 is LPS)
    '''

    # Check input 
    if not os.path.isfile(ant_coords_fcsv_file):
        raise ValueError('Invalid input. Existing...') 

    # Find the co-ordinate format (LPS/RAS)
    #--------------------------------------------------------
    # Find line that has string "CoordinateSystem" in the file and read everything to the right of the "=" sign". 
    # If that is "0", then it is RAS. Else, it is LPS. 
    coords_file = open(ant_coords_fcsv_file, 'r')
    coord_sys = None # Initialize the co-ordinate system 
    for line in coords_file:
        # Check if the line has "CoordinateSystem" (assuming there is only 1 such line)
        if 'CoordinateSystem' in line:
            # Extract the string to the right of '=' and strip it of leading and trailing white space, new lines (\n) and tabs (\n) and \r etc.
            coord_sys_int_str = line.rsplit("=", 1)[1].strip()
            # Get the integer code corresponding to the co-ordinate system
            coord_sys_int = int(coord_sys_int_str)
            # Set the co-ordinate system
            if coord_sys_int == int(0): 
                coord_sys = 'RAS'
            elif coord_sys_int == int(1):
                coord_sys = 'LPS'
            else:
                # Raise error for any other value
                raise ValueError('The co-ordinate system in the antenna coords file {} is invalid. Exiting...'.format(os.path.basename(ant_coords_fcsv_file)))
            # Stop search after the co-ordinate system is found
            break
    # Close fcsv file
    coords_file.close()
    # If the coordinateSystem was not found in the file, raise error
    if coord_sys is None:
        raise ValueError('The co-ordinate system in the antenna coords file {} was not found. Exiting...'.format(os.path.basename(ant_coords_fcsv_file)))


    # Read the antenna co-ordinates from file
    #---------------------------------------------------------------------
    ant_coords_dict ={}
    # Read the antenna coords fcsv file as a csv file (skip all comment lines and only read the lines that have values.
    # Note that the header is commented, hence not read. We work with indices later).
    ant_coords_df = pd.read_csv(ant_coords_fcsv_file, skip_blank_lines=True, comment='#', header=None) # Note: We read everything as string as to make sure we don't miss any leading 0s.
    # Retain only valid rows
    ant_coords_df.drop_duplicates(keep='first', inplace=True)
    # Set row indices to start from 0 again
    ant_coords_df.reset_index(drop=True, inplace=True)

    # Extract the x,y,z co-ordinates of the antenna tip (always denoted by 'F-1' in the 11th column, which is the 
    # markup label, and the first row entry in the fcsv file). Also note that columns with headers [1,2,3] correspond
    # to x, y, z co-ordinates respectively.
    tip_coord_df = ant_coords_df.loc[ant_coords_df[11] == 'F-1', [1,2,3]]
    # If data frame empty, raise exception
    if tip_coord_df.empty:
        raise ValueError('The antenna coords file {} does not have tip coords (F-1). Exiting...'.format(os.path.basename(ant_coords_fcsv_file)))
    # Read co-ordinates. Note that the row coords are 0, since that is the first row
    x_tip = tip_coord_df.loc[0,1]
    y_tip = tip_coord_df.loc[0,2]
    z_tip = tip_coord_df.loc[0,3]

    # Extract the x,y,z co-ordinates of the antenna back1 (always denoted by 'F-2' in the 11th column, which is the 
    # markup label, and the first row entry in the fcsv file). Also note that columns with headers [1,2,3] correspond
    # to x, y, z co-ordinates respectively.
    back1_coord_df = ant_coords_df.loc[ant_coords_df[11] == 'F-2', [1,2,3]]
    # If data frame empty, raise exception
    if back1_coord_df.empty:
        raise ValueError('The antenna coords file {} does not have back1 coords (F-2). Exiting...'.format(os.path.basename(ant_coords_fcsv_file)))
    # Read co-ordinates. Note that the row coords are 1, since that is the second row
    x_back1 = back1_coord_df.loc[1,1]
    y_back1 = back1_coord_df.loc[1,2]
    z_back1 = back1_coord_df.loc[1,3]

    # Extract the x,y,z co-ordinates of the antenna back2 (always denoted by 'F-3' in the 11th column, which is the 
    # markup label, and the first row entry in the fcsv file). Also note that columns with headers [1,2,3] correspond
    # to x, y, z co-ordinates respectively.
    back2_coord_df = ant_coords_df.loc[ant_coords_df[11] == 'F-3', [1,2,3]]
    # If data frame empty, raise exception
    if back2_coord_df.empty:
        raise ValueError('The antenna coords file {} does not have back2 coords (F-3). Exiting...'.format(os.path.basename(ant_coords_fcsv_file)))
    # Read co-ordinates. Note that the row coords are 2, since that is the third row
    x_back2 = back2_coord_df.loc[2,1]
    y_back2 = back2_coord_df.loc[2,2]
    z_back2 = back2_coord_df.loc[2,3]

    # If co-ordinates in RAS, convert them to LPS
    if coord_sys == 'RAS':
        # Tip
        x_tip = -1. * x_tip
        y_tip = -1. * y_tip
        # Back 1
        x_back1 = -1. * x_back1
        y_back1 = -1. * y_back1
        # Back 2
        x_back2 = -1. * x_back2
        y_back2 = -1. * y_back2
    
    # Save the coords into dictionaries
    # Tip
    lps_coords_tip = {}
    lps_coords_tip['L'] = x_tip
    lps_coords_tip['P'] = y_tip
    lps_coords_tip['S'] = z_tip
    # Back 1
    lps_coords_back1 = {}
    lps_coords_back1['L'] = x_back1
    lps_coords_back1['P'] = y_back1
    lps_coords_back1['S'] = z_back1
    # Back 2
    lps_coords_back2 = {}
    lps_coords_back2['L'] = x_back2
    lps_coords_back2['P'] = y_back2
    lps_coords_back2['S'] = z_back2

    # Add co-ordinates to dictionary
    ant_coords_dict['tip_lps'] = lps_coords_tip    
    ant_coords_dict['back1_lps'] = lps_coords_back1    
    ant_coords_dict['back2_lps'] = lps_coords_back2    

    return ant_coords_dict


def get_post_ant_coords_from_files(data_base_path, case_name):
    """
    Read the antenna coordinates from the antenna coord files for this case. The files include antenna coords of 
    followup post.
    """

    # Construct case folder path
    case_folder_path = os.path.join(data_base_path, case_name)
    # Base case name is not really needed here since the antenna coords files start with full case name. This code was just copied as is.
    # Store the name of the base name of the case. i.e. subj14a is subj14. This is required 
    # because when querying for the post_subj14 files in the post folder, the scan file itself 
    # only has the base case name in its name, i.e., post_subj14.nrrd. Whereas the specific 
    # ablation segmentation has the a, b, c suffixes as well. Hence while searching for post, 
    # we search using just the base_case_name  
    base_case_name = case_name 
    # Check if a folder of the name exists. If not, this case must have multiple ablations.
    if not os.path.isdir(case_folder_path):
        # This case must have multiple ablations since otherwise it would have been found. It should have a 
        # lower case alphabet suffix such as 'a', 'b', 'c' etc. Remove the alphabet suffix and try again.
        if os.path.isdir(os.path.join(data_base_path, case_name[:-1])):
            case_folder_path = os.path.join(data_base_path, case_name[:-1])
            base_case_name = case_name[:-1] 
        else:
            raise ValueError('Case not found. Exiting...')


    # Get the followup post antenna coords file
    is_followup_post_ant_coords_file_found = False
    followup_post_ant_coords_file = get_files(os.path.join(case_folder_path, 'post'), case_name + '*' + 'antcoord' + '*', '.fcsv')
    # If the size of the list is not 1, raise an exception
    if len(followup_post_ant_coords_file) == 1:
        is_followup_post_ant_coords_file_found = True
        #raise ValueError('Either found more than 1 antenna coords file or not found any at all for this case. Exiting...')
    else:    
        # Check inside the specific ablation folder inside the subject directory, with suffix a, b, c etc. 
        followup_post_ant_coords_file = get_files(os.path.join(case_folder_path, 'post', 'post_' + case_name), case_name + '*' + 'antcoord' + '*', '.fcsv')
        if len(followup_post_ant_coords_file) == 1:
            is_followup_post_ant_coords_file_found = True 


    # Create a dictionary of the antenna co-ordinates existence conditions, antenna co-ordinates and return
    ant_coords_dict = {}
    ant_coords_dict['is_followup_post_ant_coords_file_found'] = is_followup_post_ant_coords_file_found
    # If the valid files are found, add them
    if is_followup_post_ant_coords_file_found:
        ant_coords_dict['followup_post_ant_coords'] = read_ant_coords_fcsv_file(followup_post_ant_coords_file[0])

    return ant_coords_dict



def get_lps_coords(df_row, col_name_with_all_coords):
    '''
    Function takes a data frame row as input and column name of the co-ordinates and return co-ordinates in LPS 
    Note that the co-ordinates include values for all 3 cases, each starting with the name of the axis (L/R, P/A, S/I),
    all separated by commas.
    '''

    # Debug
    #print(df_row['watts'])
    #print(df_row[col_name_with_all_coords])

    # Check if the co-ordinates exist. If not empty, we assume it is a string. 
    if pd.isnull(df_row[col_name_with_all_coords]):
        raise ValueError('The co-ordinates cell is empty for this case. Exiting...')

    # Check if the antenna coords are valid (i.e. the value is not 'na' or None or '' empty string or '-' or 'o', then it is assumed valid)
    if (df_row[col_name_with_all_coords].strip() == 'na') or (not bool(df_row[col_name_with_all_coords].strip()) or (df_row[col_name_with_all_coords].strip() == '-') or (df_row[col_name_with_all_coords].strip() == 'o')):
        raise ValueError('The co-ordinates cell is invalid for this case. Exiting...')

    # Set variable for the lps co-ordinates
    lps_coords = {}
    # Get the co-ordinates
    coords = (df_row[col_name_with_all_coords].strip()).split(',')
    # Convert the co-ordinates to LPS and save only the float co-ordinate values
    # L axis
    coords[0] = coords[0].strip()
    # Read the co-ordinate value and convert it to float
    coord_val = float(coords[0][1:])
    if coords[0][0].upper() == 'R': # Check first charactor
        # Set L value
        lps_coords['L'] = -1.0*coord_val
    elif coords[0][0].upper() == 'L':
        # Already in L, do nothing
        lps_coords['L'] = coord_val
    else:
        raise ValueError('The L axis co-ordinate has a prefix charactor that is not L or R. This is invalid. Exiting...')
    # P axis
    coords[1] = coords[1].strip()
    # Read the co-ordinate value and convert it to float
    coord_val = float(coords[1][1:])
    if coords[1][0].upper() == 'A':
        # Set P value
        lps_coords['P'] = -1.0*coord_val
    elif coords[1][0].upper() == 'P':
        # Already in P, do nothing
        lps_coords['P'] = coord_val
    else:
        raise ValueError('The P axis co-ordinate has a prefix charactor that is not P or A. This is invalid. Exiting...')
    # S axis
    coords[2] = coords[2].strip()
    # Read the co-ordinate value and convert it to float
    coord_val = float(coords[2][1:])
    if coords[2][0].upper() == 'I':
        # Set S value
        lps_coords['S'] = -1.0*coord_val
    elif coords[2][0].upper() == 'S':
        # Already in S, do nothing
        lps_coords['S'] = coord_val
    else:
        raise ValueError('The S axis co-ordinate has a prefix charactor that is not S or I. This is invalid. Exiting...')
    
    return lps_coords


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


def get_data_files(data_base_path, case_name):
    """
    Get data files from the shared drive.
    """

    # Construct case folder path
    case_folder_path = os.path.join(data_base_path, case_name)
    # Store the name of the base name of the case. i.e. subj14a is subj14. This is required 
    # because when querying for the post_subj14 files in the post folder, the scan file itself 
    # only has the base case name in its name, i.e. post_subj14.nrrd. Whereas the specific 
    # ablation segmentation has the a, b, c suffixes as well. Hence while searching for post, 
    # we search using just the base_case_name  
    base_case_name = case_name 
    # Check if a folder of the name exists. If not, this case must have multiple ablations.
    if not os.path.isdir(case_folder_path):
        # This case must have multiple ablations since otherwise it would have been found. It should have a 
        # lower case alphabet suffix such as 'a', 'b', 'c' etc. Remove the alphabet suffix and try again.
        if os.path.isdir(os.path.join(data_base_path, case_name[:-1])):
            case_folder_path = os.path.join(data_base_path, case_name[:-1])
            base_case_name = case_name[:-1] 
        else:
            raise ValueError('Case not found. Exiting...')


    # Get the pre scan and segmentation label in the pre folder
    # Check in the subject directory 
    pre_files = get_files(os.path.join(case_folder_path, 'pre'), 'pre_' + '*' + base_case_name, '.nrrd')
    # If the size of the list is not 2, raise an exception
    are_pre_files_found = False 
    if len(pre_files) == 2:
        are_pre_files_found = True 
    else:    
        # Check inside the specific ablation folder inside the subject directory 
        pre_files = get_files(os.path.join(case_folder_path, 'pre', 'pre_' + case_name), 'pre_' + '*' + base_case_name, '.nrrd')
        if len(pre_files) == 2:
            are_pre_files_found = True 

    # If exactly two pre files found, check further if they are the scan and the segmentation label.
    if are_pre_files_found:
        # Find the segmentation file
        if 'seg' in os.path.basename(pre_files[0]).lower():
            pre_seg_file = pre_files[0]
            pre_scan_file = pre_files[1]
        elif 'seg' in os.path.basename(pre_files[1]).lower():
            pre_seg_file = pre_files[1]
            pre_scan_file = pre_files[0]
        else:
            # Segmentation file not found. This is incomplete case that cannot be used. 
            # TODO This doesn't catch the case where the segmentation is a valid file, but the scan is not. The scan is assumed to be valid for now and this needs to be fixed.
            are_pre_files_found = False


    # Get the followup post scan and segmentation label in the post folder
    # Check in the subject directory 
    followup_post_files = get_files(os.path.join(case_folder_path, 'post'), 'post_' + '*' + base_case_name, '.nrrd')
    # If the size of the list is not 2, raise an exception
    are_followup_post_files_found = False 
    if len(followup_post_files) == 2:
        are_followup_post_files_found = True 
    else:    
        # Check inside the specific ablation folder inside the subject directory 
        followup_post_files = get_files(os.path.join(case_folder_path, 'post', 'post_' + case_name), 'post_' + '*' + base_case_name, '.nrrd')
        if len(followup_post_files) == 2:
            are_followup_post_files_found = True 

    # If exactly two post files found, check further if they are the scan and the segmentation label.
    if are_followup_post_files_found:
        # Find the segmentation file
        if 'seg' in os.path.basename(followup_post_files[0]).lower():
            followup_post_seg_file = followup_post_files[0]
            followup_post_scan_file = followup_post_files[1]
        elif 'seg' in os.path.basename(followup_post_files[1]).lower():
            followup_post_seg_file = followup_post_files[1]
            followup_post_scan_file = followup_post_files[0]
        else:
            # Segmentation file not found. This is incomplete case that cannot be used. 
            # TODO This doesn't catch the case where the segmentation is a valid file, but the scan is not. The scan is assumed to be valid for now and this needs to be fixed.
            are_followup_post_files_found = False

    # Create a dictionary of the output file paths and existence conditions and return
    data_files_dict = {}
    data_files_dict['are_pre_files_found'] = are_pre_files_found 
    data_files_dict['are_followup_post_files_found'] = are_followup_post_files_found
    # If the valid files are found, add them
    if are_pre_files_found:
        data_files_dict['pre_seg_file'] = pre_seg_file
        data_files_dict['pre_scan_file'] = pre_scan_file
    if are_followup_post_files_found:
        data_files_dict['followup_post_seg_file'] = followup_post_seg_file
        data_files_dict['followup_post_scan_file'] = followup_post_scan_file

    return data_files_dict



def copy_data_files(data_files_dict, case_name, output_base_path, copy_pre=False, copy_followup_post=False, overwrite=False):
    """
    Copy data files to respective output folders.
    TODO: We have not used the overwrite input. Currently, the file will be overwritten if they already exist.
    """

    # Construct out directory path to copy files and create a new case folder
    output_case_folder_path = os.path.join(output_base_path, case_name)
    # If output folder doesn't alredy exist, create it. But check if the input files are valid. If not, don't create 
    # any directory at the output.
    if data_files_dict['are_pre_files_found'] or \
        data_files_dict['are_followup_post_files_found']:
        # Create a directory at the output path if it doesn't already exist
        if not os.path.exists(output_case_folder_path):
            os.makedirs(output_case_folder_path)
    else:
        print('Source files do not exist. No output directory created as there is nothing to copy.')
        return


    # Pre scan
    if copy_pre:
        try:
            # Create output scan file name
            output_file_name = os.path.join(output_case_folder_path, os.path.basename(data_files_dict['pre_scan_file']))
            # Copy file
            shutil.copyfile(data_files_dict['pre_scan_file'], output_file_name)
        except KeyError:
            # File doesn't exist for the case
            pass
    # Pre seg
    if copy_pre:
        try:
            # Create output seg file name
            output_file_name = os.path.join(output_case_folder_path, os.path.basename(data_files_dict['pre_seg_file']))
            # Copy file
            shutil.copyfile(data_files_dict['pre_seg_file'], output_file_name)
        except KeyError:
            # File doesn't exist for the case
            pass

    # Followup post scan
    if copy_followup_post:
        try:
            # Create output scan file name
            output_file_name = os.path.join(output_case_folder_path, os.path.basename(data_files_dict['followup_post_scan_file']))
            # Copy file
            shutil.copyfile(data_files_dict['followup_post_scan_file'], output_file_name)
        except KeyError:
            # File doesn't exist for the case
            pass
    # Followup post seg
    if copy_followup_post:
        try:
            # Create output seg file name
            output_file_name = os.path.join(output_case_folder_path, os.path.basename(data_files_dict['followup_post_seg_file']))
            # Copy file
            shutil.copyfile(data_files_dict['followup_post_seg_file'], output_file_name)
        except KeyError:
            # File doesn't exist for the case
            pass

    return



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



if __name__ == '__main__':


    # Read pre excel file. Note that this excel is used to drive the data processing
    #----------------------------------------------------------------------------------------------
    pre_op_file = r'/home/path_to_ablation_folder/pre_data.xlsx'
    pre_op_df = pd.read_excel(pre_op_file, 'NeedleNotes', dtype=str) # Note: We read everything as string as to make sure we don't miss any leading 0s.
    # Retain only valid rows
    pre_op_df.drop_duplicates(keep='first', inplace=True)
    # Set row indices to start from 0 again
    pre_op_df.reset_index(drop=True, inplace=True)

    # Read followup post proc. data excel file
    #-------------------------------------------
    post_op_file = r'/home/path_to_ablation_folder/post_data.xlsx'
    post_op_df = pd.read_excel(post_op_file, 'NeedleNotes', dtype=str) # Note: We read everything as string as to make sure we don't miss any leading 0s.
    # Retain only valid rows
    post_op_df.drop_duplicates(keep='first', inplace=True)
    # Set row indices to start from 0 again
    post_op_df.reset_index(drop=True, inplace=True)

    # Image data files base path
    data_base_path = r'/home/path_to_ablation_folder'

    # Output base path
    output_base_path = r'/home/path_to_ablation_folder/output'


    # Dictionary to store relevant ablation information across all cases. 
    ablation_info_all = {}


    # Decide what to retain here
    #-------------------------------
    # Iterate through all rows of the pre op excel file, store the relevant ablation data and move the ablation images to 
    # respective locations. This is done for pre and post op images. Note that pre op excel file is used as the driver.
    # Keep track of cases already processed
    processed_cases = []
    # Cases that failed when reading the antenna coords from file
    post_ant_coords_from_file_fail = []
    # Debug
    ant_coords_from_file_post_not_good = []
    ant_coords_good_post = []
    pre_good_case = []

    # Variable to use only the new antenna co-ordinates saved as fcsv files. If this is set to False, the code will first check for new antenna co-ordinates in 
    # fcsv files, and if the file doesn't exist, will use the old antenna co-ordinates from the excel sheet.
    use_only_new_post_antenna_coords_fcsv = False # Post co-ordinates were done only for the ones missing in the excel sheet. The excel sheet ones are still good. 
    # In any case, we first check if there is a fcsv file, and if not, we read from the excel sheet. This is future proof, if we decide redo some of those and 
    # save as fcsv files.   


    # Iterate across all cases
    for index, row in pre_op_df.iterrows():

        print('Processing case {} at index {}'.format(row['subjXX'].strip(), index))

        # Debug
        #---------------------
        # Post op excel file
        #---------------------
        # post_case_row = post_op_df.loc[post_op_df['subjXX'].str.strip() == row['subjXX'].strip()]
        # # Check if the number of returned rows is > 1 and if yes, there is a problem. 'subjXX' should be unique across rows.
        # if len(post_case_row) > 1:
        #     raise ValueError('Two or more rows found with the same subjXX {} in the follow-up post excel file. This is invalid. Exciting...'.format(row['subjXX']))
        # lps_coords = {}
        # try:
        #     lps_coords = get_lps_coords(post_case_row.iloc[0], 'need tip coord')
        # except ValueError:
        #     pass
        # print(lps_coords)
        #



        # Check if the case was already added to the ablation dictionary. If yes, we have a problem since the subjXX should 
        # be unique across rows. 
        if row['subjXX'].strip() not in processed_cases:
            # Add to processed cases list
            processed_cases.append(row['subjXX'].strip())
        else:
            raise ValueError('SubjXX {} found in multiple rows in the post op excel file. This is invalid. Exciting...'.format(row['subjXX'].strip()))


        # Get data files existence and path information for both pre and followup post. If the data files do not exist, 
        # we can skip the case already
        #-------------------------------------------------------------------------------------------------------------
        data_files_dict = get_data_files(data_base_path, row['subjXX'].strip()) 
        # Add to the ablation info dict of the case
        ablation_info['are_pre_files_found'] = data_files_dict['are_pre_files_found']
        ablation_info['are_followup_post_files_found'] = data_files_dict['are_followup_post_files_found']
        # Data files. If they don't exist, it excepts and we ignore it.
        # Pre
        try: 
            ablation_info['pre_seg_file'] = data_files_dict['pre_seg_file']
            ablation_info['pre_scan_file'] = data_files_dict['pre_scan_file']
        except KeyError:
            #pass
            # The pre files do not exist. continue to next case
            continue
        # Followup post
        try: 
            ablation_info['followup_post_seg_file'] = data_files_dict['followup_post_seg_file']
            ablation_info['followup_post_scan_file'] = data_files_dict['followup_post_scan_file']
        except KeyError:
            #pass
            # The followup files do not exist. continue to next case
            continue
        

        print('Completed getting data files existence and path information')


        # Read the pre tumor centroid from the excel sheet and if not present, compute it from the segmentation
        # We changed this. We now compute the tumor center from the segmetnation itself using a faster version of 
        # the centroid computation code. 
        #--------------------------------------------------------------------------------------------------------
        is_pre_tumor_center_coords_valid = False # Assuming the tumor center from the excel sheet is invalid. 
        lps_coords_tumor_center = {}

        # # COMMENTING THIS SINCE WE COMPUTE THE CENTER FROM THE SEGMENTATION NOW. 
        # # Read tumor center coords from file. If it doesn't exist, compute it from the tumor segmentation.
        # try:
        #     lps_coords_tumor_center = get_lps_coords(row, 'pre_tumor_coord')
        #     is_pre_tumor_center_coords_valid = True
        # except ValueError:
        #     #wo_abl_center_coords_missing.append(row['subjXX'].strip())
        #     pass

        # If the center co-ordinates do not exist in the excel file, compute them from the segment
        if not is_pre_tumor_center_coords_valid:
            try:
                pre_tumor_center_lps = get_segment_center(ablation_info['pre_seg_file'])
                lps_coords_tumor_center['L'] = pre_tumor_center_lps[0] 
                lps_coords_tumor_center['P'] = pre_tumor_center_lps[1]
                lps_coords_tumor_center['S'] = pre_tumor_center_lps[2]
                is_pre_tumor_center_coords_valid = True
                pre_good_case.append(row['subjXX'].strip()) 
            except Exception as e:
                # If the center co-ordinates cannot be computed, something wrong with the segmentation file. Continue to next case.
                continue

        # Store them
        ablation_info['pre_tumor_center_coords'] = lps_coords_tumor_center
        ablation_info['is_pre_tumor_center_coords_valid'] = is_pre_tumor_center_coords_valid



        # Followup post (1 month followup scan) read antenna position
        #-----------------------------------------------------------------------------------------
        # Find the corresponding row in the followup post excel using the subject name as the key
        post_case_row = post_op_df.loc[post_op_df['subjXX'].str.strip() == row['subjXX'].strip()]
        # Check if the number of returned rows is > 1 and if yes, there is a problem. 'subjXX' should be unique across rows.
        if len(post_case_row) > 1:
            raise ValueError('Two or more rows found with the same subjXX {} in the follow-up post excel file. This is invalid. Exciting...'.format(row['subjXX']))

        # Get the followup post antenna coordinates from file (if it exists) for this case. If it doesn't exist, use the ones in the 
        # excel sheet (fcsv files do not exist for most cases at this point, but we still check them first, just so in the future, if 
        # we decide to redo the coords, we first check the fcsv file)
        #-----------------------------------------------------------------------------------------------------------------------------
        try:
            # Get antenna co-ordinates from file
            ant_coords_dict_from_file = get_post_ant_coords_from_files(data_base_path, row['subjXX'].strip())
        except:
            ant_coords_dict_from_file = None
            post_ant_coords_from_file_fail.append(row['subjXX'].strip())
            # Commented out pass. With pass, we will read the excel sheet co-ordinates if they exist. If not, we skip the case. But, we redid the antenna
            # coords from scratch again and saved them as files (confirm this). Hence, we are not using any of the excel sheet co-ordinates done by Mario.
            # Hence changed the pass to continue.
            # pass
            if use_only_new_post_antenna_coords_fcsv:
                continue
            else:
                pass


        # Get LPS co-ordinates of the antenna tip, back1 and back2
        #----------------------------------------------------------
        is_followup_post_antenna_coords_valid = False

        # Check and get LPS co-ordinates of antenna tip
        # Initialize the needle tip, back1, back2 coords
        lps_coords_tip = {}
        lps_coords_back1 = {}
        lps_coords_back2 = {}
        # Use co-ordinates read from file if they exist
        if ant_coords_dict_from_file is not None: # This would be None only if the extracting the coords from file excepted. The control flow will not even 
            #come here if we used only new antenna co-ordinates from fcsv files, since we would continue to the next case.  
            # If valid coords were found from files, add them here
            if ant_coords_dict_from_file['is_followup_post_ant_coords_file_found']:
                lps_coords_tip = ant_coords_dict_from_file['followup_post_ant_coords']['tip_lps']
                lps_coords_back1 = ant_coords_dict_from_file['followup_post_ant_coords']['back1_lps']
                lps_coords_back2 = ant_coords_dict_from_file['followup_post_ant_coords']['back2_lps']

        # Check if the antenna coords from file are valid
        # Check the validity (completeness) of the co-ordinates    
        if (len(lps_coords_tip) != int(3)) or (len(lps_coords_back1) != int(3)) or (len(lps_coords_back2) != int(3)):
            ant_coords_from_file_post_not_good.append(row['subjXX'].strip())
            # The antenna co-ordinates were recreated saved as fcsv files. Hence, do not bother reading them from excel if specified that 
            # way
            if not use_only_new_post_antenna_coords_fcsv:
                # The antenna co-ordinates read from files are not complete. Hence read the co-ordinates from the excel files
                # Needle tip
                lps_coords_tip = {}
                try:
                    lps_coords_tip = get_lps_coords(post_case_row.iloc[0], 'need tip coord')
                except ValueError:
                    pass
                # Needle back 1
                lps_coords_back1 = {}
                try:
                    lps_coords_back1 = get_lps_coords(post_case_row.iloc[0], 'need back 1 coord')
                except ValueError:
                    pass
                # Needle back 2
                lps_coords_back2 = {}
                try:
                    lps_coords_back2 = get_lps_coords(post_case_row.iloc[0], 'need back 2 coord')
                except ValueError:
                    pass


        # Check the validity (completeness) of the co-ordinates    
        if (len(lps_coords_tip) == 3) and (len(lps_coords_back1) == 3) and (len(lps_coords_back2) == 3):
            is_followup_post_antenna_coords_valid = True
            ant_coords_good_post.append(row['subjXX'].strip())
            
        # Store all the followup post antenna co-ordinates for this case
        post_antenna_coords = {}
        post_antenna_coords['lps_coords_tip'] = lps_coords_tip
        post_antenna_coords['lps_coords_back1'] = lps_coords_back1
        post_antenna_coords['lps_coords_back2'] = lps_coords_back2

        # Add co-ordinates info to ablation_info dict
        ablation_info['post_antenna_coords'] = post_antenna_coords
        ablation_info['is_followup_post_antenna_coords_valid'] = is_followup_post_antenna_coords_valid

        print('Completed reading followup post scan antenna co-ordinates')



        # Considering only pre and followup scan. 
        # Move files when both pre and followup have scan and segmentation files, we could compute tumor center on pre, and followup
        # has valid antenna co-ordinates  
        #---------------------------------------------------------------------------------------------------------------------------
        # Check if the case directory already exists in the output path. If yes, skip copying files for this case. Assume that if the directory exists, the case 
        # already has everything in it and complete. 
        output_case_folder_path = os.path.join(output_base_path, row['subjXX'].strip())
        if not os.path.exists(output_case_folder_path):
            # Copy files if they have all relevant data
            if (ablation_info['are_pre_files_found'] and ablation_info['are_followup_post_files_found'] and \
                ablation_info['is_followup_post_antenna_coords_valid'] and ablation_info['is_pre_tumor_center_coords_valid']):                       
                copy_data_files(data_files_dict, row['subjXX'].strip(), output_base_path, copy_pre=True, copy_followup_post=True)

        print('Case has all the required data. Completed saving all the data to the output directory.')

        # Save the ablation info dictionary
        ablation_info_all[row['subjXX'].strip()] = ablation_info

        print('Completed processing case {} at index {}'.format(row['subjXX'].strip(), index))


    # Save the ablation info of all cases to file
    ablation_info_all_file = os.path.join(output_base_path, 'pre_post_ablation_info_all')
    np.savez_compressed(ablation_info_all_file, ablation_info_all=ablation_info_all)
    print("Completed saving processed data to disk")


