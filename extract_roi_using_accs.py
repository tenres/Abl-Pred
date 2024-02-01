################################################################################################################################################################################################################################
# Copyright © 2024, Krishna Nand Keshava Murthy, Etay Ziv, Memorial Sloan-Kettering Cancer Center, Memorial Hospital for Cancer and Allied Diseases, and Sloan-Kettering Institute for Cancer Research, all rights reserved. 
# This software is for academic research purposes only. You may only download the software if you are an employee of a nonprofit research institute and will use the code solely for academic, nonprofit research. All Users 
# must agree to be bound by the conditions described herein before downloading the software. For all other uses including non-academic use or commercial use of the software including commercial use, please contact 
# willk12@mskcc.org.
#
# THIS SOFTWARE IS PROVIDED “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY. OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Authors hereby provide Users with a revocable, non-exclusive license to the ABL-PRED code for academic research purposes only.  Users acknowledge and agree that they will not engage in unauthorized copying, reproduction, 
# distribution, modification, display, public performance, sale, reposting, preparation of derivative works based on, or use of the ABL-PRED code for any other purpose under this license. Upon revocation of the license, 
# Users agree to remove and/or shall cause to remove all copies of ABL-PRED code contained in or on the Users’ premises, systems, or any other equipment or location otherwise under Users’ control. 
#
# YOU UNDERSTAND THAT BY CLICKING THIS LINK (OR ACCESSING/EXECUTING THIS CODE), YOU AGREE TO BE BOUND BY THE TERMS OF THIS LICENSE.
################################################################################################################################################################################################################################
#
#
# Extracting applicator/antenna centric co-ordinate system as a generic tool to standardize ablation zones across cases and focus on ROI
#------------------------------------------------------------------------------------------------------------------------
# This code extracts the applicator centric co-ordinate system based scan representation for focused ablation zone data analysis.

# Notes:
# - This code can be used for extracting antenna centric bounding boxes in followup post and pre ablation zones. 
# - The driver for the bounding box extraction has to be followup post since they may have the antenna co-ordinates to place the bounding box. 
# - It is required that all images whose bounding is being extracted (pre and followup post) are all in the same space, i.e., they are all co-registered. 
# - We take the two images as well as the followup ablation segmentation and tumor pre segmentation as input, extract antenna centric bounding boxes and 
# - write them as output.
# - Key: If any of the post scan or post ablation segmentation is missing for a case, it is considered a failure and we move on to the next case.    
#
#
# Crop ablation scans to a local co-ordinate system surrounding the tumor/ablation zone.
#
# The sampled region is a cubical box for now.
#
# Notes:
# In this code, the voxel space indices are labeled r(row), c(column), and s(slice).
# The transform to and from x,y,z can be to either LPS or RAS co-ordinate system
#


# Imports
import SimpleITK as sitk
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
import nibabel as nib
import os
import glob
from sklearn.model_selection import train_test_split, KFold
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from collections import OrderedDict
import pickle
import re


def crs_to_xyz_transform(sitk_image):
    """
    Return the c(column), r(row), s(slice) voxel space to x,y,z patient co-ordinate system transform. The patient co-ordinate
    system could be LPS/RAS depending on the nrrd format (since we take an sitk image and itk uses LPS, it doesn't matter if the original 
    co-ordinates are in RAS/LPS).
    The output is an affine transformation matrix
    References:
    https://www.slicer.org/wiki/Coordinate_systems
    https://discourse.slicer.org/t/building-the-ijk-to-ras-transform-from-a-nrrd-file/1513
    https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch4.html
    https://discourse.itk.org/t/are-the-vectors-defining-the-direction-of-a-simpleitk-image-normalized/810

    :param sitk_image : Simpleitk image.

    :return: crs to xyz transform

    Note:
    - It is assumed in the below, c,r,s are used for voxel co-ordinate indices along columns, rows and slices axes.
    """

    # Check if the series contains no files
    if not sitk_image:
        raise ValueError('Series header is empty..')

    # # Note required since ITK uses LPS
    # # Check if the anatomical co-ordinate is neither of LPS or RAS
    # if series_header['space'] != 'left-posterior-superior' and series_header['space'] != 'right-anterior-superior':
    #     raise ValueError('Series is neither LPS or RAS. Other co-ordinates are currently not supported..')

    # Get pixel spacing (in the order x,y,z, i.e. c,r,s, i.e. along columns, along rows and along slices)
    image_spacing = sitk_image.GetSpacing()
    # Get directional cosines (order: positive row axis direction, positive column axis direction, positive slice axis direction)
    # Note that the directions are not normalized.
    image_orientation = sitk_image.GetDirection()
    # Normalize the directional cosines
    row_cosine = image_orientation[:3] / np.linalg.norm(image_orientation[:3])
    col_cosine = image_orientation[3:6] / np.linalg.norm(image_orientation[3:6])
    slice_cosine = image_orientation[6:] / np.linalg.norm(
        image_orientation[6:])  # Checked that it is the same as np.cross(row_cosine, col_cosine)
    # Get origin (order: positive row, positive col, positive slice axes)
    image_origin = sitk_image.GetOrigin()

    # Construct crs to xyz transform. Notice the order is cols(c), rows(r), slice(s) axes in voxel space to x,y,z (see reference above). The default x,y,z space is LPS.
    crs_to_xyz_transform = np.identity(4, dtype=np.float32)
    crs_to_xyz_transform[:3, 0] = row_cosine * image_spacing[0]
    crs_to_xyz_transform[:3, 1] = col_cosine * image_spacing[1]
    crs_to_xyz_transform[:3, 2] = slice_cosine * image_spacing[2]
    crs_to_xyz_transform[:3, 3] = image_origin
    # # Check if the anatomical co-ordinate is RAS (We don't need to do this check since itk uses LPS already)
    # if series_header['space'] == 'right-anterior-superior':
    #     lps_to_ras = np.diag([-1, -1, 1, 1])
    #     crs_to_xyz_transform = lps_to_ras * crs_to_xyz_transform

    # Return the transform
    return crs_to_xyz_transform


def xyz_to_crs_transform(sitk_image):
    """
    Return the x,y,z patient co-ordinate system to c(column), r(row), s(slice) voxel space transform. The patient co-ordinate
    system could be LPS/RAS depending on the nrrd format.
    The output is an affine transformation matrix
    References:
    https://www.slicer.org/wiki/Coordinate_systems
    https://discourse.slicer.org/t/building-the-ijk-to-ras-transform-from-a-nrrd-file/1513
    https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch4.html
    https://discourse.itk.org/t/are-the-vectors-defining-the-direction-of-a-simpleitk-image-normalized/810

    :param sitk_image : The simpleitk image

    :return: xyz to crs transform

    Note:
    - It is assumed in the below, c,r,s are used for voxel co-ordinate indices along columns, rows and slices axes.
    """

    # Check if the series contains no files
    if not sitk_image:
        raise ValueError('Series header is empty..')

    # # Note required since itk uses LPS
    # # Check if the anatomical co-ordinate is neither of LPS or RAS
    # if series_header['space'] != 'left-posterior-superior' and series_header['space'] != 'right-anterior-superior':
    #     raise ValueError('Series is neither LPS or RAS. Other co-ordinates are currently not supported..')

    # Compute crs to xyz transform
    crs_to_xyz = crs_to_xyz_transform(sitk_image)

    # Compute the inverse matrix to get the the xyz_to_crs transform
    xyz_to_crs = np.linalg.inv(
        crs_to_xyz)  # Will fail if matrix inverse does not exist (e.g. gantry tilted acquisition?)

    # Return the transform
    return xyz_to_crs


def crs_to_xyz(sitk_image, crs_coordinates):
    """
    Return the x,y,z co-ordinates of the voxel space c(col), r(row), s(slice) co-ordinates in the anatomical co-ordinate
    system (LPS/RAS depending on the nrrd format).
    The output is a tuple of x,y,z co-ordinates
    References:
    https://www.slicer.org/wiki/Coordinate_systems
    https://discourse.slicer.org/t/building-the-ijk-to-ras-transform-from-a-nrrd-file/1513
    https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch4.html
    https://discourse.itk.org/t/are-the-vectors-defining-the-direction-of-a-simpleitk-image-normalized/810

    :param sitk_image : Simple itk image containing the image geometry information
    :param crs_coordinates : Numpy matrix of c,r,s co-ordinates along rows, i.e. 1st row is c, 2nd row is r and 3rd row is s

    :return: A tuple of x,y,z co-ordinates in that order

    Note:
    - It is assumed in the below, c,r,s are used for voxel co-ordinate indices along columns, rows and slices axes.
    """

    # Check if the series contains no files
    if (not sitk_image) or (crs_coordinates.size == 0):
        raise ValueError('Series header or crs co-ordinate is empty..')

    # # Check if the anatomical co-ordinate is neither of LPS or RAS (Not required since ITK uses LPS by default)
    # if series_header['space'] != 'left-posterior-superior' and series_header['space'] != 'right-anterior-superior':
    #     raise ValueError('Series is neither LPS or RAS. Other co-ordinates are currently not supported..')

    # Get pixel spacing (in the order x,y,z, i.e. c,r,s, i.e. along columns, along rows and along slices)
    image_spacing = sitk_image.GetSpacing()
    # Get directional cosines (order: positive row axis direction, positive column axis direction, positive slice axis direction)
    # Note that the directions are not normalized.
    image_orientation = sitk_image.GetDirection()
    # Normalize the directional cosines
    row_cosine = image_orientation[:3] / np.linalg.norm(image_orientation[:3])
    col_cosine = image_orientation[3:6] / np.linalg.norm(image_orientation[3:6])
    slice_cosine = image_orientation[6:] / np.linalg.norm(
        image_orientation[6:])  # Checked that it is the same as np.cross(row_cosine, col_cosine)
    # Get origin (order: positive row, positive col, positive slice axes)
    image_origin = sitk_image.GetOrigin()

    # Construct crs to xyz transform. Notice the order is cols(c), rows(r), slice(s) axes in voxel space to x,y,z (see reference above). The default x,y,z space is LPS.
    crs_to_xyz_transform = np.identity(4, dtype=np.float32)
    crs_to_xyz_transform[:3, 0] = row_cosine * image_spacing[0]
    crs_to_xyz_transform[:3, 1] = col_cosine * image_spacing[1]
    crs_to_xyz_transform[:3, 2] = slice_cosine * image_spacing[2]
    crs_to_xyz_transform[:3, 3] = image_origin
    # # Check if the anatomical co-ordinate is RAS (We don't need to do this check since itk uses LPS already)
    # if series_header['space'] == 'right-anterior-superior':
    #     lps_to_ras = np.diag([-1, -1, 1, 1])
    #     crs_to_xyz_transform = lps_to_ras * crs_to_xyz_transform

    # Create an augmented array of c,r,s indices. Notice that the first row in the augmented matrix is the
    # column indices, second row is row indices. This is because the transformation matrix crs_to_xyz expects this order.
    # See the above listed references.
    crs_coordinates_aug = np.empty([4, crs_coordinates[0, :].size],
                                   dtype=np.float32)  # Create a 4 x n vector of co-ordinates
    crs_coordinates_aug[:3, :] = crs_coordinates
    crs_coordinates_aug[3, :] = 1.0

    # Transform voxel co-ordinates to anatomical co-ordinates
    xyz_coordinates = np.matmul(crs_to_xyz_transform, crs_coordinates_aug)

    # Return the x,y,z co-ordinate
    return xyz_coordinates[0, :], xyz_coordinates[1, :], xyz_coordinates[2, :]


def get_voxel_xyz_in_anatomical(series_header):
    """
    Get the x,y,z co-ordinates of the voxels in the anatomical co-ordinate system (LPS/RAS depending on the nrrd format).
    The co-ordinates are computed from the whole volume. The output is a tuple of x,y,z co-ordinates as 3D numpy arrays.
    If the shape of the numpy array of the image volume (shape of the voxel array as well as the sizes dictionary) after
    reading it using pynrrd is (m,n,k), the size of the returned x,y,z arrays are also (m,n,k).
    References:
    https://www.slicer.org/wiki/Coordinate_systems
    https://discourse.slicer.org/t/building-the-ijk-to-ras-transform-from-a-nrrd-file/1513

    :param series_header : Header of the series with volume origin, direction and spacing information

    :return: A tuple of 3D numpy arrays for each of the x,y,z co-ordinates in that order

    Note:
    - It is assumed here that the series is already sorted by slice position.
    - In the reference above, i,j,k are used to refer to columns (c), rows (r) and slices (s) axes in the voxel space.
    """

    # Check if the series contains no files
    if series_header == None:
        raise ValueError('Series header empty..')

    # Check if the anatomical co-ordinate is neither of LPS or RAS
    if series_header['space'] != 'left-posterior-superior' and series_header['space'] != 'right-anterior-superior':
        raise ValueError('Series is neither LPS or RAS. Other co-ordinates are currently not supported..')

    # Construct crs to xyz transform. Notice the order is cols(c), rows(r), slice(s) axes in voxel space to x,y,z (see reference above). The default x,y,z space is LPS.
    crs_to_xyz_transform = np.identity(4, dtype=np.float32)
    crs_to_xyz_transform[:3, 0] = series_header['space directions'][0]
    crs_to_xyz_transform[:3, 1] = series_header['space directions'][1]
    crs_to_xyz_transform[:3, 2] = series_header['space directions'][2]
    crs_to_xyz_transform[:3, 3] = series_header['space origin']
    # Check if the anatomical co-ordinate is RAS
    if series_header['space'] == 'right-anterior-superior':
        lps_to_ras = np.diag([-1, -1, 1, 1])
        crs_to_xyz_transform = lps_to_ras * crs_to_xyz_transform

    # Compute meshgrid for all the 3D voxel space indices. The 3D voxel data read using pynrrd is in the same order as a
    # numpy array. i.e. we can access elements as voxel_array(r,c,s) where r is the row of a slice, c is column and s is
    # the slice index. We construct the 3D voxel space indices using meshgrid to match the size of the voxel data grid.
    r_range = np.arange(series_header['sizes'][0])  # Row co-ordinates
    c_range = np.arange(series_header['sizes'][1])  # Column co-ordinates
    s_range = np.arange(series_header['sizes'][2])  # Slice co-ordinates
    [r_idx, c_idx, s_idx] = np.meshgrid(r_range, c_range, s_range,
                                        indexing='ij')  # Compute meshgrid. Pick ij since we want it in the same order as the voxel values read using pynrrd. They are in the same order as a numpy array i.e. (rows, columns, slice), accessed as numpy_array[row, col, slice]. We want to maintain this since we want to interpolate from the voxel values.

    # Create an augmented array of c,r,s indices of the series. Notice that the first row in the augmented matrix is the
    # column indices, second row is row indices. This is because the transformation matrix crs_to_xyz expects this order.
    # See the above listed references. It is also important to note that the ravel() passes the 3D array as a single list
    # in some order. i.e. pixels are traversed in some specific order when linearizing the 3D array. This pixel order is
    # not affected by listing the column co-ordinates as the first row in the augmented matrix and row co-ordinates as
    # the second row. The final x,y,z after transformation are still in the same order and can be reshaped to the original
    # size, i.e. the size of the r,c,s meshgrid or voxel data grid.
    crs_coordinates = np.empty([4, r_idx.size],
                               dtype=np.float32)  # Note that r_idx.size gives the total number of elements in the r_idx matrix
    crs_coordinates[0, :] = c_idx.ravel()
    crs_coordinates[1, :] = r_idx.ravel()
    crs_coordinates[2, :] = s_idx.ravel()
    crs_coordinates[3, :] = np.ones(r_idx.size)

    # Transform voxel co-ordinates to anatomical co-ordinates
    xyz_coordinates = np.matmul(crs_to_xyz_transform, crs_coordinates)

    # Reshape the arrays into the original grid (#rows, #cols, #slices). This is the same as the shape of the voxel data grid.
    x_coords = xyz_coordinates[0, :].reshape(series_header['sizes'][0], series_header['sizes'][1],
                                             series_header['sizes'][2])
    y_coords = xyz_coordinates[1, :].reshape(series_header['sizes'][0], series_header['sizes'][1],
                                             series_header['sizes'][2])
    z_coords = xyz_coordinates[2, :].reshape(series_header['sizes'][0], series_header['sizes'][1],
                                             series_header['sizes'][2])

    # Return the x,y,z co-ordinates
    return x_coords, y_coords, z_coords


def sample_bounding_box_cube_anatomical(cube_center_xyz, fov, size, res=1):
    """
    Get the x,y,z co-ordinates of the cubic bounding box in the anatomical co-ordinate system (LPS/RAS depending on the nrrd format).
    The output is a tuple of x,y,z co-ordinates as 3D numpy arrays.
    References:
    https://www.slicer.org/wiki/Coordinate_systems
    https://discourse.slicer.org/t/building-the-ijk-to-ras-transform-from-a-nrrd-file/1513

    :param cube_center_xyz : Center of the cube in anatomical co-ordinate system
    :param fov : Field of view of the cube in mm
    :param size : Cube size (Number of grid points along each dimension)
    :param res : Sampling resolution (distance between adjacent grid points along each dimension)

    :return: A tuple of 3D numpy arrays for each of the x,y,z co-ordinates of the cube in that order. The arrays would
    be shaped as #rows, #cols, #slices

    Note:
    - It is assumed here that the series is already sorted by slice position.
    - It is assumed that the bounding box is a cube
    """

    # Check if the input is valid
    if (cube_center_xyz.size == 0) or (not fov) or (not res):
        raise ValueError('Invalid input..')

    # Compute the extents of the x,y,z co-ordinates of the volume
    x_min = cube_center_xyz[0] - fov / 2
    x_max = cube_center_xyz[0] + fov / 2
    y_min = cube_center_xyz[1] - fov / 2
    y_max = cube_center_xyz[1] + fov / 2
    z_min = cube_center_xyz[2] - fov / 2
    z_max = cube_center_xyz[2] + fov / 2

    # Get the range of values along each dimension
    # Using interpixel distance (res)
    # # x co-ordinate range
    # x_range = np.arange(x_min, x_max, res)
    # # y co-ordinate range
    # y_range = np.arange(y_min, y_max, res)
    # # z co-ordinate range
    # z_range = np.arange(z_min, z_max, res)
    #
    # Using size (number of pixels along each side of cube) (size). We chose to use size instead of res since having thes
    # right size is important from a numerical stability point of view.
    # Note that in the below, we include the last point as well. This will make the fov exceed by 1 voxel totally
    # (half voxel on the left and half voxel on the right).
    # x co-ordinate range
    x_range = np.linspace(x_min, x_max, size)
    # y co-ordinate range
    y_range = np.linspace(y_min, y_max, size)
    # z co-ordinate range
    z_range = np.linspace(z_min, z_max, size)

    # Meshgrid to get the x,y,z co-ordinates of the grid points
    [x_coords, y_coords, z_coords] = np.meshgrid(x_range, y_range, z_range,
                                                 indexing='ij')  # We choose ij indexing to be consistent with other places where it is used. This will be consistent with (#rows, #cols, #slices), which is what pynrrd uses.

    # Return the x, y and z co-ordinates of the cube
    return x_coords, y_coords, z_coords



def rotation_from_angle_axis( v, th ):

    """
    Compute rotation matrix from rotation axis and rotation angle. This function essentially computes the rotation
    matrix given the rotation axis and angle of rotation around the axis.
    :param v:
    :param th:
    :return: Rotation matrix
    """

    # Debug
    #v = axis_v
    #th = th

    v = v / np.linalg.norm(v)
    c = np.cos(th)
    s = np.sin(th)
    C = 1-c
    x = v[0]
    y = v[1]
    z = v[2]
    return np.array([(x*x*C+c, x*y*C-z*s, x*z*C+y*s),
                     (y*x*C+z*s, y*y*C+c, y*z*C-x*s),
                     (z*x*C-y*s, z*y*C+x*s, z*z*C+c)])


def rotate_vector_to_vector( v0, v1 ):
    """
    This function essentially computes the rotation matrix to rotate v0 to v1 using the axis angle form.
    :param v0: Initial vector
    :param v1: Final vector
    :return: Rotation matrix
    """

    # Debug
    #v0 = [1.,0.,0.]
    #v1 = antenna_pos_vec

    # Check the validity of the tuple (not done yet)
    v0 = np.array(v0)
    v1 = np.array(v1)

    # Convert the vectors to unit vector
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Cross product of the vectors
    axis_v = np.cross(v0, v1)
    th = np.arccos( np.dot( v0, v1 ) ) # Not unit, length is sin(th)
    return rotation_from_angle_axis( axis_v, th )


def single_applicator_pose(tip, tail):
    """
    Given antenna tip, tail, this function computes the rotation matrix to rotate the vector lying
    on the x axis [1 0 0] to align with the antenna position tail-tip. Using this we can rotation other points to align
    with the antenna.
    :param tip : Antenna tip co-ordinates tuple
    :param tail : Antenna tail co-ordinates tuple
    :return: Rotation matrix to rotate
    """

    # Check if the input is valid
    if (not tip) or (not tail):
        raise ValueError('Invalid input..')

    # Antenna position vector
    antenna_pos_vec = np.array(tail) - np.array(tip)
    # Compute rotation matrix
    return rotate_vector_to_vector( [1.,0.,0.], antenna_pos_vec)


def sample_antenna_centric_bounding_box_in_anatomical(fov, size, antenna_orientation=None, cube_center=None, rot_angle_abt_ant_axis=None, return_cube_at_origin=False):
    """
    Get the x,y,z co-ordinates of the cubic bounding box in the anatomical co-ordinate system (LPS/RAS depending on the
    nrrd format). Note that since we use SimpleITk to read the images, everything is in LPS by default. We assume this in our scripts. 
    The co-ordinates of the cube are aligned with the antenna. The output is a tuple of x,y,z co-ordinates
    as 3D numpy arrays.
    References:
    https://www.slicer.org/wiki/Coordinate_systems
    https://discourse.slicer.org/t/building-the-ijk-to-ras-transform-from-a-nrrd-file/1513

    :param fov : Field of view of the cube in mm
    :param size : Cube size (Number of grid points along each dimension)
    :param antenna_orientation : antenna_orientation[0] = tip of the antenna as a tuple of co-ordinates;
    antenna_orientation[1] = tail of the antenna as a tuple of co-ordinates.
    :param cube_center = The center of the cube in its final position. Note that this was just the tip in earlier work. But later, we offset this from the tip 
    using a set distance along the antenna, towards the tail. This x,y,z co-ordinate is that final position. 
    :param rot_angle_abt_ant_axis : Rotation angle about the antenna axis. If requested, the grid will be rotated by this angle about the antenna axis.
    :param cube_at_origin_only : if only interested in the cube at the origin, no rotation or offset is required. Return the result earlier. Note that for this case, tip, tail doesn't matter.  
    :return: A tuple of 3D numpy arrays for each of the x,y,z co-ordinates of the cube in that order. The arrays would
    be shaped as #rows, #cols, #slices

    Note:
    - It is assumed here that the series is already sorted by slice position.
    - It is assumed that the bounding box is a cube
    """

    # # Testing
    # # Post/followup
    # R87.3, P6.4, I99.1
    # R86.8, A1.4, I97.9
    # R86.8, A6.9, I97.9
    #
    # fov = 80
    # size = 80
    # antenna_orientation = [(-87.3, 6.4, -99.1), (-86.8, -6.9, -97.9)]
    # antenna_orientation = antenna_orientation_origin
    # cube_center = cube_center_origin    


    # Check if the input is valid
    if (not fov) or (not size):
        raise ValueError('Invalid input..')
    
    # Set dimensions and co-ordinates of the grid
    #margin = fov/2
    # Note that we did the below instead because, otherwise, the grid would not be unit sampled (1mm). E.g. when fov = 100, xg = np.linspace(-50, 50, 100) 
    # will yield (-50., -48.98989899, ... 48.98989899,  50.) etc. There are totally 100 points, but sampling is less than 1mm. This is because there are 
    # 101 points if we consider 1mm resolution (-50, -49,..-1, 0, 1, ..., 49, 50). So, we will have to go from -99/2 = 49.5 to 99/2 = 49.5. Now if we do, 
    # xg = np.linspace(-49.5, 49.5, 100), we get (-49.5, -48.5, ... -0.5, 0, 0.5, ..., 48.5, 49.5). There are exactly 100 points, including the end points,
    # 50 on each side of 0. Note that 0 is not included in these configurations, so there is no point corresponding to 0 in the co-ordinate system. For e.g. 
    # the antenna will not be aligned with the 0 axis, it may be half a pixel off. Note that each point is the center of a voxel. Hence number of voxels = 
    # number of points. 100 points = 100 voxels of 1mm width each = 100mm in total fov. This is tallies with the passed in fov. Note however that this works 
    # for both when the fov is even/odd. Also, the fov and the size should be matched for 1mm resolution. i.e. fov = 99mm should be accompanied by size = 99.
    # Finally, careful when the fov and grid size are different values altogether. 
    margin = (fov-1)/2 
    # Write this
    grid_size = np.array([size, size, size])
    # origin = tip - margin
    grid_extent = np.array([2 * margin, 2 * margin, 2 * margin])
    # grid_spacing = grid_extent / grid_size

    # Create a 3D grid of points aligned with the antenna tip as the center and along the antenna axis at the origin.
    # The grid is on the x - y plane and aligned with z axis. It is centered at the origin.
    # Grid extents
    xmin = -grid_extent[0] / 2
    xmax = grid_extent[0] / 2
    ymin = -grid_extent[1] / 2
    ymax = grid_extent[1] / 2
    zmin = -grid_extent[2] / 2
    zmax = grid_extent[2] / 2
    # Grid points
    xg = np.linspace(xmin, xmax, grid_size[0])
    yg = np.linspace(ymin, ymax, grid_size[1])
    zg = np.linspace(zmin, zmax, grid_size[2])
    [x_initial, y_initial, z_initial] = np.meshgrid(xg, yg, zg, indexing='ij')

    xyz_initial_coordinates = np.empty([3, x_initial.ravel().size],
                               dtype=np.float32)  # Note that x_initial.ravel().size gives the total number of elements in the x_initial matrix
    xyz_initial_coordinates[0, :] = x_initial.ravel()
    xyz_initial_coordinates[1, :] = y_initial.ravel()
    xyz_initial_coordinates[2, :] = z_initial.ravel()

    # If only interested in the cube at the origin, no rotation is required. Just return the initial co-ordinates.
    if return_cube_at_origin:
        print('Returning the cube at the origin. The antenna tip and tail co-ordinates do not matter.')
        return xyz_initial_coordinates[0, :], xyz_initial_coordinates[1, :], xyz_initial_coordinates[2, :]

    # Compute bounding box aligned with the antenna since cube_at_origin is false.
    if (antenna_orientation is None) or (cube_center is None):
        raise ValueError('Missing antenna orientation and/or center of cube around the antenna. Exiting...')

    # Extract antenna tip and tail
    tip = antenna_orientation[0]
    tail = antenna_orientation[1]

    # Rotation matrix to align with the antenna
    ant_rotation_matrix = single_applicator_pose(tip, tail)

    # We rotate the grid about the antenna axis if requested by first rotating it about the x axis at the origin and then placing it at the antenna. 
    # This is when we want to generate all possible rotations of the grid around the antenna. After this rotation, the grid is no longer upright along z axis.
    if rot_angle_abt_ant_axis is not None:
        # Construct rotation matrix for rotation about the x axis at the origin
        rot_abt_x_matrix = rotation_from_angle_axis([1.,0.,0.], rot_angle_abt_ant_axis * np.pi / 180.)
        # Rotate the grid points about the x axis
        xyz_initial_coordinates = np.matmul(rot_abt_x_matrix, xyz_initial_coordinates)
    
    # Transform anatomical co-ordinates in the initial position at the origin to the final position along the antenna
    # xyz_antenna_coordinates = np.matmul(rotation_matrix, xyz_initial_coordinates) + np.matlib.repmat(tip, 1, x_initial.ravel().size)
    xyz_antenna_coordinates = np.matmul(ant_rotation_matrix, xyz_initial_coordinates) + cube_center

    # Return the x, y and z co-ordinates of the cube
    #x_cube = np.reshape(interpolated_voxel_array, x_cube.shape)
    return xyz_antenna_coordinates[0, :], xyz_antenna_coordinates[1, :], xyz_antenna_coordinates[2, :]


def crop_and_save_roi(source_img_pair, label_substring, fov, size, dest_path):
    """
    Crop the 3d volume to focus only on the region surrounding the tumor and treatment zones. The roi size and sampling
    resolution are specified as inputs. The output are written out as nii.gz files.

    :param source_img_pair : A list of input files that need to be cropped (either pre scan and tumor segmentation pair
    or followup scan, and treatment zone segmentation pair.
    :param fov : The fov of the sampled cube
    :param size : Size of the cropped cube
    :param dest_path : The destination path where the output files will be written

    :return: None

    Note:
    - It is assumed here that the series is already sorted by slice position.
    - It is assumed that the bounding box is a cube
    """

    # Some sanity check on the arguments (need to be added)

    # Get the label file (note that currently, segmentations in multiple label files are not supported)
    label_index = 0
    if label_substring in os.path.basename(source_img_pair[1]):
        label_index = 1
    # Get the tumor image series index
    series_index = not label_index
    # print(source_img_pair[0])
    # print(source_img_pair[1])

    ## Read image series (note that by default pynrrd returns image in fortran order, i.e. x,y,z, i.e. c,r,s). The usual
    ## access order working with numpy arrays is the opposite c order, i.e. z,y,x, i.e. s,r,c. But here pynrrd return the
    ## numpy array in c,r,s order, unless specified.
    # image_series = nrrd.read(source_img_pair[series_index])
    # # Read the label file in a similar way
    # seg_label = nrrd.read(source_img_pair[label_index])
    # # Check if the anatomical co-ordinate is LPS, if not throw an exception for now. We expect it to be an LPS
    # if image_series[1]['space'] != 'left-posterior-superior' or seg_label[1]['space'] != 'left-posterior-superior':
    #     raise ValueError('Either the image series or label is not LPS. We expect all files to be in LPS..')
    # # Check if the segmentation has only 2 labels (0 and something else)
    # if np.unique(seg_label[0]).size != 2:
    #     raise ValueError('The label file has a problem..')

    # Read the image using Simpleitk (note that simpleitk works in LPS co-ordinate system. Also though simpleitk reads
    # and works in the fortran order (i.e. x,y,z, i.e. c,r,s), the GetArrayFromImage returns a numpy array in the C order
    # i.e. z,y,x, i.e. s,r,c (this is unlike pynrrd). Hence we have to reorder the numpy array to bring it back to fortran
    # order
    # https://discourse.itk.org/t/importing-image-from-array-and-axis-reorder/1192
    image_series = sitk.ReadImage(source_img_pair[series_index])
    image_array = sitk.GetArrayFromImage(image_series)
    image_array = np.transpose(image_array, (2, 1, 0))
    # Read label file in a similar way
    label_series = sitk.ReadImage(source_img_pair[label_index])
    label_array = sitk.GetArrayFromImage(label_series)
    label_array = np.transpose(label_array, (2, 1, 0))

    # The images are already in lps in ITK. We don't have to check that, unlike pynrrd.

    # # This is useful only if interpolating in the patient space using griddata. Commented out currently since we interpolate
    # # in the voxel space.
    # # Get the anatomical co-ordinates of all the voxels of the scan. The x,y,z co-ordinates are in the same order/size
    # # as the voxel data array, which is #rows, #cols, #slices. Hence they can be used to interp.
    # #-------------------------------------------------------------------------------------------------------------------
    # x_anatomical, y_anatomical, z_anatomical = get_voxel_xyz_in_anatomical(source_img_pair[series_index])

    # Find number of connected components in the segmentation/label array. This is accounting for images that have
    # multiple rois like tumors.
    # This connected component analysis was not done afterall. Though there were many connected
    # components in the images when computed using the simpleitk and matlab tools, upon visual inspection, they all seemed
    # to be the same structure (not separate structures). Hence not doing any conn components here. All images seem to have
    # only 1 tumor segmented if they had multiple. As a first step, assume there is only one component. Checked a lot of
    # them and they all have only a single component

    # Construct a cube in the world space around the center of the tumor. The cube is oriented along the z in the anatomical
    # co-ordinate system.
    # -------------------------------------------------------------------------------------------------------------------
    # Get the voxel indices that are non zero in the label. Note that np.where extracts row co-ordinates as first array,
    # the column co-ordinates as the second array and the slice co-ordinates as the third array.
    seg_indices = list(np.where(label_array != 0))
    # seg_indices[0], seg_indices[1] = seg_indices[1], seg_indices[0] Not required

    # Get the mean voxel co-ordinate to get the center of the tumor (we actually need this in the world space. But since
    # matrix multiplication is a linear operation, we can do it in pixel space and then transform the pixel space mean to
    # world space). Note that the tumor center can be outside the tumor.
    seg_mean_rcs = np.array([sum(seg_indices[0]) / len(seg_indices[0]), sum(seg_indices[1]) / len(seg_indices[1]),
                             sum(seg_indices[2]) / len(seg_indices[2])])

    # Get the world co-ordinate corresponding to the center voxel co-ordinate. Note that the we are arranging the matrix
    # in c(col), r(row), s(slice) order since this is the order required by the voxel to world transformation.
    seg_mean_crs_vec = np.ones((3, 1), dtype=np.float32)
    seg_mean_crs_vec[0, 0] = seg_mean_rcs[1]
    seg_mean_crs_vec[1, 0] = seg_mean_rcs[0]
    seg_mean_crs_vec[2, 0] = seg_mean_rcs[2]
    seg_mean_xyz = crs_to_xyz(image_series, seg_mean_crs_vec)

    # The below will probably go inside a loop for different bounding boxes
    # -----------------------------------------------------------------------------------------
    # Sample a bounding box in the anatomical co-ordinate system centered at the mean x,y,z of the segmentation.
    # For now, we have decided to keep it at 5cm x 5cm x 5cm, 8cm x 8cm x 8cm, 10cm x 10cm x 10cm size and 15cm x 15cm x 15cm sizes
    # seg_mean_xyz is currently a list of numpy arrays. Convert it into a plain numpy array.
    seg_mean_xyz = np.concatenate(seg_mean_xyz, axis=0)

    # Sample the cubic bounding box in the anatomical space
    x_cube, y_cube, z_cube = sample_bounding_box_cube_anatomical(seg_mean_xyz, fov, size)

    # # Directly interpolating in the patient space requires griddata, which is extremely slow for some reason. The RegularGridInterpolant
    # # doesn't work for this case since it requires vectors of x,y,z co-ordinates of the source grid, which means it has to be
    # # a rectangular grid. An oblique grid cannot be fully specified with just the vectors of x,y,z points.
    # # Interpolate from series onto the cube
    # voxel_standard = griddata((x_anatomical.ravel(), y_anatomical.ravel(), z_anatomical.ravel()), image_series[0].ravel(), (x_cube, y_cube, z_cube), method='linear', fill_value=np.amin(image_series[0]))

    # Interpolate in the voxel space. This will make it possible to use the RegularGridInterpolant since the pixel grid
    # is regular. We will have to take all points in the patient space to voxel space using the inverse affine transformation.
    # This will only work if the inverse exists. Compute the xyz to crs transformation matrix. Also note the final order
    # of the voxel dimensions c,r,s.
    xyz_to_crs = xyz_to_crs_transform(image_series)

    # Apply the transformation on the x,y,z points of the cube to transform them to voxel space.
    # Create an augmented array of x,y,z indices of the series. The x,y,z points are stored in the same order as the voxel value
    # array read using pynrrd.
    xyz_coordinates = np.empty([4, x_cube.ravel().size],
                               dtype=np.float32)  # Note that x_cube.ravel().size gives the total number of elements in the x_cube matrix
    xyz_coordinates[0, :] = x_cube.ravel()
    xyz_coordinates[1, :] = y_cube.ravel()
    xyz_coordinates[2, :] = z_cube.ravel()
    xyz_coordinates[3, :] = np.ones(x_cube.ravel().size)
    # Transform anatomical co-ordinates to voxel co-ordinates. Also note the final order
    # of the voxel dimensions c,r,s.
    crs_coordinates = np.matmul(xyz_to_crs, xyz_coordinates)

    # Compute meshgrid for all the 3D voxel space indices. The 3D voxel data read using pynrrd is in the same order as a
    # numpy array. i.e. we can access elements as voxel_array(r,c,s) where r is the row of a slice, c is column and s is
    # the slice index. We construct the 3D voxel space indices using meshgrid to match the size of the voxel data grid.
    # NOTE: This shuold probably be col, row, slice, to be consistent with fortran ordering that we have maintained throughout.
    # But here #rows is same as #cols and will be ok. Raising an exception if not.
    if image_series.GetSize()[0] != image_series.GetSize()[1]:
        raise ValueError('The number of rows and columns are not the same. This is strange. Exiting..')
    r_range = np.arange(image_series.GetSize()[0])  # Row co-ordinates
    c_range = np.arange(image_series.GetSize()[1])  # Column co-ordinates
    s_range = np.arange(image_series.GetSize()[2])  # Slice co-ordinates
    ## This is not required for regular grid interpolation
    # [r_idx, c_idx, s_idx] = np.meshgrid(r_range, c_range, s_range, indexing='ij')  # Compute meshgrid. Pick ij since we want it in the same order as the voxel values read using pynrrd. They are in the same order as a numpy array i.e. (rows, columns, slice), accessed as numpy_array[row, col, slice]. We want to maintain this since we want to interpolate from the voxel values.

    try:
        # Interpolate the cube voxel values in the voxel space from the standard voxel grid. This can be done using regular
        # grid interpolation and is much faster than doing the interpolation in the patient space using grid data.
        intensity_interpolating_function = RegularGridInterpolator((r_range, c_range, s_range), image_array,
                                                                   bounds_error=False, fill_value=np.amin(
                image_array))  # I think this is the way to specify interpolation to extrapolate outside points
        # Interpolate the label. Notice that the label is converted to binary (0-1) before interpolation.
        label_interpolating_function = RegularGridInterpolator((r_range, c_range, s_range), label_array > 0,
                                                               bounds_error=False,
                                                               fill_value=0)  # fill_value is 0 since lthe min intensity of abel_array > 0 will be 0; I think this is the way to specify interpolation to extrapolate outside points
    except Exception as e:
        print('Interpolation failed')
        return

    # Query points (dstack used here to extract each c,r,s co-ordinates of the cube voxels in the voxel space as a list).
    # Note that the second row is the first dimension because the xyz_to_crs conversion has the voxel co-ordinates in
    # the c,r,s order. Whereas the original grid data in the voxel space is in the r,c,s order. Hence swapping dimensions
    # 0 and 1 here.
    query_points = np.dstack((crs_coordinates[1, :], crs_coordinates[0, :], crs_coordinates[2, :]))
    interpolated_voxel_array = intensity_interpolating_function(query_points, method='linear')
    interpolated_label_array = label_interpolating_function(query_points,
                                                            method='nearest')  # Nearest for labels seems the common way for label interpolation
    # interpolated_label_array = label_interpolating_function(query_points, method='linear')
    # Reshape into a voxel grid
    cube_voxel_array = np.reshape(interpolated_voxel_array, x_cube.shape)
    cube_label_array = np.reshape(interpolated_label_array, x_cube.shape)
    # Convert the segmentation mask into binary (0-1) mask. This will take care of converting the fractional values at
    # the edges after interpolation from a binary mask.
    # cube_label_array = cube_label_array > 0.5 # This is not finalized
    cube_label_array = cube_label_array.astype('uint8', copy=False)
    # # Write to nrrd file.
    # nrrd.write('output_intensity.nrrd', cube_voxel_array)
    # nrrd.write('output_label.nrrd', cube_label_array)
    # Write to nifti file.
    cube_voxel_array_nib = nib.Nifti1Image(cube_voxel_array, np.eye(4))
    cube_label_array_nib = nib.Nifti1Image(cube_label_array, np.eye(4))

    # Write to nifti files
    # nib.save(cube_voxel_array_nib, os.path.join(dest_path, 'post_img.nii.gz'))
    # nib.save(cube_label_array_nib, os.path.join(dest_path, 'post_seg.nii.gz'))
    nib.save(cube_voxel_array_nib, os.path.join(dest_path, os.path.basename(source_img_pair[series_index])))
    nib.save(cube_label_array_nib, os.path.join(dest_path, os.path.basename(source_img_pair[label_index])))

    # if type == 'tumor':
    #     nib.save(cube_voxel_array_nib, os.path.join(dest_path, 'tumor.nii.gz'))
    #     nib.save(cube_label_array_nib, os.path.join(dest_path, 'tumor-label.nii.gz'))
    # elif type == 'followup':
    #     nib.save(cube_voxel_array_nib, os.path.join(dest_path, 'followup.nii.gz'))
    #     nib.save(cube_label_array_nib, os.path.join(dest_path, 'followup-label.nii.gz'))
    # else:
    #     ValueError('Unknown type')


# REMOVE ME
def crop_and_save_roi_along_antenna(source_img, fov, size, antenna_orientation, case_name, output_data_path, num_rotations = 1, type='image_series', mask='sphere', all_rotations=False, vendor_model=None, modality_identifiers=None):
#def crop_and_save_roi_along_antenna(source_img, fov, size, antenna_orientation, case_name, output_data_path, num_rotations = 36, type='image_series', mask='sphere', all_rotations=False, vendor_model=None, modality_identifiers=None):
    """
    Crop the 3d volume to focus only on the region surrounding the roi along the antenna, 
    and followup treatment zones. The roi size, sampling resolution, center of the cropped region (e.g. tip of the
    antenna) and orientation of the antenna are specified as inputs. The output is written out as nii.gz files with an identity affine matrix. 
    This assumes that the bounding box has cubical voxels with 1mm x 1mm x 1mm size. The distance between voxels along any dimension is 1mm. 
    An identify affine matrix puts the bounding box in alignment with the world co-ordinate systems.

    :param source_img : Input file that need to be cropped (e.g. followup scan.
    :param fov : The fov of the sampled cube
    :param size : Size of the cropped cube
    :param antenna_orientation : antenna_orientation[0] = tip; antenna_orientation[1] = tail of the antenna. Each a tuple of LPS co-ordinates.
    :param case_name : Name of the case. This is used to construct the case identifier string. 
    :param modality_identifier : A 4 numbered string starting from '0000', for identifying the modality. This is used to construct the data file name.
    :param output_data_path : The destination path where the output files will be written
    :param num_rotations : Total number of rotations of the cube to be considered around the antenna. By default, we consider 36 rotations, i.e. every 10 deg.
    It seems enough sampling resolution. Note that the number of rotations specified has to be able to divide 360 exactly. i.e. the rotation angle resolution
    is required to be an integer by the current version of the code. 
    :param type : Type of the input image: could be 'image_series' (gray scale images) or 'seg_label' (binary images) 
    :param all_rotations : Flag to indicate if to compute bounding boxes across al rotations from 0 to 360 deg around the antenna or just one rotation at 0 deg w.r.t antenna axis. 
    :return: None

    Note:
    - It is assumed here that the series is already sorted by slice position.
    - It is assumed that the bounding box is a cube
    - param num_rotations: The reason the degree resolution has to be an integer is, when we write to nifty files, it seems the name of the image is set to 
    the string until the first encountered period. When we have float, e.g. 'subj41_rotAngle_350.0deg_0001.nii.gz', nifti sets 'subj41_rotAngle_350' as the 
    image name. The problem with this is, for a different modality (channel e.g. vendor model), e.g., 'subj41_rotAngle_350.0deg_0002.nii.gz', nifti will 
    set the image name to still be 'subj41_rotAngle_350'. This can be confusing for visualization or downstream algorithms that consume this data. Hence, to 
    avoid this, we can either restrict the degrees to only ints or use 'point' instead of period, if we want to denote floats. We used the former approach
    here and hence only integer degrees are allowed. 
    """

    # Some sanity check on the arguments (need to be added)

    # # Get the label file (note that currently, segmentations in multiple label files are not supported)
    # label_index = 0
    # if label_substring in os.path.basename(source_img_pair[1]):
    #     label_index = 1
    # # Get the tumor image series index
    # series_index = not label_index
    # # print(source_img_pair[0])
    # # print(source_img_pair[1])

    ## Read image series (note that by default pynrrd returns image in fortran order, i.e. x,y,z, i.e. c,r,s). The usual
    ## access order working with numpy arrays is the opposite c order, i.e. z,y,x, i.e. s,r,c. But here pynrrd return the
    ## numpy array in c,r,s order, unless specified.
    # image_series = nrrd.read(source_img_pair[series_index])
    # # Read the label file in a similar way
    # seg_label = nrrd.read(source_img_pair[label_index])
    # # Check if the anatomical co-ordinate is LPS, if not throw an exception for now. We expect it to be an LPS
    # if image_series[1]['space'] != 'left-posterior-superior' or seg_label[1]['space'] != 'left-posterior-superior':
    #     raise ValueError('Either the image series or label is not LPS. We expect all files to be in LPS..')
    # # Check if the segmentation has only 2 labels (0 and something else)
    # if np.unique(seg_label[0]).size != 2:
    #     raise ValueError('The label file has a problem..')

    
    # Read the image using Simpleitk (note that simpleitk works in LPS co-ordinate system). 
    #----------------------------------------------------------------------------------------------------------
    # Note:
    # 1) The co-ordinate system of ITK is LPS. The data in SimpleITK Image is stored in the (x,y,z) order. i.e. The top left hand corner of the image (each slice) is 
    # the origin, the x axis increases to right, y axis increases towards bottom, and z goes inside the screen (it is the slice index). So, simple ITK Image(20,30)
    # means 20 units along columns (x axis), 30 units along rows (y axis). 
    # 2) Numpy stores a multi-dimensional array in the r (row), c (column).
    # 3) Notice that ITK and numpy store multi dimensional data in different orders. For e.g. the numpy(20,30) would be 20 units along rows (y axis), 30 units
    # along columns (x axis)
    # 4) Finally, for 3D arrays, the order in ITK/SimpleITK is x (columns), y (rows), z (slices), whereas in numpy, it is s (z axis), r (y axis), c (x axis). The
    # slice axis swaps places.
    # Though simpleitk reads and works in the fortran order (i.e. x,y,z, i.e. columns, rows, slices), the GetArrayFromImage returns a numpy array in the C order
    # i.e. z,y,x, i.e. slices, rows, columns (this is unlike pynrrd). Hence we have to reorder the numpy array to bring it back to fortran order, since that is 
    # the convention we have adopted in this work. 
    # https://discourse.itk.org/t/importing-image-from-array-and-axis-reorder/1192
    image_series = sitk.ReadImage(source_img)
    image_array = sitk.GetArrayFromImage(image_series)
    image_array = np.transpose(image_array, (2, 1, 0))
    #image_array = np.transpose(image_array, (1, 2, 0)) # In earlier work, this was enabled. Not sure why. This is not what is suggested in the link.
    # # Read label file in a similar way
    # label_series = sitk.ReadImage(source_img_pair[label_index])
    # label_array = sitk.GetArrayFromImage(label_series)
    # label_array = np.transpose(label_array, (2, 1, 0))

    # The images are already in lps in ITK. We don't have to check that, unlike pynrrd.

    # # This is useful only if interpolating in the patient space using griddata. Commented out currently since we interpolate
    # # in the voxel space.
    # # Get the anatomical co-ordinates of all the voxels of the scan. The x,y,z co-ordinates are in the same order/size
    # # as the voxel data array, which is #rows, #cols, #slices. Hence they can be used to interp.
    # #-------------------------------------------------------------------------------------------------------------------
    # x_anatomical, y_anatomical, z_anatomical = get_voxel_xyz_in_anatomical(source_img_pair[series_index])

    # Find number of connected components in the segmentation/label array. This is accounting for images that have
    # multiple rois like tumors.
    # This connected component analysis was not done afterall. Though there were many connected
    # components in the images when computed using the simpleitk and matlab tools, upon visual inspection, they all seemed
    # to be the same structure (not separate structures). Hence not doing any conn components here. All images seem to have
    # only 1 tumor segmented if they had multiple. As a first step, assume there is only one component. Checked a lot of
    # them and they all have only a single component

    # # Construct a cube in the world space around the center of the tumor. The cube is oriented along the z in the anatomical
    # # co-ordinate system.
    # # -------------------------------------------------------------------------------------------------------------------
    # # Get the voxel indices that are non zero in the label. Note that np.where extracts row co-ordinates as first array,
    # # the column co-ordinates as the second array and the slice co-ordinates as the third array.
    # seg_indices = list(np.where(label_array != 0))
    # # seg_indices[0], seg_indices[1] = seg_indices[1], seg_indices[0] Not required
    #
    # # Get the mean voxel co-ordinate to get the center of the tumor (we actually need this in the world space. But since
    # # matrix multiplication is a linear operation, we can do it in pixel space and then transform the pixel space mean to
    # # world space). Note that the tumor center can be outside the tumor.
    # seg_mean_rcs = np.array([sum(seg_indices[0]) / len(seg_indices[0]), sum(seg_indices[1]) / len(seg_indices[1]),
    #                          sum(seg_indices[2]) / len(seg_indices[2])])
    #
    # # Get the world co-ordinate corresponding to the center voxel co-ordinate. Note that the we are arranging the matrix
    # # in c(col), r(row), s(slice) order since this is the order required by the voxel to world transformation.
    # seg_mean_crs_vec = np.ones((3, 1), dtype=np.float32)
    # seg_mean_crs_vec[0, 0] = seg_mean_rcs[1]
    # seg_mean_crs_vec[1, 0] = seg_mean_rcs[0]
    # seg_mean_crs_vec[2, 0] = seg_mean_rcs[2]
    # seg_mean_xyz = crs_to_xyz(image_series, seg_mean_crs_vec)

    # The below will probably go inside a loop for different bounding boxes
    # -----------------------------------------------------------------------------------------
    # # Sample a bounding box in the anatomical co-ordinate system centered at the mean x,y,z of the segmentation.
    # # For now, we have decided to keep it at 5cm x 5cm x 5cm, 8cm x 8cm x 8cm, 10cm x 10cm x 10cm size and 15cm x 15cm x 15cm sizes
    # # seg_mean_xyz is currently a list of numpy arrays. Convert it into a plain numpy array.
    # seg_mean_xyz = np.concatenate(seg_mean_xyz, axis=0)


    # Patient to voxel space transform for interpolation in the voxel space
    # -----------------------------------------------------------------------  
    # Interpolate in the voxel space. This will make it possible to use the RegularGridInterpolant since the pixel grid
    # is regular. We will have to take all points in the patient space to voxel space using the inverse affine transformation.
    # This will only work if the inverse exists. Compute the xyz to crs transformation matrix. Also note the final order
    # of the voxel dimensions c (columns), r (rows), s (slices).
    xyz_to_crs = xyz_to_crs_transform(image_series)


    # Sample bounding box around antenna, interpolate from the image, retrict the values to a sphere and write them to file
    #---------------------------------------------------------------------------------------------------------------------------
    if all_rotations:
        # If true, do this for all rotation about the antenna axis.
        rot_angles_abt_ant_axis = np.linspace(0, 360, num_rotations+1)[:-1] # Only when we specify num_rotations+1 rotations (e.g. 36+1 = 37) do we get rotations at whole integer angles (i.e. 0 deg, 10 deg, 20 deg.. 350 deg etc..). But now the (num_rotations+1)th rotation (e.g. 37th) is the same as 0th rotation, hence considering only until num_rotationth (36th) rotation. 
    else:
        # Do it for only 1 rotation, e.g. 0 degrees.
        rot_angles_abt_ant_axis = np.array([0])

    # Set center of the cube
    # The tip was set as the center of the ablation bounding box in our previous work. Here we offset it by 1 cm from the tip, such that the center of the 
    # cube is to the left of the tip when looking down from the base of the antenna to towards the tip. The reason we do this is, ablation zones can be large 
    # to the left of the tip comapred to the right. At least this is what the vendor model seems to suggest. In some real cases, the ablation zone beyond the tip is 
    # huge as well. Hence, we set the center of the cube/bounding box to some intermediate value, leaving enough room for both left (proximal end of the ablation zone) 
    # of the tip and to the right of the tip (distal end of the ablation zone) to contain the ablation zone fully. 
    # center the ablation zone
    tip = antenna_orientation[0] # Tip
    tail = antenna_orientation[1] # Tail
    # Offset length along the antenna 
    offset_length = fov/4 # This was seen to cover all sizes of fov well. May be some scope for improvement for 32 and 48, could be slightly further right. i.e. increase offset.
    # Center offset added to tip. The center is moved along the antenna, from the tip towards the back of the antenna.
    center_offset = ((np.array(tail) - np.array(tip)) / np.linalg.norm(np.array(tail) - np.array(tip))) * offset_length # We move 10mm from tip towards the back of the antenna
    #cube_center = np.transpose(np.array([(tip)])) + np.transpose(np.array([((0,-10,0))])) # This approach was wrong since we were moving the center first at the origin by (0,-10,0) and then adding the tip co-ordinates. This would result in a final bounding box not centered at the antenna anymore since the resulting vector additino is a parallelogram.
    cube_center = np.transpose(np.array([(tip)])) + np.transpose(np.array([(center_offset)])) 

    # Iterate through all rotations, compute bounding boxes and svae them to file
    for rot_angle in rot_angles_abt_ant_axis:

        #Debug
        #rot_angle = rot_angles_abt_ant_axis[0]

        # Sample the cubic bounding box in the anatomical space
        #-------------------------------------------------------------------
        # Extract the cube co-ordinates. x_cube, y_cube and z_cube are already vectors and not ordered. To order them into a
        # cube, reshape it accordingly.

        # Get bounding box co-ordinates
        x_cube, y_cube, z_cube = sample_antenna_centric_bounding_box_in_anatomical(fov, size, antenna_orientation=antenna_orientation, cube_center=cube_center, rot_angle_abt_ant_axis=rot_angle)

        # # Directly interpolating in the patient space requires griddata, which is extremely slow for some reason. The RegularGridInterpolant
        # # doesn't work for this case since it requires vectors of x,y,z co-ordinates of the source grid, which means it has to be
        # # a rectangular grid. An oblique grid cannot be fully specified with just the vectors of x,y,z points.
        # # Interpolate from series onto the cube
        # voxel_standard = griddata((x_anatomical.ravel(), y_anatomical.ravel(), z_anatomical.ravel()), image_series[0].ravel(), (x_cube, y_cube, z_cube), method='linear', fill_value=np.amin(image_series[0]))

        # Apply the transformation on the x,y,z points of the cube to transform them to voxel space of image_series.
        # Create an augmented array of x,y,z indices of the series. The x,y,z points are stored in the same order as the voxel value
        # array read using pynrrd.
        xyz_coordinates = np.empty([4, x_cube.ravel().size],
                                dtype=np.float32)  # Note that x_cube.ravel().size gives the total number of elements in the x_cube matrix
        xyz_coordinates[0, :] = x_cube.ravel()
        xyz_coordinates[1, :] = y_cube.ravel()
        xyz_coordinates[2, :] = z_cube.ravel()
        xyz_coordinates[3, :] = np.ones(x_cube.ravel().size)
        # Transform anatomical co-ordinates to voxel co-ordinates. Also note the final order
        # of the voxel dimensions c (column), r (row), s (slice).
        crs_coordinates = np.matmul(xyz_to_crs, xyz_coordinates)

        # Compute meshgrid for all the 3D voxel space indices. The 3D voxel data read using pynrrd is in the same order as a
        # numpy array. i.e. we can access elements as voxel_array(r,c,s) where r is the row of a slice, c is column and s is
        # the slice index. We construct the 3D voxel space indices using meshgrid to match the size of the voxel data grid.
        # NOTE: This should probably be col, row, slice, to be consistent with fortran ordering that we have maintained throughout.
        # But here #rows is same as #cols and will be ok. Raising an exception if not.
        #------------------------------------------------------------------------------------------------------
        # THE BELOW IF CONDITION WAS EXCEPTING SINCE AFTER CROPPING THE SCANS TO 100MM RADIUS SPHERES FOR REGISTRATION, THE CONDITION IS NOT TRUE ANYMORE. 
        # IT WOULD BE TRUE FOR SCANNER IMAGES, BUT NOT FOR CROPPED IMAGES WHEN THE CROP IS NOT ALIGNED WITH THE RECTANGULAR X,Y,Z CO-ORDINATE SYSTEM OF THE SCAN
        # HENCE COMMETING THIS LINE. AND WE ALSO CORRECTED THE ORDERING TO columns, rows, slices AS SUGGESTED. THE QUERY IS IN c, r, s ORDER AND HENCE THE GRID 
        # CO-ORDINATES FROM WHICH WE ARE INTERPOLATING NEEDS TO ALSO BE IN THE SAME ORDER. FURTHER, THE IMAGE ARRAY ITSELF HAS ALREADY BEEN ORDERED TO columns, rows, and slices IN
        # THE BEGINNING. SO, THE RegularGrigInterpolation FUNCTION NEEDS THE CO-ORDINATES IN THE SAME ORDER AS WELL. HENCE COMMENTING THE BELOW CODE AND CORRECTING 
        # FOR ORDER. 
        # if image_series.GetSize()[0] != image_series.GetSize()[1]:
        #     raise ValueError('The number of rows and columns are not the same. This is strange. Exiting..')
        #r_range = np.arange(image_series.GetSize()[0])  # Row co-ordinates
        #c_range = np.arange(image_series.GetSize()[1])  # Column co-ordinates
        #s_range = np.arange(image_series.GetSize()[2])  # Slice co-ordinates
        # Because the image_array has already been re-ordered into c/x, r/y, s/z order.
        c_range = np.arange(image_series.GetSize()[0])  # x/Column co-ordinates
        r_range = np.arange(image_series.GetSize()[1])  # y/Row co-ordinates
        s_range = np.arange(image_series.GetSize()[2])  # z/Slice co-ordinates

        ## This is not required for regular grid interpolation
        # [r_idx, c_idx, s_idx] = np.meshgrid(r_range, c_range, s_range, indexing='ij')  # Compute meshgrid. Pick ij since we want it in the same order as the voxel values read using pynrrd. They are in the same order as a numpy array i.e. (rows, columns, slice), accessed as numpy_array[row, col, slice]. We want to maintain this since we want to interpolate from the voxel values.
        print('Before interpolation')
        try:
            if type == 'image_series':
                # Interpolate the cube voxel values in the voxel space from the standard voxel grid. This can be done using regular
                # grid interpolation and is much faster than doing the interpolation in the patient space using grid data.
                #interpolating_function = RegularGridInterpolator((r_range, c_range, s_range), image_array,
                #                                                           bounds_error=False, fill_value=np.amin(
                #        image_array))  # I think this is the way to specify interpolation to extrapolate outside points    
                # This was changed to -1024 from the min value of the image array. It is likely the same, except when the image is cropped. Hence setting this explicitly. Also, later 
                # we extract a sphere and set the cube pixels outside to -1024. Hence the unimportant pixels will be consistent.
                #interpolating_function = RegularGridInterpolator((r_range, c_range, s_range), image_array,
                #                                                        bounds_error=False, fill_value=-1024)  #I think this is the way to specify interpolation to extrapolate outside points
                # THE ABOVE WAS COMMENTED KEEPING CONSISTENT WITH THE column, row, slice order.
                interpolating_function = RegularGridInterpolator((c_range, r_range, s_range), image_array,
                                                                        bounds_error=False, fill_value=-1024)  #I think this is the way to specify interpolation to extrapolate outside points

            elif type == 'seg_label':
                # Interpolate the label. Notice that the label is converted to binary (0-1) before interpolation.
                #interpolating_function = RegularGridInterpolator((r_range, c_range, s_range), image_array > 0,
                #                                                    bounds_error=False,
                #                                                    fill_value=0)  # fill_value is 0 since the min intensity of abel_array > 0 will be 0; I think this is the way to specify interpolation to extrapolate outside points
                # THE ABOVE WAS COMMENTED KEEPING CONSISTENT WITH THE column, row, slice order.
                interpolating_function = RegularGridInterpolator((c_range, r_range, s_range), image_array > 0,
                                                                    bounds_error=False,
                                                                    fill_value=0)  # fill_value is 0 since the min intensity of abel_array > 0 will be 0; I think this is the way to specify interpolation to extrapolate outside points
            else:
                raise ValueError('Image type is invalid')
        except Exception as e:
            #error_msg = 'Interpolation failed: ErrorName: {c}, Message: {m}'.format(
            #    c=type(e).__name__, m=str(e))
            print('Interpolation error')
            print(e)
            return

        # Query points (dstack used here to extract each c,r,s co-ordinates of the cube voxels in the voxel space as a list).
        # Note that the second row is the first dimension because the xyz_to_crs conversion has the voxel co-ordinates in
        # the c,r,s order. Whereas the original grid data in the voxel space is in the r,c,s order. Hence swapping dimensions
        # 0 and 1 here.
        #query_points = np.dstack((crs_coordinates[1, :], crs_coordinates[0, :], crs_coordinates[2, :]))
        query_points = np.dstack((crs_coordinates[0, :], crs_coordinates[1, :], crs_coordinates[2, :]))
        if type == 'image_series':
            interpolated_array = interpolating_function(query_points, method='linear')
        elif type == 'seg_label':
            interpolated_array = interpolating_function(query_points,
                                                                method='nearest')  # Nearest for labels seems the common way for label interpolation
        else:
            raise ValueError('Image type is invalid')
        # interpolated_label_array = label_interpolating_function(query_points, method='linear')

        # Reshape into a voxel grid of the specified size
        if type == 'image_series':
            interpolated_array_cube = np.reshape(interpolated_array, (size, size, size))
        elif type == 'seg_label':
            interpolated_array_cube = np.reshape(interpolated_array, (size, size, size))
        else:
            raise ValueError('Image type is invalid')
        # Debug
        #plt.imshow(interpolated_array_cube[:,:,40])

        # Create vendor model in the same space, i.e., the space of the antenna grid, if specified. This is the space of the antenna co-ordinates since the scan has already 
        # been sampled in the antenna centric grid. Creating the vendor model is easier in this grid since it is rectangular and at the origin. Also.
        # ellipsoid equations at the origin, lying along the x-axis, are way simpler. The antenna is lying on the x axis as well. 
        # Hence we create it after sampling the image along the antenna centric grid rather than before. If we were to do it before, we would first 
        # have to create the vendor model around the antenna as a binary image and sample from it. But this is just double the work and redundant since
        # we finally want to come back to antenna centric grid. 
        if vendor_model is not None:
            # Create a cube at the origin of the antenna centric grid with the vendor model lying along the x axis (the axis along which the antenna is lying) 
            # To use the sample_antenna function, we need to pass corresponding points, which are slighly different from the inputs for sampling the antenna centric
            # grid in the first place. E.g. we just need the antenna centric grid sampled at the origin, and not require it to rotate in any angle since the antenna is 
            # now lying at the origin, along the x axis. We don't need to offset it in any way well.
            #
            # Debug
            # antenna_orientation_origin = [(0.,0.,0.), (1.,0.,0.)] # Tip, tail, antenna is lying on the x axis.
            # cube_center_origin = np.array([[0.0],[0.0],[0.0]]) # No need to add any offset since want to create the grid at the origin.
            # # Compute the co-ordinates of the antenna centric bounding box at the origin.
            # # Debug
            # fov = 96
            # size = 96
            # # subj80
            # vendor_model = {}
            # vendor_model['length'] = 28
            # vendor_model['diameter'] = 17
            # vendor_model['dist_past_tip'] = 2
            # offset_length = 10

            # Extract antenna centric grid at the origin.
            x_cube, y_cube, z_cube = sample_antenna_centric_bounding_box_in_anatomical(fov, size, return_cube_at_origin=True)
            # Vendor ellipsoid parameters. Note that the ellipsoid is lying along the x-axis.
            ellip_a = vendor_model['length'] / 2
            ellip_b = vendor_model['diameter'] / 2
            ellip_c = vendor_model['diameter'] / 2
            # Ellipsoid center x-coordinate, if we assume the tip is the origin (this was what we did in our previous work, but the ablation zone itself was not centered in the antenna centric grid. The problem is, the ablation zone is not not fully inside the bounding box sometimes. Hence not enough, we need to offset it some more).
            ellip_center_x_if_tip_origin = -1. * (ellip_a - vendor_model['dist_past_tip']) 
            # Add to the above, an addition offset to make sure the ellipsoid is approximately centered w.r.t the antenna centric bounding box.
            ellip_center_x = ellip_center_x_if_tip_origin + offset_length  
            # Go through all x,y,z points and create an image with 1s inside the ellipsoid and 0s outside 
            # Create a vector for the vendor model intensities inside the bounding box at the origin
            vendor_model_intensities = []
            for idx in range(x_cube.size):
                # # Check if point is inside vendor ellipsoid and if so, set to 1. Otherwise set to 0. 
                # NOTE: The (x_cube[idx] + ellip_center_x) must have been (x_cube[idx] - ellip_center_x), according to my understanding. i.e. we are moving 
                # the ablation zone the left of origin if ellip_center_x is negative (which is the case if it is just ellip_center_x_if_tip_origin). But that 
                # didn't match and it went in the opposite direction for some reason. This was true even after adding offset. Hence we  switched it addition
                # instead of subtraction and everything aligns well now.
                if ((x_cube[idx] + ellip_center_x)**2 / ellip_a**2) + (y_cube[idx]**2 / ellip_b**2) + (z_cube[idx]**2 / ellip_c**2) <= 1:
                    vendor_model_intensities.append(1)
                # We were just testing various combinations below. 
                # # Check if point is inside vendor ellipsoid and if so, set to 1. Otherwise set to 0.
                # if ((x_cube[idx])**2 / ellip_a**2) + ((y_cube[idx] - ellip_center_x)**2 / ellip_b**2) + (z_cube[idx]**2 / ellip_c**2) <= 1:
                #     vendor_model_intensities.append(1)
                # Check if point is inside vendor ellipsoid and if so, set to 1. Otherwise set to 0.
                #if ((x_cube[idx])**2 / ellip_a**2) + ((y_cube[idx])**2 / ellip_b**2) + (z_cube[idx]**2 / ellip_c**2) <= 1:
                #    vendor_model_intensities.append(1)
                else:
                    vendor_model_intensities.append(0)
            # Convert the vendor model intensities into a numpy array.
            # NOTE: We made sure to keep reshape in the same way we did with the interpolated image array above. interpolated_array is of size (1,N), where N
            # is the number of voxels. To be consistent, we also did first create a (1, N) numpy array, copy the list of vendor model into it, and reshape it in 
            # the same way. It might still work even if we just did vendor_model_intensities = np.array(vendor_model_intensities). Nothing else was required to 
            # obtain the right ordering.
            vendor_model_intensities_array = np.zeros((1, len(vendor_model_intensities)), dtype=np.uint8)
            vendor_model_intensities_array[0, :] = vendor_model_intensities
            #vendor_model_intensities = np.array(vendor_model_intensities, dtype=np.uint8)
            # Reshape the vendor model intensties inside the antenna centric bounding box into the shape of the bounding box
            vendor_model_cube = np.reshape(vendor_model_intensities_array, (size, size, size))
            #vendor_model_cube = np.reshape(vendor_model_intensities, (size, size, size))
            #vendor_model_cube = np.transpose(vendor_model_cube, (1, 0, 2))



        # Convert the segmentation mask into binary (0-1) mask. This will take care of converting the fractional values at
        # the edges after interpolation from a binary mask.
        # cube_label_array = cube_label_array > 0.5 # This is not finalized
        # Not sure if this is necessary.
        if type == 'seg_label':
            interpolated_array_cube = interpolated_array_cube.astype('uint8', copy=False)


        # Mask the cube to retain subset of intensities. This is done to normalize data across cases. For e.g. if we just
        # use the cube, each case may have the cube in different orientations around the antenna. This causes the sampled
        # intensities to not be symmetrical, e.g., at the corners. Using a sphere/cylinder will avoid this asymmetry. But
        # note that this may not be that important at the end of the day.
        # Get the center of the bounding box cube in voxel space
        #cube_center_voxel_space = np.matmul(xyz_to_crs, xyz_coordinates)
        #cube_center_voxel_space = cube_center_voxel_space[:3]
        # Create a grid for the bounding box region
        r_range_bb = np.arange(size)  # Row co-ordinates
        c_range_bb = np.arange(size)  # Column co-ordinates
        s_range_bb = np.arange(size)  # Slice co-ordinates
        # Retain only intensities inside the mask
        if mask == 'cube':
            # Nothing to do since cube is the starting point for all masks
            print('Using a {} mask'.format(mask))
        elif mask == 'sphere':
            print('Using a {} mask'.format(mask))
            # Retain only a sphere of intensities inside the bounding box
            for r in r_range_bb: # Row co-ordinates
                for c in c_range_bb:  # Column co-ordinates
                    for s in s_range_bb:  # Slice co-ordinates
                        if (r - (size-1)/2)**2 + (c - (size-1)/2)**2 + (s - (size-1)/2)**2 > ((size-1)/2)**2:
                            # Crop image series or a binary label like segmentation
                            if type == 'image_series':
                                interpolated_array_cube[r,c,s] = -1024 # Lowest CT intensity
                            elif type == 'seg_label':
                                interpolated_array_cube[r,c,s] = 0 # Lowest label value
                            # Crop the vendor model
                            if vendor_model is not None:
                                vendor_model_cube[r,c,s] = 0 # Lowest label value                          
        elif mask == 'rectangle':
            # Not implemented yet
            print('{} mask not implemented yet'.format(mask))
        elif mask == 'cylinder':
            # Not implemented yet
            print('{} mask not implemented yet'.format(mask))
        else:
            raise ValueError('Image type is invalid')


        # Write images/arrays to output files
        #--------------------------------------------------------------------------------------------------
        # # Write to nrrd files.
        # nrrd.write('output_intensity.nrrd', cube_voxel_array)
        # nrrd.write('output_label.nrrd', cube_label_array)
        # Writing to nifty file since nnunet requires nifty.

        # Create the Nifti images. Assume an identity affine matrix for voxel-to-world transform 
        # Primary image (could be the gray scale image series or a segmentation binary image)
        interpolated_array_cube_nib = nib.Nifti1Image(interpolated_array_cube, np.eye(4))
        # Vendor model if specified
        if vendor_model is not None:
            vendor_model_cube_nib = nib.Nifti1Image(vendor_model_cube, np.eye(4))

        # Create file name for the nifti files and save them
        # 1. We have to do it in the format required by the training algorithm
        # 2. Multiple channels of the same file need to specified as trailing 0001, 0002 etc, while the different rotations will generate different unique cases, specified as part of case identifier.
        # Case identifier
        # Note that we are converting the rot_angle to ints and not float. The reason is, when we write to nifty files, it seems the name of the image is set to 
        # the string until the first encountered period. When we have float, e.g. 'subj41_rotAngle_350.0deg_0001.nii.gz', nifty sets 'subj41_rotAngle_350' as the 
        # image name. The problem with this is, for a different modality (channel e.g. vendor model), e.g., 'subj41_rotAngle_350.0deg_0002.nii.gz', nifty will 
        # set the image name to still be 'subj41_rotAngle_350'. This can be confusing for visualization or downstream algorithms that consume this data. Hence, to 
        # avoid this, we can either restrict the degress to only ints or use 'point' instead of period, if we want to denote floats. We used the former approach
        # here and hence only integer degrees are allowed. So, the number of rotations specified in the input has to be able to divide 360 exactly. Raising an 
        # exception otherwise  
        if not math.isclose(int(rot_angle), float(rot_angle)):
            raise ValueError('The specified number of rotations does not result in integer rotation angles. This causes a problem in naming the final nifty file as explained in the current version of the code. Exiting...')
        case_identifier = case_name + '_rotAngle_' + str(int(rot_angle)) + 'deg'

        # Create output file name for primary image (could be the gray scale image series or a segmentation binary image)
        if modality_identifiers is not None:
            primary_image_output_file_name = case_identifier + '_' + modality_identifiers['primary_image'] + '.nii.gz'
        else:
            primary_image_output_file_name = case_identifier + '.nii.gz'
        # Save output file
        nib.save(interpolated_array_cube_nib, os.path.join(output_data_path, primary_image_output_file_name))

        # If specified, create output file name for Vendor model and save it.
        if vendor_model is not None:
            # Create output vendor model file name
            if modality_identifiers is not None:
                vendor_model_output_file_name = case_identifier + '_' + modality_identifiers['vendor_model'] + '.nii.gz'
            else:
                vendor_model_output_file_name = case_identifier + '.nii.gz'
            # Save output file
            nib.save(vendor_model_cube_nib, os.path.join(output_data_path, vendor_model_output_file_name))



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


def VendorModelApplicatorTissueTypeAndTimeAndPowerSpec(tissue):
    """ Returns the vendor model dimensions for different ablation treatment parameters    
    """
    
    # Refer to vendor model

    return vendor_spec


def interpolated_vendor_model(applicator, tissue, power, time):
    '''
    Computes the vendor model dimension for ablation treatment parameters not listed in the vendor spec. 
    '''
    # Refer to vendor model

    return interpolated_vendor_model



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



def save_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


# THIS IS NOT NEEDED FOR THE CODE AS THE ANTENNA COORDINATES ARE ALREADY STORED IN THE DICT AND READ FROM THERE. 
# BUT THIS WAS ONLY ADDED TO READ IN THE NEW ANTENNA CO-ORDINATES
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

    # # Add co-ordinates to dictionary
    # ant_coords_dict['tip_lps'] = lps_coords_tip    
    # ant_coords_dict['back1_lps'] = lps_coords_back1    
    # ant_coords_dict['back2_lps'] = lps_coords_back2    

    # Add co-ordinates to dictionary
    ant_coords_dict['lps_coords_tip'] = lps_coords_tip    
    ant_coords_dict['lps_coords_back1'] = lps_coords_back1    
    ant_coords_dict['lps_coords_back2'] = lps_coords_back2    

    return ant_coords_dict



if __name__ == '__main__':

    # Set the reference registation and antenna space among pre and post (followup post). This is the fixed space 
    # to which one of the other space images (moving space) will be registered to.
    # =================================================================================================================================
    # Set the reference registration space. This is the space to which all images are registered to. 
    ref_reg_space = 'followup_post'
    moving_reg_space = 'pre'

    # Antenna co-ordinates will be extracted from this space primarily
    ref_antenna_space = 'followup_post'


    # Set all image files which require the antenna aligned bounding box crops
    # Note that only one pair, i.e. fixed image and moving can be specified as input 
    # at this point. For e.g. if followup is the reference space, then crop_followup_post = True
    # and only crop_pre can be true.  
    #============================================================================================
    crop_followup_post = True
    crop_pre = True

    # Verify settings
    if ref_reg_space == 'followup_post':
        assert (crop_followup_post == True) and (crop_pre == True), "The assignment of the image spaces that are going to be cropped is wrong."


    # Set paths and initialize variables
    #============================================================================================
    # Set input data folder
    elif crop_pre:
        data_base_folder = r'/home/path_to_registered_data'
    else:
        raise ValueError('Neither of crop_pre is True. Something is off. Exiting...')

    # Set cropped output data path. Note that the path includes the angle resolution of the cropping. E.g. we start with 30 degree rotation 
    # resolutiom, i.e., we sample the bounding box every 30 degree rotations.
    rot_resolution = 30
    num_rotations = int(360 / rot_resolution)
    rot_res_dir_name = str(rot_resolution) + 'DegRotRes'    
    if crop_pre:
        output_data_path = os.path.join(r'/home/path_to_registered_data_output')
    else:
        raise ValueError('Neither of crop_pre is True. Something is off. Exiting...')

    # Read in the ablation information dictionary. 
    # There was an issue with allow_pickle set to false by default in the later versions. 
    # Check here: https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa
    # Work around:
    #--------------------------------------------------------------------------------
    # save current np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # Load ablation information 
    elif crop_pre:
        ablation_info_all = np.load(os.path.join(r'/home/path_to_processed_ablation_info'))
    else:
        raise ValueError('Neither of crop_pre is True. Something is off. Exiting...')    
    # Restore np.load for future normal usage
    np.load = np_load_old 
    #---------------------
    # Get the ablation info dictionary. Note that we need the [()] at the end since numpy saves the dictionary as a numpy array. Check here: https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa
    ablation_info_all = ablation_info_all['ablation_info_all'][()]
    # # Debug
    # for key, value in ablation_info_all.items():
    #     print(key)

    

    # List all the valid cases with the relevant files with full paths
    #===============================================================================================
    # Initialize the valid case list, file path dictionary and invalid list
    valid_cases_dict = {}
    invalid_cases_list = []
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

            # Check if all the required files that need to be cropped exist for the case. 
            # Notes:
            # - For all registrations, we assume a 70mm (initially 100mm) radius sphere crop is the starting point and not the full image as there is no need for the full image. The interpolation
            #   to compute the bounding box is faster as well if we use the crop. Further, the registrations use a 50mm (tried 30mm for pre to post) radius sphere mask to focus the registrations on all
            #   images. Because of this, the registration is accurate inside that mask and decreases in accuracy as move away. So, there is no reason to keep the whole image 
            #   since beyong 50mm radius, the registration is not accurate and hence not that usable. We could go to may be 75mm radius as the upper limit, which is within the 100mm
            #   radius crop. Looking at the size of the cubes based on the above point, we can go until approx 150mm side (75mm radius). This is enough for our experiments to build a graph.
            # - Based on the above, the files we will search for include:
            #            
            #   Pre to post
            #   1) The cropped pre image and tumor segmentation
            #   2) The cropped pre tumor segmentation
            #   3) The cropped followup post image
            #   4) The cropped followup ablation segmentation. 
            

            # Pre and followwup post
            if crop_pre and crop_followup_post:
                if ref_reg_space == 'followup_post':
                    # Pre files
                    #-------------------------------------------------------------------------------------
                    # # THE BELOW IS INCORRECT SINCE IT IS READING THE PRE FILES BEFORE REGISTRATION. HENCE COMMENTED
                    # # The pre segmentations are cropped in each registration attempt with decreasing cropped radii.
                    # # Hence there may be multiple pre segmentation files. Use the one that has the smallest radius in 
                    # # its names since that is the one for which the registration succeeded. But the pre scan itself 
                    # # should be only 1 since it is saved only when the registration succeeds.
                    # #-------------------------------------------------
                    # # Find all pre cropped files. This should give back at least 3 files, the scan, regmask and seg. There may be multiple seg files.   
                    # pre_files = get_files(subject_folder, 'pre_' + '*' + '*cropped*', '.nrrd') # Note that the function adds additional wild cards between the inputs
                    # # Get the pre scan file
                    # #-----------------------------
                    # pre_scan_file = []
                    # for pre_file in pre_files:
                    #     if ('seg' in os.path.basename(pre_file)) or ('regmask' in os.path.basename(pre_file)):
                    #         continue
                    #     pre_scan_file.append(pre_file)                     
                    # # There should be only 1 pre scan file. Check and store it. If not found, the registration must have 
                    # # failed. Store it as an invalid file and continue to the next case.  
                    # if len(pre_scan_file) != 1:
                    #     invalid_cases_list.append(case_name)
                    #     continue
                    # # Pre scan file
                    # pre_scan_file = pre_scan_file[0]                    
                    # # Get the pre segmentation file
                    # #------------------------------------
                    # # Find the pre segmentation file. This could give us multiple files since the segmentation files are 
                    # # saved at the beginning of every registration attempt (i.e., with decreasing mask sizes).    
                    # pre_seg_files = get_files(subject_folder, 'pre_' + '*' + '*seg*' + '*cropped*', '.nrrd') # Note that the function adds additional wild cards between the inputs
                    # if not pre_seg_files:
                    #     invalid_cases_list.append(case_name)
                    #     continue
                    # # Find the segmentation file with the smallest crop radius. This is the one for which the registration succeeded.
                    # # Sort in increasing order and take the first seg file.
                    # pre_seg_files.sort()
                    # pre_seg_file = pre_seg_files[0]       

                    # # Check valid cropped scan and seg files were found. If not, count this case as failure and continue to the next one.
                    # if (not pre_scan_file) or (not pre_seg_file):
                    #     invalid_cases_list.append(case_name)
                    #     continue


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

                    # TODO: We forgot to resample the tumor segmentation into the post space. This is required 
                    # TODO: to be added as an additional channel during the pre to post prediction. This can 
                    # TODO: done by reading the registration parameters back and resampling the segmentation
                    # TODO: for now. But this needs to be included in the registation code in the next iteration
                    # TODO: We can then include reading the segmentation file here!  

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
                                    == pre_scan_file_crop_radius 
                                        #== os.path.splitext(os.path.basename(pre_seg_file))[0].rsplit('_', 1)[1] 

                elif ref_reg_space == 'pre':
                    print('Not implemented currently')
                else:
                    raise ValueError('Unknown reference registration space. Exiting...')

            else:
                raise ValueError('The combination of moving and fixed space is not support at this point. Exiting...')

            # If the case didn't fail so far, then it is valid. Add the corresponding files with full path to dictionary 
            # Create a sub-dictionary with file paths
            valid_files_dict = {}
            if bool(followup_post_scan_file):
                valid_files_dict['followup_post_scan_file'] = followup_post_scan_file
            if bool(followup_post_seg_file):
                valid_files_dict['followup_post_seg_file'] = followup_post_seg_file
            if bool(pre_scan_file):
                valid_files_dict['pre_scan_file'] = pre_scan_file
            if bool(pre_seg_file):
                valid_files_dict['pre_seg_file'] = pre_seg_file
            # Add sub-dictionary to valid cases dictionary
            valid_cases_dict[case_name] = valid_files_dict 

    print('Completed storing case files with full path. There were totally {} valid and {} invalid cases'.format(len(valid_cases_dict), len(invalid_cases_list)))


    # Divide the valid cases into train and test groups using scikit-learn. 
    #---------------------------------------------------------------------------------------------------
    # Note that we split the cases at patient level with 
    # 86% train and 14% test. This was done at the patient level to address any questions about data leakage by having ablations
    # from the same patient in both train and test.
    #----------------------------------------------------------------------------------------------------------------------------------
    # Create list of cases
    valid_cases_list = list(valid_cases_dict.keys())
    # Create the list of cases at the subject level (i.e. counting multiple ablations/cases from the same subject only once)
    valid_subject_list = []
    for case in valid_cases_list:
        valid_subject_list.append(case[:6]) # case[:6] will only extract the subject name and leave out the letter at the end indicating the specific ablation. e.g. only 'subj47' will be extracted from 'subj47a'    
    # Unique subjects and their counts (histogram)
    valid_subject_list_unique, counts = np.unique(valid_subject_list, return_counts=True)
    valid_subject_list_train, valid_subject_list_test = train_test_split(valid_subject_list_unique, test_size=0.14, random_state=1)
    # Create the list of valid train and test cases (not subjects)
    # Train
    valid_cases_list_train = []
    for subject in valid_subject_list_train:
        # Find all cases that belong to this subject, i.e. find all cases that have the subject as the substring
        subject_cases = [case for case in valid_cases_list if subject in case]
        # Add these cases to the cases train list
        valid_cases_list_train.extend(subject_cases)
    # Test
    valid_cases_list_test = []
    for subject in valid_subject_list_test:
        # Find all cases that belong to this subject, i.e. find all cases that have the subject as the substring
        subject_cases = [case for case in valid_cases_list if subject in case]
        # Add these cases to the cases train list
        valid_cases_list_test.extend(subject_cases)


    # Split train into train and validation. 
    # NOTE: The train cases are further split into train and validation at the level of ablations and not subjects. I.e.
    # e.g., subj83b and subj83a are treated as distinct cases and one may be in train and another in validaiton sets.
    # We only take care to make sure different rotations of the same subject do not end up in train and validation at the same
    # time. I.e, subj83b_30deg and subj83b_120deg cannot be in train and validation respectively at the same time.
    #-----------------------------------------------------------------------------------------------------------------------
    # This is done here and not leave it to nnunet to do the split for the following reason: For all 
    # train cases, we extract "n" rotational positions of BBs (e.g. 12) around the antenna and consider them as separate cases.
    # nnunet does the split internally into train and val, treating them as independent cases and hence causing rotations from the same case ending 
    # up in both train and validation sets. This causes overfitting and poor early stopping (for selecting the best 
    # model) since the validation curve follows training very closely, not increasing at any point with #epochs. This 
    # may result in poor performance on the test set because of overfitting. It may as well be that the improvement in 
    # performance may not be a lot if we account for this, but still accounting for this would be the right way. So, 
    # instead of letting nnunet create the data splits, we are creating that file here ourselves, which needs to be copied to the 
    # preprocessed folder in nnunet. Then nnunet will use this split file. 
    splits = []
    splits_file = "splits_final.pkl"
    #all_keys_sorted = np.sort(list(self.dataset.keys())) # Not sorting the cases as it doesn't solve any purpose.
    kfold = KFold(n_splits=5, shuffle=True, random_state=12345) 
    for i, (train_idx, test_idx) in enumerate(kfold.split(valid_cases_list_train)):
        train_cases = np.array(valid_cases_list_train)[train_idx]
        test_cases = np.array(valid_cases_list_train)[test_idx]
        len(train_cases)
        len(test_cases)
        print(train_cases[0])
        print(test_cases[0])
        # For each cases, create the corresponding key, that needs to be written into splits_final.pkl
        # Get the number of rotation angles of the antenna centric BBs
        rot_angles_abt_ant_axis = np.linspace(0, 360, num_rotations+1)[:-1] # Only when we specify num_rotations+1 rotations (e.g. 36+1 = 37) do we get rotations at whole integer angles (i.e. 0 deg, 10 deg, 20 deg.. 350 deg etc..). But now the (num_rotations+1)th rotation (e.g. 37th) is the same as 0th rotation, hence considering only until num_rotationth (36th) rotation. 
        # Train
        train_keys = []
        for tr_case in train_cases:
            # Iterate through all rotations, compute bounding boxes and save them to file
            for rot_angle in rot_angles_abt_ant_axis:
                # Get the key corresponding to the case
                tr_key = tr_case + '_rotAngle_' + str(int(rot_angle)) + 'deg'
                train_keys.append(tr_key)
        train_keys = np.array(train_keys)
        # Test
        test_keys = []
        for ts_case in test_cases:
            # Iterate through all rotations, compute bounding boxes and svae them to file
            for rot_angle in rot_angles_abt_ant_axis:
                # Get the key corresponding to the case
                ts_key = ts_case + '_rotAngle_' + str(int(rot_angle)) + 'deg'
                test_keys.append(ts_key)
        test_keys = np.array(test_keys)        
        # Create splits and write to pkl file
        splits.append(OrderedDict())
        splits[-1]['train'] = train_keys
        splits[-1]['val'] = test_keys
    # Save splits to pickle file
    save_pickle(splits, splits_file)

    # Iterate through all cases, compute the antenna centric bounding box crops and save the cropped files into train and test case folders. Adhere to the 
    # conventions required by the nnU-net code for training. 
    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    # Overwrite existing cases
    overwrite = True
    # We specify the fov of the cropped cube. This can range from side 32mm to side 176. The reason being the registrations were done using spherical masks of radius
    # 50mm. So the registration is going to be less accurate if we go with a square side > 100mm. Eventually, a sphere is extracted from these squares. So, we go until 
    # 176mm side ~ 90mm radius. 
    bb_params = []
    bb_params.append({'fov': 64, 'side': 64}) 

    # Iterate through all the different cube sizes and process the cases. We iterate through the cube size so that all the cases are available for each cube size 
    # before we go to the next cube size.
    for bb_param in bb_params:

        # Create new bounding box output folder
        new_bounding_box_folder = os.path.join(output_data_path, str(bb_param['fov']))
        if not os.path.exists(new_bounding_box_folder):
            os.makedirs(new_bounding_box_folder)

        # Create task folder
        if crop_pre and crop_followup_post:
            output_task_path = os.path.join(new_bounding_box_folder, 'nnUNet_raw_data_base', 'nnUNet_raw_data', 'Task202_preToFollowupAblPred50mmRegMask') # Custom tasks starting from 200 in our case
        else:
            raise ValueError('Neither of crop_pre is True. Something is off. Exiting...')

        # Create the nnunet task folder
        if not os.path.exists(output_task_path):
            os.makedirs(output_task_path)

        # Train
        #--------
        # Iterate through all training cases and create the respective antenna centric cropped files.
        #-------------------------------------------------------------------------------------------------------
        # Create train and label directory
        output_train_images_path = os.path.join(output_task_path, 'imagesTr')
        output_train_labels_path = os.path.join(output_task_path, 'labelsTr')
        # Create images and labels directories
        if not os.path.exists(output_train_images_path):
            os.makedirs(output_train_images_path)
        if not os.path.exists(output_train_labels_path):
            os.makedirs(output_train_labels_path)
        # Iterate through all cases
        followup_post_missing_antenna_coords_train = [] # Just to see if there are any. In the latest iteration, this should be empty.
        for case_name in valid_cases_list_train:
            # 1. Get the antenna co-ordinates from followup post
            #-----------------------------------------------------------------------------------------------------------------------------------------
            antenna_coords = []
            if ref_reg_space == 'followup_post':

                if moving_reg_space == 'pre':
                    # Get the antenna co-ordinates from the reference registration space, i.e., followup post in this case
                    if ablation_info_all[case_name]['is_followup_post_antenna_coords_valid']:
                        antenna_coords = ablation_info_all[case_name]['post_antenna_coords']
                    else:
                        # All the cases should have antenna in the followup. If not, raise an exception. 
                        raise ValueError('Followup post does not have antenna co-ordinates. This case shouldnt have been copied otherwise. This is not possible at this stage. Please check again. Exiting...')
                else:
                    raise ValueError('Invalid moving registration space')
            
            elif ref_reg_space == 'pre':
                print('Not implemented currently')
            else:
                raise ValueError('Incorrect reference registration space. Exiting...')


            # 2. Compute antenna centric 3D uniform grid around the ablation zones and sample onto them. Compute the antenna 
            # centric vendor model as well. Write all these to nifty files in the format required by the NNunet 
            # algorithm for training.
            # --------------------------------------------------------------------------------------------------------------------------------------------
            # Files that are created include:
            # Input (imagesTr folder)
            # a) The vendor model aligned with the antenna (likely extracted from followup post)
            # Output (labelsTr folder)
            # b) The post proc. binary ablation zone. 
            # Notes:
            # -- All 3D positions of the grid around the ablation zone are considered. This the marginialization method of handling rotation uncertainty. 
            # So, we treat each rotation as a separate case. We noticed that considering rotations every 10 degrees seems sufficient since the information does
            # doesn't change too drastically. 
            # -- The vendor model corresponding to each of these will be the same and saved as a separate modality. 
            # a) and b) Compute antenna centric bounding box of the pre scan and vendor model and save them to files
            #--------------------------------------------------------------------------------------------------------------------
            # Prepare input
            if crop_pre:
                source_img = valid_cases_dict[case_name]['pre_scan_file']
            else:
                raise ValueError('Invalid space specified for cropping') 

            fov = bb_param['fov'] 
            size = bb_param['side'] 
            antenna_orientation = [(antenna_coords['lps_coords_tip']['L'], antenna_coords['lps_coords_tip']['P'], antenna_coords['lps_coords_tip']['S']), (antenna_coords['lps_coords_back1']['L'], antenna_coords['lps_coords_back1']['P'], antenna_coords['lps_coords_back1']['S'])]
            # Modality identifier string are appended to each file. These essentially distinguish the various channel inputs to the neural network. E.g. CT image and vendor model.
            # This conforms to the nnunet model input specification.  
            modality_identifiers = {}
            modality_identifiers['primary_image'] = '0000'
            modality_identifiers['vendor_model'] = '0001'
            # Specify Vendor model
            vendor_model = {}
            vendor_model['diameter'], vendor_model['length'], vendor_model['dist_past_tip'] = interpolated_vendor_model('Vendor_applicator', 'lung', float(ablation_info_all[case_name]['power']), float(ablation_info_all[case_name]['time']))
            # Compute and save bounding boxes
            crop_and_save_roi_along_antenna(source_img, fov, size, antenna_orientation, case_name, output_train_images_path, num_rotations = num_rotations, type='image_series', mask='sphere', all_rotations=True, vendor_model=vendor_model, modality_identifiers=modality_identifiers)
            
            # c) Compute antenna centric bounding box of the post scan ablation segmentation
            #--------------------------------------------------------------------------------------------------------------------
            # Prepare input
            source_img = valid_cases_dict[case_name]['followup_post_seg_file'] 
            fov = bb_param['fov'] 
            size = bb_param['side'] 
            antenna_orientation = [(antenna_coords['lps_coords_tip']['L'], antenna_coords['lps_coords_tip']['P'], antenna_coords['lps_coords_tip']['S']), (antenna_coords['lps_coords_back1']['L'], antenna_coords['lps_coords_back1']['P'], antenna_coords['lps_coords_back1']['S'])]
            # Compute and save bounding boxes
            crop_and_save_roi_along_antenna(source_img, fov, size, antenna_orientation, case_name, output_train_labels_path, num_rotations = num_rotations, type='seg_label', mask='sphere', all_rotations=True)

            # Debug
            #-------
            # For the purposes of debugging, create the post image antenna centric bounding box as well.
            # Prepare input
            output_train_label_images_path = os.path.join(output_task_path, 'labelImagesTr')
            # Create images and labels directories
            if not os.path.exists(output_train_label_images_path):
                os.makedirs(output_train_label_images_path)
            # Save cropped images to file            
            source_img = valid_cases_dict[case_name]['followup_post_scan_file'] 
            fov = bb_param['fov'] 
            size = bb_param['side'] 
            antenna_orientation = [(antenna_coords['lps_coords_tip']['L'], antenna_coords['lps_coords_tip']['P'], antenna_coords['lps_coords_tip']['S']), (antenna_coords['lps_coords_back1']['L'], antenna_coords['lps_coords_back1']['P'], antenna_coords['lps_coords_back1']['S'])]
            modality_identifiers = {}
            modality_identifiers['primary_image'] = '0002'
            # Compute and save bounding boxes
            crop_and_save_roi_along_antenna(source_img, fov, size, antenna_orientation, case_name, output_train_label_images_path, num_rotations = num_rotations, type='image_series', mask='sphere', all_rotations=True, modality_identifiers=modality_identifiers)


        # Test
        #--------
        # Iterate through all test cases and create the respective antenna centric cropped files.
        #-------------------------------------------------------------------------------------------------------
        # Create test and label directory
        output_test_images_path = os.path.join(output_task_path, 'imagesTs')
        output_test_labels_path = os.path.join(output_task_path, 'labelsTs')
        # Create images and labels directories
        if not os.path.exists(output_test_images_path):
            os.makedirs(output_test_images_path)
        if not os.path.exists(output_test_labels_path):
            os.makedirs(output_test_labels_path)
        # Iterate through all cases
        followup_post_missing_antenna_coords_test = [] # Just to see if there are any. In the latest iteration, this should be empty.
        for case_name in valid_cases_list_test:
            # 1. Get the antenna co-ordinates from followup post.
            #-----------------------------------------------------------------------------------------------------------------------------------------
            antenna_coords = []
            if ref_reg_space == 'followup_post':

                if moving_reg_space == 'pre':
                    # Get the antenna co-ordinates from the reference registration space, i.e., followup post in this case
                    if ablation_info_all[case_name]['is_followup_post_antenna_coords_valid']:
                        antenna_coords = ablation_info_all[case_name]['post_antenna_coords']
                    else:
                        # All the cases should have antenna in the followup. If not, raise an exception. 
                        raise ValueError('Followup post does not have antenna co-ordinates. This case shouldnt have been copied otherwise. This is not possible at this stage. Please check again. Exiting...')
                else:
                    raise ValueError('Invalid moving registration space')
            
            elif ref_reg_space == 'pre':
                print('Not implemented currently')
            else:
                raise ValueError('Incorrect reference registration space. Exiting...')


            # 2. Compute antenna centric 3D uniform grid around the ablation zones and sample onto them. Compute the antenna 
            # centric vendor model as well. Write all these to nifty files in the format required by the NNunet 
            # algorithm for training.
            # --------------------------------------------------------------------------------------------------------------------------------------------
            # Files that are created include:
            # Input (imagesTs folder)
            # a) The vendor model aligned with the antenna (likely extracted from followup post)
            # Output (labelsTs folder)
            # b) The post proc. binary ablation zone. 
            # Notes:
            # -- All 3D positions of the grid around the ablation zone are considered. This the marginialization method of handling rotation uncertainty. 
            # So, we treat each rotation as a separate case. We noticed that considering rotations every 10 degrees seems sufficient since the information does
            # doesn't change too drastically. 
            # -- The vendor model corresponding to each of these will be the same and saved as a separate modality. 
            # a) and b) Compute antenna centric bounding box of the pre scan and vendor model and save them to files
            #--------------------------------------------------------------------------------------------------------------------
            # Prepare input
            elif crop_pre:
                source_img = valid_cases_dict[case_name]['pre_scan_file']
            else:
                raise ValueError('Invalid space specified for cropping') 

            fov = bb_param['fov'] 
            size = bb_param['side'] 
            antenna_orientation = [(antenna_coords['lps_coords_tip']['L'], antenna_coords['lps_coords_tip']['P'], antenna_coords['lps_coords_tip']['S']), (antenna_coords['lps_coords_back1']['L'], antenna_coords['lps_coords_back1']['P'], antenna_coords['lps_coords_back1']['S'])]
            # Modality identifier string are appended to each file. These essentially distinguish the various channel inputs to the neural network. E.g. CT image and vendor model.
            # This conforms to the nnunet model input specification.  
            modality_identifiers = {}
            modality_identifiers['primary_image'] = '0000'
            modality_identifiers['vendor_model'] = '0001'
            # Specify Vendor model
            vendor_model = {}
            vendor_model['diameter'], vendor_model['length'], vendor_model['dist_past_tip'] = interpolated_vendor_model('Vendor_applicator', 'lung', float(ablation_info_all[case_name]['power']), float(ablation_info_all[case_name]['time']))
            # Compute and save bounding boxes
            crop_and_save_roi_along_antenna(source_img, fov, size, antenna_orientation, case_name, output_test_images_path, num_rotations = num_rotations, type='image_series', mask='sphere', all_rotations=True, vendor_model=vendor_model, modality_identifiers=modality_identifiers)
            
            # c) Compute antenna centric bounding box of the post scan ablation segmentation
            #--------------------------------------------------------------------------------------------------------------------
            # Prepare input
            source_img = valid_cases_dict[case_name]['followup_post_seg_file'] 
            fov = bb_param['fov'] 
            size = bb_param['side'] 
            antenna_orientation = [(antenna_coords['lps_coords_tip']['L'], antenna_coords['lps_coords_tip']['P'], antenna_coords['lps_coords_tip']['S']), (antenna_coords['lps_coords_back1']['L'], antenna_coords['lps_coords_back1']['P'], antenna_coords['lps_coords_back1']['S'])]
            # Compute and save bounding boxes
            crop_and_save_roi_along_antenna(source_img, fov, size, antenna_orientation, case_name, output_test_labels_path, num_rotations = num_rotations, type='seg_label', mask='sphere', all_rotations=True)

            # Debug
            #-------
            # For the purposes of debugging, create the post image antenna centric bounding box as well.
            # Prepare input
            output_test_label_images_path = os.path.join(output_task_path, 'labelImagesTs')
            # Create images and labels directories
            if not os.path.exists(output_test_label_images_path):
                os.makedirs(output_test_label_images_path)
            # Save cropped images to file            
            source_img = valid_cases_dict[case_name]['followup_post_scan_file'] 
            fov = bb_param['fov'] 
            size = bb_param['side'] 
            antenna_orientation = [(antenna_coords['lps_coords_tip']['L'], antenna_coords['lps_coords_tip']['P'], antenna_coords['lps_coords_tip']['S']), (antenna_coords['lps_coords_back1']['L'], antenna_coords['lps_coords_back1']['P'], antenna_coords['lps_coords_back1']['S'])]
            modality_identifiers = {}
            modality_identifiers['primary_image'] = '0002'
            # Compute and save bounding boxes
            crop_and_save_roi_along_antenna(source_img, fov, size, antenna_orientation, case_name, output_test_label_images_path, num_rotations = num_rotations, type='image_series', mask='sphere', all_rotations=True, modality_identifiers=modality_identifiers)




