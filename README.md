# Abl-Pred

This repository contains code for predicting ablation zones from pre procedure imaging and ablation treatment parameters (applicator position/orientation, power, duration).

The different parts of the code perform different functions:
1. "data_preparation.py": Reads and organizes the data in ablation workflows. For e.g., the 1) pre procedure and 1 month followup post procedure scans; 2) ablation treatment parameters (applicator position, power, duration). Organizes them into specific formats, folders and files. The code assumes an input folder structure and file name structure, and ensures we have all necessary files.
2. "ablation_workflow_registration.py": Novel deformable registration methodology optimized for ablation workflows. For e.g., given the pre and followup post procedure scan, this script registers the pre scan with followup post scan using our novel deformable registration methodology optimized for ablation workflows. 
3. "pull_through_registration.py": After registration, transforms any object in the moving image space to the final fixed image space using interpolation. For e.g., the tumor segmentation in the pre scan is transformed into the followup post scan space using tri-linear interpolation.
4. "align_antenna_and_rotation.py": Applicator centric co-ordinate system to standardize ablation scan data and focus on ROI. Once after all the scans and segmentations are in the same space, e.g., the followup post space, this script extracts the applicator centric co-ordinate system from the registered pre and followup post procedure images. 
5. "nnUnet": This is code "as is" from the nnUnet version v1 repository: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1. Our ablation prediction algorithm is based on the nnUnet code. In our case, the input has two channels: 1) pre proc. scan and 2) the 3D vendor model representative of power and duration of ablation. The output is a single channel prediction of the binary 3D ablation zone shape. The nnUnet code does not take two channels as input by default. It does so in the 3d_lowres → 3d_cascade_fullres setup. So, the 3d_cascade_fullres code is to be used for reading the vendor mdoel as the 2nd channel, in-place of the nnUnet 3d_lowres output. This can be done by pointing the 3d_cascade_fullres nnUnet code to the path to the vendor model files appropriately.
