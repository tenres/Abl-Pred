# Abl-Pred

This repository contains code for predicting ablation zones from pre procedure imaging and ablation treatment parameters (applicator position/orientation, power, duration).

The different parts of the code perform different functions:
1. "data_preparation.py": Reads and organizes the data in ablation workflows. For e.g., the 1) pre procedure and 1 month followup post procedure scans; 2) ablation treatment parameters (applicator position, power, duration). Organizes them into specific formats, folders and files. The code assumes an input folder structure and file name structure, and ensures we have all necessary files.
2. "workflow_registration.py": Novel deformable registration methodology optimized for ablation workflows. This code registers scans in the ablation workflow. E.g., given the pre and followup post procedure scan, this script registers the pre scan with followup post scan using our novel deformable registration methodology optimized for ablation workflows. 
3. "pull_through_registration.py": After registration, transforms any object in the moving image space to the final fixed image space using interpolation. For e.g., the tumor segmentation in the pre scan is transformed into the followup post scan space using tri-linear interpolation.
4. "extract_roi_using_accs.py": Novel applicator centric co-ordinate system as a generic tool to standardize ablation scan data and focus on ROI. Once after all the scans and segmentations are in the same space, this script extracts the applicator centric co-ordinate system. E.g., from the registered pre and followup post procedure images. 
5. "nnUnet": Ablation zone prediction algorithm. Our ablation prediction algorithm is based on the nnUnet code. This folder is "as is" from the nnUnet version v1 repository: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1. In our case, the input has two channels: 1) pre proc. scan and 2) the 3D vendor model representative of power and duration of ablation. The output is a single channel prediction of the binary 3D ablation zone shape. The nnUnet code does not take two channels as input by default. It does so in the 3d_lowres → 3d_cascade_fullres setup. So, the 3d_cascade_fullres code is to be used for reading the vendor mdoel as the 2nd channel, in-place of the nnUnet 3d_lowres output. This can be done by pointing the 3d_cascade_fullres nnUnet code to the path to the vendor model files appropriately.

## Requirements
Python 3.8 Library dependencies: pandas (1.3.5), numpy (1.21.5), matplotlib (3.5.3), scikit-learn (1.0.2), pytorch (1.12.1), nnUnet (v1), scipy (1.7.3), simpleitk (2.0.0), simpleelastix (v0.10.0). 

Installation: see specific libraries for details. Expected install time ~30min.
