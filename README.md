# Abl-Pred

This repository contains code for predicting ablation zones from pre procedure imaging and ablation treatment parameters (applicator position/orientation, power, duration).

The different parts of the code perform different functions:
1. "pre_to_followuppost_datacopy.py": Assuming a folder structure and file name structure for the pre-procedure scan and 1 month followup post procedure scan, this script ensures we have all files and copies them into respective folder structures.
2. "pre_to_followup_registration.py": Given the pre and followup post procedure scan, this script registers them using our novel deformable image registration procedure optimized for ablation workflows. 
3. After registration, the tumor segmentation is pulled into ablation zone space.
4. "align_antenna_and_rotation.py": Once after registration, this script extracts the applicator centric co-ordinate system from the registered pre and followup post procedure images. This is used for standardizing the orientations of ablation zones across cases and focusing on the region of interest.
5. "nnUnet": This is code "as is" from the nnUnet repository: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1. Our ablation prediction algorithm is based on the nnUnet code. In our case, the input has two channels: 1) pre proc. scan and 2) the 3D vendor model representative of power and duration of ablation. The output is a single channel prediction of the binary 3D ablation zone shape as expected on the followup CT scan. The nnUnet code does not take two channels as input by default. It does so in the 3d_lowres → 3d_cascade_fullres setup. So, the 3d_cascade_fullres code is to be used for reading the vendor mdoel as the 2nd channel, in-place of the 3d_lowres output. This can be done by pointing the 3d_cascade_fullres code to the path to the vendor model files appropriately.
