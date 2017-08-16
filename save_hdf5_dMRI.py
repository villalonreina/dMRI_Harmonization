#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 12:53:46 2017

@author: jvillalo
"""

import nibabel as nib
import h5py
import numpy as np

def make_hdf5_diffusion:
    
    dirInput = '/Users/jvillalo/Documents/IGC/Conferences/MICCAI_2017/Challenge/Test_data/Prisma/st/Masked_brains/'
    train_subjects = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I']
    val_subjects = ['J']
    test_subjects = ['K']
    suffix = '_dwi_masked.nii.gz'
    
    train_shape = (8, 96, 96, 60, 36)
    val_shape = (1, 96, 96, 60, 36)
    test_shape = (1, 96, 96, 60, 36)
    
    f = h5py.File("./test_diffusion.hdf5", mode='w')
    f.create_dataset("train_img", train_shape, np.float32)
    
    for i in range(len(train_subjects)):
    
        imagestring = ''.join([dirInput, train_subjects[i], suffix])
        img = nib.load(imagestring)
        dataimg = img.get_data()
        print('dataimg.shape (%d, %d, %d, %d)' % dataimg.shape)
        f["train_img"][i, ...] = dataimg
    
    
    f.create_dataset("val_img", val_shape, np.float32)
    imagestring = ''.join([dirInput, val_subjects[0], suffix])
    img = nib.load(imagestring)
    dataimg = img.get_data()
    print('dataimg.shape (%d, %d, %d, %d)' % dataimg.shape)
    f["val_img"][0, ...] = dataimg
    
    f.create_dataset("test_img", test_shape, np.float32)
    imagestring = ''.join([dirInput, test_subjects[0], suffix])
    img = nib.load(imagestring)
    dataimg = img.get_data()
    print('dataimg.shape (%d, %d, %d, %d)' % dataimg.shape)
    f["test_img"][0, ...] = dataimg
    
    f.close()

if __name__ == "__main__":
    make_hdf5_diffusion()