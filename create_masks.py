# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:25:20 2020

@author: karbogas

"""

import torch
import os
from videoloader import trafic4cast_dataset
import pickle

def mask_creator(source_root, target_root, masks_target, city, filter_test_times, num_workers, batch_size, shuffle):
    """
        Creates a sum array needed to create a threshold mask
        
        input: 
            source_root:        storage of the raw image data
            target_root:        storage of preprocessed image data
            masks_target:       storage of masks
            city:               city for which the mask should be built
            filter_test_times:  Only build the mask using the test times?
            num_workers:        Workers for the data loader
            batch_size:         Batch size for the data loader
            shuffle:            Shuffle the dataset?
                
        output:
            mask_dict:          dictionary containing the sum array
    
    """
    
    # If mask folder is not available yet, create it
    path = os.path.join(masks_target, 'full_masks')
    if not os.path.exists(path):
        os.mkdir(path)
    
    string = city + '_fullMask.dict'
    path = os.path.join(path, string)

    # If the sum array is already available, load it
    if os.path.exists(path):
        print('Array for ' + city + ' already available, loading it')
        mask_dict = pickle.load(open(path, 'rb'))
        
        
    else:
        print('Building array for ' + city)
        
        # Load data in image form
        dataset_train = trafic4cast_dataset(source_root, target_root,
                                                split_type='training', cities = [city], filter_test_times = filter_test_times)
        dataset_val = trafic4cast_dataset(source_root, target_root,
                                              split_type='validation', cities = [city], filter_test_times = filter_test_times)
        
        train_loader = torch.utils.data.DataLoader(dataset_train, num_workers = num_workers, batch_size = batch_size, shuffle = shuffle)
        val_loader = torch.utils.data.DataLoader(dataset_val, num_workers = num_workers, batch_size = batch_size, shuffle = shuffle)
        
        valueSum = 0
        
        # training and validation data
        loader_list = [train_loader, val_loader]
        
        # Calculate the sum per pixel for each batch. Add all batches together
        for loader in loader_list:
            for batch_idx, (data, Y, context) in enumerate(loader):
                
                
                batchSum = torch.sum(data, (0,1,2))
                valueSum = valueSum + batchSum 
                
        valueSum = valueSum.numpy()
        
        # Create and save sum array dictionary
        mask_dict = {'sum': valueSum}
    
        print('writing file for ' + city)
        pickle.dump(mask_dict, open(path, "wb"))
        print('done')
    
    return mask_dict