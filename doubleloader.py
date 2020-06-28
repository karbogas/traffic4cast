# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:27:25 2020

@author: karbogas
"""

import torch
import pickle
import os
import glob
import h5py
import numpy as np

class both_datasets(torch.utils.data.Dataset):
    def __init__(self,image_root,graph_root, city, filter_test_times, includeHeading = False):
        """
         Loads data in image and graph form
         
         input:
             image_root:        Storage of image data
             graph_root:        Storage of graph data
             city:              City for which data is loaded
             filter_test_times: Only load data for test times?
             includeHeading:    Also load data of heading channel?
                             
         output: 
             x:                 Training data
             y:                 Ground truth (graph)
             imY:               Ground trugh (image)
        """
        
        super(both_datasets, self).__init__()
        
        self.image_root = image_root
        self.graph_root = graph_root
        self.city = city
        self.filter_test_times = filter_test_times
        self.includeHeading = includeHeading
        
        
        # Load test times
        if self.filter_test_times:
            dictTT = {}
            dictTemp = pickle.load(open(os.path.join('.', 'utils', 'test_timestamps.dict'), "rb"))
            for city, values in dictTemp.items():
                values.sort()
                dictTT[city] = values
                
            self.test_dict = dictTT
        
        # List of paths of graph files
        graph_paths = []
        graph_paths = graph_paths + glob.glob(
            os.path.join(self.graph_root, self.city, 'validate',
                         '*.h5'))

        self.graph_paths = graph_paths
        
        
        # List of paths of image files
        image_paths = []
        image_paths = image_paths + glob.glob(
            os.path.join(self.image_root, self.city, self.city + '_validation',
                         '*.h5'))
        
        self.image_paths = image_paths
        
        
        
    def __len__(self):
        if self.filter_test_times:
            return len(self.graph_paths) * 5
        else:
            return len(self.graph_paths) * 274
        
        
    def __getitem__(self, idx):
        
        # Get file names from index
        if self.filter_test_times:
            file = idx // 5
            timeNr = idx % 5
            timestamp =  self.test_dict[self.city][timeNr]
        else: 
            file = idx // 274
            timestamp = idx % 274
        
        gPath = self.graph_paths[file]
                
        # Load graph file and get array
        f = h5py.File(gPath, 'r')
        graph = f.get('array')
        
        # Get training dataset (graph)
        x = torch.from_numpy(graph[timestamp:timestamp+12, :, :])
        
        # Retransform and remove heading if necessary
        x = x.permute(2,0,1)
        if not self.includeHeading:
            x = x[:,:,0:2]
        x = x.contiguous().view(x.size()[0],-1).float()
        x = x/255
        
        # Get ground truth (graph)
        y = torch.from_numpy(graph[timestamp+12:timestamp+15, :, :])
        
        # Retransform and remove heading if necessary
        y = y.permute(2,0,1)
        if not self.includeHeading:
            y = y[:,:,0:2]
        y = y.contiguous().view(y.size()[0],-1).float()
        
        
        
        # Load image data
        iPath = self.image_paths[file]
        f = h5py.File(iPath, 'r')
        image = f.get('array')
        
        # Load ground truth, with or without heading channel
        if self.includeHeading:
            imY = image[timestamp+12:timestamp+15, :, :, :]
        else:
            imY = image[timestamp+12:timestamp+15, 0:2, :, :]
        
        # Retransform image data
        imY = np.moveaxis(imY, (0, 1), (2, 3))
        imY = np.reshape(imY, (495, 436, 6))
        imY = torch.from_numpy(imY)
        imY = imY.permute(2,0,1)
        imY = imY.to(dtype=torch.float)
        
        return x,y,imY