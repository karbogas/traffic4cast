# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:45:48 2020

@author: karbogas
"""

import torch
import torch_geometric
import pickle
import os
import glob
import h5py

class geometric_dataset(torch_geometric.data.Dataset):
    def __init__(self,root,edge_index,posTensor = 0, split_type='train',  city= 'Berlin', filter_test_times = False, includeHeading = False):
        """
         Loads data in graph form
         
         input:
            root:               Folder where the graph data is stored
            edge_index:         Tensor containing connection information
            posTensor:          Tensor containing position information
            split_type:         Type of dataset (training/validation)
            city:               City for which data should be loaded
            filter_test_times:  Only load data for test times?
            includeHeading:     Load heading channel?
                             
         output: 
            data:               Graph object for the training
        """
        
        super(geometric_dataset, self).__init__()
        
        self.root = root
        self.edge = edge_index
        #self.transform = transform
        #self.pre_transform = pre_transform
        #self.pre_filter = pre_filter
        self.split_type = split_type
        self.city = city
        self.filter_test_times = filter_test_times
        self.posTensor = posTensor
        self.includeHeading = includeHeading
        
        # List of paths of graph files
        paths = []
        paths = paths + glob.glob(
            os.path.join(self.root, self.city, self.split_type,
                         '*.h5'))

        self.paths = paths
        
        # Load test times
        if self.filter_test_times:
            dictTT = {}
            dictTemp = pickle.load(open(os.path.join('.', 'utils', 'test_timestamps.dict'), "rb"))
            for city, values in dictTemp.items():
                values.sort()
                dictTT[city] = values
                
            self.test_dict = dictTT
        
        
    def len(self):
        if self.filter_test_times:
            return len(self.paths) * 5
        else:
            return len(self.paths) * 274
        
    def get(self, idx):
        
        # Get file names from index
        if self.filter_test_times:
            file = idx // 5
            timeNr = idx % 5
            timestamp =  self.test_dict[self.city][timeNr]
            
        else:
            file = idx // 274
            timestamp = idx % 274
        
        path = self.paths[file]
                
        # Load graph file and get array
        f = h5py.File(path, 'r')
        graph = f.get('array')
        
        # Get training dataset
        x = torch.from_numpy(graph[timestamp:timestamp+12, :, :])
        
        # Retransform and remove heading if necessary
        x = x.permute(2,0,1)
        if not self.includeHeading:
            x = x[:,:,0:2]
        x = x.contiguous().view(x.size()[0],-1).float()
        x = x/255
        
        # Get ground truth 
        y = torch.from_numpy(graph[timestamp+12:timestamp+15, :, :])
        
        # Retransform and remove heading if necessary
        y = y.permute(2,0,1)
        if not self.includeHeading:
            y = y[:,:,0:2]
        y = y.contiguous().view(y.size()[0],-1).float()
        
        
        data = torch_geometric.data.Data(x=x, edge_index=self.edge,y=y, pos = self.posTensor)
        
        
        
        return data
    