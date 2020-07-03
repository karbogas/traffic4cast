# -*- coding: utf-8 -*-
"""
@author: karbogas
"""

import torch
import numpy as np
import os
from utils.secondloader import traffic_dataset
import pickle
from configPreprocess import config
from utils.create_masks import mask_creator
import shutil
import scipy
import networkx as nx
import h5py
from utils.OSM import getMaskFromOSM, getStreetClusters

if __name__ == '__main__':
    
    # Load config
    source_root = config['source_root']
    target_root = config['target_root']
    masks_target = config['masks_target']
    graph_target = config['graph_target']
    cities = config['cities']
    num_workers = config['num_workers']
    batch_size = config['batch_size']
    filter_test_times = config['filter_test_times']
    shuffle = config['shuffle']
    threshold = config['threshold']
    delete_masks = config['delete_masks']
    delete_graphs = config['delete_graphs']
    mask_type = config['mask_type']
    k_list = config['k_list']
    
    # If Mask folder exists, delete masks if selected
    # If folder doesn't exist, create it.
    if os.path.exists(masks_target):
        if delete_masks:
            for element in os.listdir(masks_target):
                if element != 'full_masks':
                    os.remove(os.path.join(masks_target, element))            
    else:
        os.mkdir(masks_target)
    
    # If Graph folder exists, delete graphs if selected 
    # If folder doesn't exist, create it.
    if os.path.exists(graph_target):
        if delete_graphs:
            for element in os.listdir(graph_target):
                shutil.rmtree(os.path.join(graph_target, element))            
    else:
        os.mkdir(graph_target)
    mask_type = 'xxx'
    # Preprocessing for every city
    for city in cities:
        
        path = os.path.join(masks_target, city + '.mask')
        
        # If mask is available, load it
        if os.path.exists(path):
            mask = pickle.load(open(path, 'rb'))
        else:
            # Create threshold mask
            if mask_type != 'OSM':
                # get sum array
                sumDict = mask_creator(source_root = source_root, target_root = target_root, masks_target = masks_target, city = city, filter_test_times=filter_test_times, num_workers=0, batch_size=4, shuffle=shuffle)
                maskSum = sumDict['sum']
                # Apply threshold
                mask = (maskSum > threshold)
                pickle.dump(mask,open(path,'wb'))
            
            # Create OSM mask
            else:
                mask = getMaskFromOSM(city, masks_target)
                
        mask_idx = np.where(mask)
        
        # Create a graph network in the size of the image
        m,n = mask.shape
        G = nx.grid_2d_graph(m, n)
        
        # Add connections between all neighboring nodes
        rows=range(m)
        columns=range(n)
        G.add_edges_from( ((i,j),(i-1,j-1)) for i in rows for j in columns if i>0 and j>0 )
        G.add_edges_from( ((i,j),(i-1,j+1)) for i in rows for j in columns if i>0 and j<max(columns))
        
        # Only keep nodes that are part of the mask
        inv_mask = ~mask
        del_nodes = np.where(inv_mask)
        del_nodes = list(zip(del_nodes[0],del_nodes[1]))
        
        G.remove_nodes_from(del_nodes)
        
        # Build adjacency matrix
        A = nx.to_scipy_sparse_matrix(G)
        
        # Add self loops
        A = A + scipy.sparse.identity(A.shape[0], dtype='bool',format='csr')
        
        # Store connectivity info as tensor
        edge_tuple = A.nonzero()
        edge_array = np.stack(edge_tuple)
        edge_array = edge_array.astype(np.int64)
        edge_index = torch.LongTensor(edge_array)
        
        # Save edge index as file
        filename = city + '.edge'
        path = os.path.join(masks_target,filename)
        if not os.path.exists(path):
            pickle.dump(edge_index, open(path, 'wb'))
            
            
        # For all selected kernel sizes, create cluster files (if not available yet)
        for k in k_list:
            filename = city + str(k) + '.clusters'
            path = os.path.join(masks_target,filename)
            if not os.path.exists(path):
                getStreetClusters(city, masks_target, k)
                
        
        # Data loader (loads full arrays from the files)
        dataset_train = traffic_dataset(source_root, target_root,
                                                split_type='training', cities = [city])
        dataset_val = traffic_dataset(source_root, target_root,
                                                split_type='validation', cities = [city])
        dataset_test = traffic_dataset(source_root, target_root,
                                                split_type='test', cities = [city])
        
        train_loader = torch.utils.data.DataLoader(dataset_train, num_workers = num_workers, batch_size = batch_size, shuffle = shuffle)
        val_loader = torch.utils.data.DataLoader(dataset_val, num_workers = num_workers, batch_size = batch_size, shuffle = shuffle)
        test_loader = torch.utils.data.DataLoader(dataset_test, num_workers = num_workers, batch_size = batch_size, shuffle = shuffle)
        
        # Build folders if not available yet
        path = os.path.join(graph_target,city)
        if not os.path.exists(path):
            os.mkdir(path)
            
        # For the training dataset
        path = os.path.join(path,'train')
        if not os.path.exists(path):
            os.mkdir(path)
        
        
        for i, imdata in enumerate(train_loader, 0):
            x_im, date = imdata
            
            # Define file name
            date = str(date)
            date = date[2:10]
            file_name = str(date) + '.h5'
            
            # Convert to graph and save
            path = os.path.join(graph_target,city,'train',file_name)
            if not os.path.exists(path):
                x = x_im[...,mask_idx[0],mask_idx[1]]
                x = x[0,:,:,:]
                size = x.size()[2]
                f_target = h5py.File(path, 'w')
                dset = f_target.create_dataset('array', (288, 3, size),
                                           chunks=(1, 3, size),
                                           dtype='uint8', data=x,
                                           compression=None)
                f_target.close()

        # For the validation dataset
        path = os.path.join(graph_target,city,'validate')
        if not os.path.exists(path):
            os.mkdir(path)
            
            
        
        for i, imdata in enumerate(val_loader, 0):
            x_im, date = imdata
            
            # Define file name
            date = str(date)
            date = date[2:10]
            file_name = str(date) + '.h5'
            
            # Convert to graph and save
            path = os.path.join(graph_target,city,'validate',file_name)
            if not os.path.exists(path):
                x = x_im[...,mask_idx[0],mask_idx[1]]
                x = x[0,:,:,:]
                size = x.size()[2]
                f_target = h5py.File(path, 'w')
                dset = f_target.create_dataset('array', (288, 3, size),
                                           chunks=(1, 3, size),
                                           dtype='uint8', data=x,
                                           compression=None)

                f_target.close()
        
        # For the test dataset
        path = os.path.join(graph_target,city,'test')
        if not os.path.exists(path):
            os.mkdir(path)
            
            
        
        for i, imdata in enumerate(test_loader, 0):
            x_im, date = imdata
            
            # Define file name
            date = str(date)
            date = date[2:10]
            file_name = str(date) + '.h5'
        
            path = os.path.join(graph_target,city,'test',file_name)
            
            # Convert to graph and save
            if not os.path.exists(path):
                x = x_im[...,mask_idx[0],mask_idx[1]]
                x = x[0,:,:,:]
                size = x.size()[2]
                f_target = h5py.File(path, 'w')
                dset = f_target.create_dataset('array', (288, 3, size),
                                           chunks=(1, 3, size),
                                           dtype='uint8', data=x,
                                           compression=None)

                f_target.close()