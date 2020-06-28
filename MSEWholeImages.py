# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:34:53 2020

@author: karbogas
"""

from configValidation import config
from doubleloader import both_datasets
import pickle
import os
import torch_geometric
import torch
from modelCollection import KipfNet, KipfNetDoublePool
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    run_folder = config['run_folder']
    graph_storage = config['graph_folder']
    image_folder = config['image_folder']
    mask_storage = config['mask_folder']
    cities = config['cities']
    filter_test_times = config['filter_test_times']
    device = torch.device(0)
    num_workers = 0
    shuffle = False
    
    # Select parameters
    double_cluster = True
    config['double_cluster'] = double_cluster
    cluster_k_single = 3
    config['cluster_k_single'] = cluster_k_single
    cluster_k_double = [3,6]
    config['cluster_k_double'] = cluster_k_double
    layers = 3
    config['layers'] = layers
    coords = True
    config['coords'] = coords
    n_hidden = 8
    config['n_hidden'] = n_hidden
    n_hidden2 = 64
    config['n_hidden2'] = n_hidden2
    n_hidden3 = 32
    config['n_hidden3'] = n_hidden3
    conv = 'Cheb'
    config['conv'] = conv
    K_block = 10
    config['K_block'] = K_block
    K_mix = 1
    config['K_mix'] = K_mix
    skipconv = True
    config['skipconv'] = skipconv
    p = 1
    config['p'] = p
    batch_size = 4
    config['batch_size'] = batch_size
    learning_rate = 0.01
    config['learning_rate'] = learning_rate
    
    # Multiple runs to check
    folder_list = [r'C:\Users\konst\Desktop\runs\Berlin_ChebConv_DoubleCluster_3Layers', r'C:\Users\konst\Desktop\runs\Moscow_ChebConv_DoubleCluster_3Layers', r'C:\Users\konst\Desktop\runs\Istanbul_ChebConv_DoubleCluster_3Layers']
        
    # Multiple runs applied to multiple cities
    for run_folder in folder_list:
        for city in cities:
            
            # Load mask and edge information
            edgepath = os.path.join(mask_storage, city + '.edge')
            edge_index = pickle.load(open(edgepath,'rb'))
            
            maskpath = os.path.join(mask_storage, city + '.mask')
            mask = pickle.load(open(maskpath,'rb'))
            
            # Get position tensor from mask
            maskpos = np.where(mask)
            arr = np.asarray(maskpos)
            posTensor = torch.from_numpy(arr.transpose()).float()
            
            # dataloader (graph and image)
            dataset = both_datasets(image_folder,graph_storage, city, filter_test_times)
            
            val_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size = 1, num_workers = 0)
            
            # get pooling kernel sizes
            cluster_k1 = cluster_k_double[0]
            cluster_k2 = cluster_k_double[1]
            
            # load cluster files
            clusterpath1 = os.path.join(mask_storage, city + str(cluster_k1) + '.clusters')
            clusters1 = torch.from_numpy(pickle.load(open(clusterpath1,'rb'))).to(device)
            clusterpath2 = os.path.join(mask_storage, city + str(cluster_k2) + '.clusters')
            clusters2 = torch.from_numpy(pickle.load(open(clusterpath2,'rb'))).to(device)
            maxCluster1 = torch.max(clusters1) + 1
            maxCluster2 = torch.max(clusters2) + 1
        
            # Load category file
            catpath = os.path.join(mask_storage, city + str(cluster_k1) + '.cats')
            categories = torch.from_numpy(pickle.load(open(catpath,'rb'))).to(device)
            
            # Load model
            model = KipfNetDoublePool(clusters1 = clusters1, maxCluster1 = maxCluster1,clusters2 = clusters2, clustering = 'Street', maxCluster2 = maxCluster2, categories = categories, coords = coords, n_hidden = n_hidden, n_hidden2 = n_hidden2, n_hidden3 = n_hidden3, K_block = K_block, K_mix = K_mix, skipconv = skipconv, conv = conv, layers = layers, p = p).to(device)
            model.load_state_dict(torch.load(os.path.join(run_folder, 'checkpoint.pt'), map_location=device))
                    
            running_loss = 0.0
            
            model.eval()
            
            numBatches = len(val_loader)
            final = False
            start = True
            batch = torch.tensor([0])
            
            with torch.no_grad():
                # For each batch
                for idx, (x,y,imY) in enumerate(val_loader, 0):
                    x = x[0,:,:]
                    y = y[0,:,:]
                    imY = imY[0,:,:,:].to(device)
                    
                    # Build graph data and apply to device
                    graph = torch_geometric.data.Data(x=x,y=y,edge_index=edge_index, pos=posTensor)
                    graph.batch = batch
                    graph = graph.to(device)
                    
                    if idx == 1:
                        start = False
                    
                    if idx+1 == numBatches:
                        final = True
                    
                    # Forward pass, backward pass, optimize
                    prediction = model(graph, final, start)
                    
                    # Clamp to possible area
                    prediction = torch.clamp(prediction, 0, 255, out=None)
                    
                    # Transform prediction to image form
                    y_predict = prediction.cpu().detach().numpy()
                    y_predict = np.moveaxis(y_predict, 1,0)
                    predict_im = np.zeros((6, 495, 436))
                    predict_im[:,maskpos[0],maskpos[1]] = y_predict
                    predict_im = torch.from_numpy(predict_im).float().to(device)
                    
                    
                    # Calculate loss on whole image
                    loss = torch.nn.functional.mse_loss(predict_im, imY)
                    # Add to running loss
                    running_loss = running_loss + loss
                    
                    # Progress information
                    if (idx+1) % 10 == 0:
                        print('{:d} % done'.format(int(100 * (idx+1) / len(val_loader))))
                    
            # Print and save MSE
            print('MSE: {:.2f}'.format(running_loss / len(val_loader)))
            
            pickle.dump(running_loss / len(val_loader), open(os.path.join(run_folder, city + '.mse'), 'wb'))