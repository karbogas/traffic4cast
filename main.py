# -*- coding: utf-8 -*-
"""
@author: karbogas
"""

from configMain import config
import torch
import torch_geometric
from utils.graphloader import geometric_dataset
from utils.modelCollection import KipfNet, KipfNetDoublePool
from utils.visual_TB import Visualizer
import os
from datetime import datetime
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import random

def networkTraining(model, train_loader, val_loader, epochs, learning_rate, device, log_path, includeHeading):
    """
        Do the whole training
    """   
    
    
    print('------ STARTING TRAINING ------')
    print('Number of epochs: ', epochs)
    print('Learning rate: ', learning_rate)
    print('Batch Size: ', config['batch_size'])
    print('City: ', config['city'])
    print('-' * 31)
    
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Initialize log file
    writer = Visualizer(log_path)
    
    # dump config file  
    with open(os.path.join(log_path,'config.json'), 'w') as fp:
        json.dump(config, fp)
        
    startTime = time.time()
    iterator = 0
    
    val_loss_min = np.Inf
    counter = 0
    
    # For each epoch
    for epoch in range(epochs):
        writer.write_lr(optimizer, iterator)

        # train for one epoch
        iterator = training(model, train_loader, optimizer, device, writer, epoch, iterator)
        
        # Early stopping (training failed)
        if iterator == -1:
            duration = time.time() - startTime
            print("Training finished (Error), took {:.2f}s".format(duration))
            break
        
        # get validation loss and save images
        valLoss = validation(model, val_loader, device, writer, iterator, log_path, includeHeading)
        
        # Early stopping (didn't improve for 2 epochs)
        if valLoss < val_loss_min:
            torch.save(model.state_dict(), os.path.join(log_path, 'checkpoint.pt'))
            val_loss_min = valLoss
            counter = 0
        elif counter == 1:
            duration = time.time() - startTime
            print("Training finished (early), took {:.2f}s".format(duration))
            break
        else:
            counter += 1
    
    # Dump statistics in tensorboard file
    duration = time.time() - startTime
    print("Training finished, took {:.2f}s".format(duration))
    
    writer.write_text('{:.2f}'.format(duration), 'Time')
    
    if device != 'cpu':
        mem = torch.cuda.max_memory_allocated(device)
        mem = mem // 1048576
        writer.write_text('{:d}'.format(mem), 'Memory')
    writer.close()
    
    
    
def training(model, train_loader, optimizer, device, writer, epoch, iterator):
    """
        Train for one epoch
    """
    model.train()
    running_loss = 0.0
    numBatches = len(train_loader)
    final = False
    start = True
    # define start time
    start_time = time.time()
    
    # For each batch
    for i, data in enumerate(train_loader, 0):
        
        data = data.to(device)
        
        # Set the gradients to zero
        optimizer.zero_grad()        
        
        # If first or last batch            
        if i == 1:
            start = False
        
        if i+1 == numBatches:
            final = True
        
        # Forward pass, backward pass, optimize
        prediction = model(data, final, start)
        
        # Calculate loss of the batch
        loss = torch.nn.functional.mse_loss(prediction, data.y)
        
        loss.backward()
        optimizer.step()
        
        # Progress bar and dump to tensorboard file
        running_loss += loss.item()
        if (i+1) % 10 == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch+1, int(100 * (i+1) / numBatches), running_loss / 10, time.time() - start_time))

            # write the train loss to tensorboard
            running_loss_norm = running_loss / 10
            writer.write_loss_train(running_loss_norm, iterator)

            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
        
        # set iterator to -1 if training failed
        if torch.isnan(loss):
            iterator = -1
            valLoss = 'Converged to NaN'
            pickle.dump(valLoss, open(os.path.join(log_path,'valLoss.data'), 'wb'))
            break
            
        iterator += 1
    
    return iterator


def validation(model, val_loader, device, writer, iterator, log_path, includeHeading):
    """
        Calculates validation loss and dumps images to tensorboard file
    """
    model.eval()
    total_loss = 0.0
    
    numBatches = len(val_loader)
    final = False
    start = True
    
    # for each batch
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            
            data = data.to(device)
            
            # If first or last batch    
            if i == 1:
                start = False
        
            if i+1 == numBatches:
                final = True
                
            # Apply model to data
            prediction = model(data, final, start)
            
            # Calculate loss of the batch
            prediction = torch.clamp(prediction, 0, 255, out=None)
            loss = torch.nn.functional.mse_loss(prediction, data.y)
            total_loss += loss.item()
            
            # Dump images from first batch
            if i == 0:
                               
                # Retransform to images
                y_predict = retransformToImage(prediction.cpu().detach().numpy(), includeHeading)
                y_true = retransformToImage(data.y.cpu().detach().numpy(), includeHeading)
                
                # Dump images
                writer.write_image(y_predict, iterator, if_predict=True, includeHeading = includeHeading)
                writer.write_image(y_true, iterator, if_predict=False, includeHeading = includeHeading)
            
    # Print and dump Total validation loss        
    valLoss = total_loss / len(val_loader)
    print("Validation loss = {:.2f}".format(valLoss))
    # write the validation loss to tensorboard
    writer.write_loss_validation(valLoss, iterator)
    
    pickle.dump(valLoss, open(os.path.join(log_path,'valLoss.data'), 'wb'))
    
    return valLoss
    

def retransformToImage(blockdata, includeHeading):
    """
        Retransforms graph data into image form
        
        Input:
            blockdata:      One batch of graph data
            includeHeading: Is the heading channel included?
                
        Output:
            images:         The same batch in image form
    """
    
    batch_size = config['batch_size']
    city = config['city']
    mask_storage = config['masks_target']
    
    # Load mask
    mask = pickle.load(open(os.path.join(mask_storage, city + '.mask'),'rb'))
    
    indexes = np.where(mask)
    
    # Retransform batch data
    data_list = np.split(blockdata, batch_size)
    
    data_vector = np.stack(data_list)
    data_vector = np.moveaxis(data_vector, 2,1)
    
    # Is the heading channel included?
    if includeHeading:
        outputSize = 6
    else:
        outputSize = 9
    
    # Initialize empty images
    images = np.zeros((batch_size, outputSize, 495, 436))
    
    # Fill in graph data
    images[...,indexes[0],indexes[1]] = data_vector
    
    images = torch.from_numpy(images)
    
    return images



if __name__ == '__main__':
    
    # Load config
    graph_storage = config['graph_target']
    mask_storage = config['masks_target']
    city = config['city']
    num_workers = config['num_workers']
    filter_test_times = config['filter_test_times']
    shuffle = config['shuffle']
    epochs = config['epochs']
    device = torch.device(config['device'])
    #device = torch.device('cpu')
    log_target = config['log_target']
    includeHeading = config['includeHeading']
    
    
    # Define parameter lists for random search
    double_cluster_list = [False, True]
    cluster_k_single_list = [3,4,6,8]
    cluster_k_double_list = [[3,6],[4,8]]
    layers_list = [1,2,3]
    coords_list = [True]
    n_hidden_list = [4,8,16,32,64]
    n_hidden2_list = [4,8,16,32,64]
    n_hidden3_list = [4,8,16,32,64]
    conv_list = ['Cheb', 'Graph']
    K_block_list = [4,6,10]
    K_mix_list = [1,2,4]
    skipconv_list = [True]
    p_list = [0.25, 1]
    batch_size_list = [2,4,6]
    learning_rate_list = [0.001, 0.005, 0.01]
    
    # Load mask and edge index
    edgepath = os.path.join(mask_storage, city + '.edge')
    edge = pickle.load(open(edgepath,'rb'))
    
    maskpath = os.path.join(mask_storage, city + '.mask')
    mask = pickle.load(open(maskpath,'rb'))
    
    # Get node position tensor from mask
    maskpos = np.where(mask)
    arr = np.asarray(maskpos)
    posTensor = torch.from_numpy(arr.transpose()).float()
    
    # Define data loader
    dataset_train = geometric_dataset(root = graph_storage, edge_index = edge, posTensor = posTensor, city = city, filter_test_times = filter_test_times, split_type='train', includeHeading = includeHeading)
    dataset_val = geometric_dataset(root = graph_storage, edge_index = edge, posTensor = posTensor, city = city, filter_test_times = filter_test_times, split_type='validate', includeHeading = includeHeading)
    
    # Do 500 random search runs
    randomsearch = 0
    while randomsearch < 500:
        
        # Randomly pick every parameter - And change it in the config
        double_cluster = random.choice(double_cluster_list)
        config['double_cluster'] = double_cluster
        cluster_k_single = random.choice(cluster_k_single_list)
        config['cluster_k_single'] = cluster_k_single
        cluster_k_double = random.choice(cluster_k_double_list)
        config['cluster_k_double'] = cluster_k_double
        layers = random.choice(layers_list)
        config['layers'] = layers
        coords = random.choice(coords_list)
        config['coords'] = coords
        n_hidden = random.choice(n_hidden_list)
        config['n_hidden'] = n_hidden
        n_hidden2 = random.choice(n_hidden2_list)
        config['n_hidden2'] = n_hidden2
        n_hidden3 = random.choice(n_hidden3_list)
        config['n_hidden3'] = n_hidden3
        conv = random.choice(conv_list)
        config['conv'] = conv
        K_block = random.choice(K_block_list)
        config['K_block'] = K_block
        K_mix = random.choice(K_mix_list)
        config['K_mix'] = K_mix
        skipconv = random.choice(skipconv_list)
        config['skipconv'] = skipconv
        p = random.choice(p_list)
        config['p'] = p
        batch_size = random.choice(batch_size_list)
        config['batch_size'] = batch_size
        learning_rate = random.choice(learning_rate_list)
        config['learning_rate'] = learning_rate
        
        
        
        # Dataloader can only be initiated here (due to batch size and learning rate)
        train_loader = torch_geometric.data.DataLoader(dataset_train, shuffle=shuffle, batch_size = batch_size, num_workers = num_workers)
        val_loader = torch_geometric.data.DataLoader(dataset_val, shuffle=shuffle, batch_size = batch_size, num_workers = num_workers)
        
        # If two pooled branches
        if double_cluster:
            
            # Get both kernel sizes
            cluster_k1 = cluster_k_double[0]
            cluster_k2 = cluster_k_double[1]
            
            # Load cluster files for both pooled branches
            clusterpath1 = os.path.join(mask_storage, city + str(cluster_k1) + '.clusters')
            clusters1 = torch.from_numpy(pickle.load(open(clusterpath1,'rb'))).to(device)
            clusterpath2 = os.path.join(mask_storage, city + str(cluster_k2) + '.clusters')
            clusters2 = torch.from_numpy(pickle.load(open(clusterpath2,'rb'))).to(device)
            maxCluster1 = torch.max(clusters1) + 1
            maxCluster2 = torch.max(clusters2) + 1
        
            # Load category file
            catpath = os.path.join(mask_storage, city + str(cluster_k1) + '.cats')
            categories = torch.from_numpy(pickle.load(open(catpath,'rb'))).to(device)
            
            # Initialize model
            model = KipfNetDoublePool(clusters1 = clusters1, maxCluster1 = maxCluster1,clusters2 = clusters2, clustering = 'Street', maxCluster2 = maxCluster2, categories = categories, coords = coords, n_hidden = n_hidden, n_hidden2 = n_hidden2, n_hidden3 = n_hidden3, K_block = K_block, K_mix = K_mix, skipconv = skipconv, conv = conv, layers = layers, p = p, includeHeading = includeHeading).to(device)
        
        # If one pooled branch
        else:
            # Load cluster and category files
            clusterpath = os.path.join(mask_storage, city + str(cluster_k_single) + '.clusters')
            clusters = torch.from_numpy(pickle.load(open(clusterpath,'rb'))).to(device)
            maxCluster = torch.max(clusters) + 1
            catpath = os.path.join(mask_storage, city + str(cluster_k_single) + '.cats')
            categories = torch.from_numpy(pickle.load(open(catpath,'rb'))).to(device)
            
            # Initialize model
            model = KipfNet(clusters = clusters, maxCluster = maxCluster, clustering = 'Street', categories = categories, coords = coords, n_hidden = n_hidden, n_hidden2 = n_hidden2, n_hidden3 = n_hidden3, K_block = K_block, K_mix = K_mix, skipconv = skipconv, conv = conv, layers = layers, p = p, includeHeading = includeHeading).to(device)
        
        # Folder where the run should be saved
        log_path = os.path.join(log_target, city + '_deviceNo-' + str(config['device']) + '_Iteration-' + str(randomsearch))
            
        # Do the training
        try:
            networkTraining(model, train_loader, val_loader, epochs, learning_rate, device, log_path, includeHeading)
    
        except RuntimeError:
            print('Out of Memory error')
            valLoss = 'Out of Memory error'
            pickle.dump(valLoss, open(os.path.join(log_path,'valLoss.data'), 'wb'))
            
        torch.cuda.reset_peak_memory_stats()
        
        randomsearch += 1