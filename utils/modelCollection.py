# -*- coding: utf-8 -*-
"""
@author: karbogas
"""

import torch
from torch_geometric.nn import ChebConv, max_pool, knn, GCNConv, GraphConv, SAGEConv
from torch_geometric.utils import add_self_loops
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torch_scatter import scatter_add


# Defines the convolution block
class Kipfblock(torch.nn.Module):
    def __init__(self, n_input, n_hidden=64, K=6, p=0.5, bn=False, conv = 'Cheb'):
        super(Kipfblock, self).__init__()
        
        # Pick convolution technique
        if conv == 'Cheb':
            self.conv1 = ChebConv(n_input, n_hidden, K=K)
        elif conv == 'GCN':
            self.conv1 = GCNConv(n_input, n_hidden)
        elif conv == 'SAGE':
            self.conv1 = SAGEConv(n_input, n_hidden)  
        elif conv == 'Graph':
            self.conv1 = GraphConv(n_input, n_hidden)  
        
        self.p = p
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.do_bn = bn
        if bn:
            self.bn = BatchNorm1d(n_hidden)

    def forward(self, x, edge_index):
        # convolutional layer + optional batch normalization + relu
        if self.do_bn:
            x = F.relu(self.bn(self.conv1(x, edge_index)))
        else:
            x = F.relu(self.conv1(x, edge_index))

        return x

# Model with single pooling 
class KipfNet(torch.nn.Module):
    def __init__(self, clusters = None, knn = 4, maxCluster = 0, clustering = 'None',categories = None, coords = False, n_hidden = 64, n_hidden2 = 32, n_hidden3 = 16, K_block = 6, K_mix = 1, skipconv = False, do_bn = True, conv = 'Cheb', layers = 1, midSkip = False, p = 0.5, includeHeading = False):
        super(KipfNet, self).__init__()
        
        self.clusters = clusters
        self.knn = knn
        self.maxCluster = maxCluster
        self.clustering = clustering
        self.categories = categories - 1
        self.skipconv = skipconv
        self.coords = coords
        self.midSkip = midSkip
        self.layers = layers
        self.bn = BatchNorm1d(n_hidden)
        self.bn2 = BatchNorm1d(n_hidden2)
        self.bn3 = BatchNorm1d(n_hidden3)
        self.p = p
        
        # Input size depends on heading channel
        if includeHeading:
            n_input = 36
        else:
            n_input = 24
        
        # Add coordinates and categories to input
        if coords:
            n_input = n_input + 2
            if categories is not None:
                n_input = n_input + 1
        if layers == 1:
            midSkip = False
            
        # Build model (selected number of convolution blocks)
        self.m# Build model (selected number of convolution blocks)oduleList1 = torch.nn.ModuleList()
        self.skipList1 = torch.nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.moduleList1.append(Kipfblock(n_input=n_input, n_hidden=n_hidden, K=K_block, bn=do_bn, conv = conv))
                n_mix = n_hidden + n_input
            elif i == 1:
                self.moduleList1.append(Kipfblock(n_input=n_hidden, n_hidden=n_hidden2, K=K_block, bn=do_bn, conv = conv))
                n_mix = n_hidden2 + n_hidden
            else:
                self.moduleList1.append(Kipfblock(n_input=n_hidden2, n_hidden=n_hidden3, K=K_block, bn=do_bn, conv = conv))
                n_mix = n_hidden3 + n_hidden2
            if midSkip:
                if i == 0:
                    if conv == 'Cheb':
                        self.skipList1.append(ChebConv(n_mix, n_hidden, K=K_mix))
                    elif conv == 'GCN':
                        self.skipList1.append(GCNConv(n_mix, n_hidden))
                    elif conv == 'SAGE':
                        self.skipList1.append(SAGEConv(n_mix, n_hidden)) 
                    elif conv == 'Graph':
                        self.skipList1.append(GraphConv(n_mix, n_hidden))
                elif i == 1:
                    if conv == 'Cheb':
                        self.skipList1.append(ChebConv(n_mix, n_hidden2, K=K_mix))
                    elif conv == 'GCN':
                        self.skipList1.append(GCNConv(n_mix, n_hidden2))
                    elif conv == 'SAGE':
                        self.skipList1.append(SAGEConv(n_mix, n_hidden2)) 
                    elif conv == 'Graph':
                        self.skipList1.append(GraphConv(n_mix, n_hidden2))
                else:
                    if conv == 'Cheb':
                        self.skipList1.append(ChebConv(n_mix, n_hidden3, K=K_mix))
                    elif conv == 'GCN':
                        self.skipList1.append(GCNConv(n_mix, n_hidden3))
                    elif conv == 'SAGE':
                        self.skipList1.append(SAGEConv(n_mix, n_hidden3)) 
                    elif conv == 'Graph':
                        self.skipList1.append(GraphConv(n_mix, n_hidden3))
                        
        #  Pooled Branch (selected number of convolution blocks)
        if clustering != 'None':
            self.moduleList2 = torch.nn.ModuleList()
            self.skipList2 = torch.nn.ModuleList()
            for i in range(layers):
                if i == 0:
                    self.moduleList2.append(Kipfblock(n_input=n_input, n_hidden=n_hidden, K=K_block, bn=do_bn, conv = conv))
                    n_mix = n_hidden + n_input
                elif i == 1:
                    self.moduleList2.append(Kipfblock(n_input=n_hidden, n_hidden=n_hidden2, K=K_block, bn=do_bn, conv = conv))
                    n_mix = n_hidden2 + n_hidden
                else:
                    self.moduleList2.append(Kipfblock(n_input=n_hidden2, n_hidden=n_hidden3, K=K_block, bn=do_bn, conv = conv))
                    n_mix = n_hidden3 + n_hidden2
                if midSkip:
                    if i == 0:
                        if conv == 'Cheb':
                            self.skipList2.append(ChebConv(n_mix, n_hidden, K=K_mix))
                        elif conv == 'GCN':
                            self.skipList2.append(GCNConv(n_mix, n_hidden))
                        elif conv == 'SAGE':
                            self.skipList2.append(SAGEConv(n_mix, n_hidden)) 
                        elif conv == 'Graph':
                            self.skipList2.append(GraphConv(n_mix, n_hidden))
                    elif i == 1:
                        if conv == 'Cheb':
                            self.skipList2.append(ChebConv(n_mix, n_hidden2, K=K_mix))
                        elif conv == 'GCN':
                            self.skipList2.append(GCNConv(n_mix, n_hidden2))
                        elif conv == 'SAGE':
                            self.skipList2.append(SAGEConv(n_mix, n_hidden2)) 
                        elif conv == 'Graph':
                            self.skipList2.append(GraphConv(n_mix, n_hidden2))
                    else:
                        if conv == 'Cheb':
                            self.skipList2.append(ChebConv(n_mix, n_hidden3, K=K_mix))
                        elif conv == 'GCN':
                            self.skipList2.append(GCNConv(n_mix, n_hidden3))
                        elif conv == 'SAGE':
                            self.skipList2.append(SAGEConv(n_mix, n_hidden3)) 
                        elif conv == 'Graph':
                            self.skipList2.append(GraphConv(n_mix, n_hidden3))
        
        # Input size for final convolution
        if layers == 1:
            n_mix = n_hidden
        elif layers == 2:
            n_mix = n_hidden2
        else:
            n_mix = n_hidden3
        
        if clustering != 'None':
            n_mix = n_mix * 2
            
        if skipconv:
            n_mix = n_mix + n_input
        
        # Output size depends on heading channel
        if includeHeading:
            n_output = 9
        else:
            n_output = 6
        
        # Select convolution type
        if conv == 'Cheb':
            self.conv_mix = ChebConv(n_mix, n_output, K=K_mix)
        elif conv == 'GCN':
            self.conv_mix = GCNConv(n_mix, n_output)
        elif conv == 'SAGE':
            self.conv_mix = SAGEConv(n_mix, n_output)  
        elif conv == 'Graph':
            self.conv_mix = GraphConv(n_mix, n_output)  
            
        
    def forward(self, data, final, start):
        x, edge_index, pos, batch = data.x, data.edge_index, data.pos, data.batch
        
        x_start = x
        
        # If no pooling
        if self.clustering == 'None':
            # Perform convolution blocks
            for i in range(self.layers):
                x_temp = x
                x = self.moduleList1[i](x,edge_index)
                if self.midSkip:
                    x = torch.cat((x, x_temp), 1)
                    x = self.skipList1[i](x,edge_index)
            
            # Input size of final convolution
            if self.skipconv:
                y = torch.cat((x, x_start),1)
            else:
                y = x
                
            # Do final convolution
            y = self.conv_mix(y, edge_index)
        
            # Add dropout layer
            if self.p != 1:
                y = F.dropout(y, training=self.training, p=self.p)
        
        
        # If grid based pooling
        elif self.clustering == '4x4':
            batchClusters = self.clusters
        
            batch_size = torch.max(batch) + 1
            
            # Divide clusters from different batches
            for i in range(1,batch_size):
                batchClusters = torch.cat((batchClusters, self.clusters + i*self.maxCluster))
            
            # Pooled branch, max pooling
            data = max_pool(batchClusters, data)
            x_t, edge_index_t, pos_t, batch_t = data.x, data.edge_index, data.pos, data.batch
            
            edge_index_t, temp = add_self_loops(edge_index_t)
            
            # Add coordinates to input
            if self.coords:
                normPos = pos / torch.max(pos)
                normPos_t = pos_t / torch.max(pos_t)
        
                x = torch.cat((x, normPos),1)
                x_t = torch.cat((x_t, normPos_t),1)
    
    
            # Perform convolution blocks in both branches
            for i in range(self.layers):
                x_temp = x
                x = self.moduleList1[i](x,edge_index)
                if self.midSkip:
                    x = torch.cat((x, x_temp), 1)
                    if i == 0:
                        bn = self.bn
                    elif i == 1:
                        bn = self.bn2
                    else:
                        bn = self.bn3
                        
                    x = F.relu(bn(self.skipList1[i](x,edge_index)))
            
            for i in range(self.layers):
                x_ttemp = x_t
                x_t = self.moduleList2[i](x_t,edge_index_t)
                if self.midSkip:
                    x_t = torch.cat((x_t, x_ttemp), 1)
                    if i == 0:
                        bn = self.bn
                    elif i == 1:
                        bn = self.bn2
                    else:
                        bn = self.bn3
                    x_t = F.relu(bn(self.skipList2[i](x_t,edge_index_t)))
            
            # Calculate knn weights for first batch (and last, since the size might be different)
            if start:
                pairs = knn(pos_t,pos,self.knn, batch_x = batch_t, batch_y = batch)
                yIdx, xIdx = pairs
                diff = pos_t[xIdx] - pos[yIdx]
                squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
                weights = 1.0 / torch.clamp(squared_distance, min = 1e-16)
                
                self.weights = weights
                self.xIdx = xIdx
                self.yIdx = yIdx
                
            if final:
                pairs = knn(pos_t,pos,self.knn, batch_x = batch_t, batch_y = batch)
                yIdx, xIdx = pairs
                diff = pos_t[xIdx] - pos[yIdx]
                squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
                weights = 1.0 / torch.clamp(squared_distance, min = 1e-16)
                
                self.weights = weights
                self.xIdx = xIdx
                self.yIdx = yIdx
                
                
            
            # Unpool pooled branch
            x_t = scatter_add(x_t[self.xIdx] * self.weights, self.yIdx, dim = 0, dim_size=pos.size(0))
            x_t = x_t / scatter_add(self.weights, self.yIdx, dim = 0, dim_size=pos.size(0))
            
            # Input size of final convolution
            if self.skipconv:
                y = torch.cat((x, x_t, x_start),1)
            else:
                y = torch.cat((x, x_t),1)
            
            # Do final convolution
            y = self.conv_mix(y, edge_index)
            
            # Add dropout layer
            if self.p != 1:
                y = F.dropout(y, training=self.training, p=self.p)
            
        # If street based pooling
        elif self.clustering == 'Street':
            batchClusters = self.clusters
            batchCat = self.categories
            
            batch_size = torch.max(batch) + 1
            
            # Divide clusters and categories from different batches
            for i in range(1,batch_size):
                batchClusters = torch.cat((batchClusters, self.clusters + i*self.maxCluster))
                batchCat = torch.cat((batchCat, self.categories + i * 5))
            
            batchCat = batchCat.long()
            data.batch = batchCat
            
            
            # Pooled branch, max pooling
            data = max_pool(batchClusters, data)
            x_t, edge_index_t, pos_t, batchCat_t = data.x, data.edge_index, data.pos, data.batch
            
            edge_index_t, temp = add_self_loops(edge_index_t)
    
            # Add coordinates and categories to input
            if self.coords:
                cats = (batchCat % 5).float()
                catsT = (batchCat_t % 5).float()
        
                normPos = pos / torch.max(pos)
                normPos_t = pos_t / torch.max(pos_t)
                normCat = (cats / 4).view(batchCat.size(0),1)
                normCat_t = (catsT / 4).view(batchCat_t.size(0),1)
        
                x = torch.cat((x, normPos, normCat),1)
                x_t = torch.cat((x_t, normPos_t, normCat_t),1)
    
            # Perform convolution blocks in both branches
            for i in range(self.layers):
                x_temp = x
                x = self.moduleList1[i](x,edge_index)
                if self.midSkip:
                    x = torch.cat((x, x_temp), 1)
                    if i == 0:
                        bn = self.bn
                    elif i == 1:
                        bn = self.bn2
                    else:
                        bn = self.bn3
                        
                    x = F.relu(bn(self.skipList1[i](x,edge_index)))
            
            for i in range(self.layers):
                x_ttemp = x_t
                x_t = self.moduleList2[i](x_t,edge_index_t)
                if self.midSkip:
                    x_t = torch.cat((x_t, x_ttemp), 1)
                    if i == 0:
                        bn = self.bn
                    elif i == 1:
                        bn = self.bn2
                    else:
                        bn = self.bn3
                    x_t = F.relu(bn(self.skipList2[i](x_t,edge_index_t)))
            
            # Calculate knn weights for first batch (and last, since the size might be different)
            if start:
                sorter = torch.argsort(batchCat)        
                backsorter = torch.argsort(sorter)
                
                pos = pos[sorter]
                batchCat = batchCat[sorter]
            
                pairs = knn(pos_t,pos,self.knn, batch_x = batchCat_t, batch_y = batchCat)
                yIdx, xIdx = pairs
                diff = pos_t[xIdx] - pos[yIdx]
                squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
                weights = 1.0 / torch.clamp(squared_distance, min = 1e-16)
                
                self.weights = weights
                self.xIdx = xIdx
                self.yIdx = yIdx
                self.backSorter = backsorter
                
            if final:
                sorter = torch.argsort(batchCat)        
                backsorter = torch.argsort(sorter)
                
                pos = pos[sorter]
                batchCat = batchCat[sorter]
            
                pairs = knn(pos_t,pos,self.knn, batch_x = batchCat_t, batch_y = batchCat)
                yIdx, xIdx = pairs
                diff = pos_t[xIdx] - pos[yIdx]
                squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
                weights = 1.0 / torch.clamp(squared_distance, min = 1e-16)
                
                self.weights = weights
                self.xIdx = xIdx
                self.yIdx = yIdx
                self.backSorter = backsorter
                
            
            # Unpool pooled branch
            x_t = scatter_add(x_t[self.xIdx] * self.weights, self.yIdx, dim = 0, dim_size=pos.size(0))
            x_t = x_t / scatter_add(self.weights, self.yIdx, dim = 0, dim_size=pos.size(0))
            
            x_t = x_t[self.backSorter]
            
            # Input size of final convolution
            if self.skipconv:
                y = torch.cat((x, x_t, x_start),1)
            else:
                y = torch.cat((x, x_t),1)
                
            # Do final convolution
            y = self.conv_mix(y, edge_index)
            
            # Add dropout layer
            if self.p != 1:
                y = F.dropout(y, training=self.training, p=self.p)
            
        return y
    
# Model with double pooling (only street based clustering)
class KipfNetDoublePool(torch.nn.Module):
    def __init__(self, clusters1 = None,clusters2 = None, knn = 4, maxCluster1 = 0, maxCluster2 = 0, clustering = 'None',categories = None,  coords = False, n_hidden = 64, n_hidden2 = 32, n_hidden3 = 16, K_block = 6, K_mix = 1, skipconv = False, do_bn = True, conv = 'Cheb', layers = 1, midSkip = False, p = 0.5, includeHeading = False):
        super(KipfNetDoublePool, self).__init__()
        
        self.clusters1 = clusters1
        self.clusters2 = clusters2
        self.knn = knn
        self.maxCluster1 = maxCluster1
        self.maxCluster2 = maxCluster2
        self.clustering = clustering
        self.categories = categories - 1
        self.skipconv = skipconv
        self.coords = coords
        self.midSkip = midSkip
        self.layers = layers
        self.bn = BatchNorm1d(n_hidden)
        self.bn2 = BatchNorm1d(n_hidden2)
        self.bn3 = BatchNorm1d(n_hidden3)
        self.p = p
        
        # Input size depends on heading channel
        if includeHeading:
            n_input = 36
        else:
            n_input = 24
        
        # Add coordinates and categories to input
        if coords:
            n_input = n_input + 2
            if categories is not None:
                n_input = n_input + 1
        if layers == 1:
            midSkip = False
            
        # Add coordinates and categories to input
        self.moduleList1 = torch.nn.ModuleList()
        self.skipList1 = torch.nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.moduleList1.append(Kipfblock(n_input=n_input, n_hidden=n_hidden, K=K_block, bn=do_bn, conv = conv))
                n_mix = n_hidden + n_input
            elif i == 1:
                self.moduleList1.append(Kipfblock(n_input=n_hidden, n_hidden=n_hidden2, K=K_block, bn=do_bn, conv = conv))
                n_mix = n_hidden2 + n_hidden
            else:
                self.moduleList1.append(Kipfblock(n_input=n_hidden2, n_hidden=n_hidden3, K=K_block, bn=do_bn, conv = conv))
                n_mix = n_hidden3 + n_hidden2
            if midSkip:
                if i == 0:
                    if conv == 'Cheb':
                        self.skipList1.append(ChebConv(n_mix, n_hidden, K=K_mix))
                    elif conv == 'GCN':
                        self.skipList1.append(GCNConv(n_mix, n_hidden))
                    elif conv == 'SAGE':
                        self.skipList1.append(SAGEConv(n_mix, n_hidden)) 
                    elif conv == 'Graph':
                        self.skipList1.append(GraphConv(n_mix, n_hidden))
                elif i == 1:
                    if conv == 'Cheb':
                        self.skipList1.append(ChebConv(n_mix, n_hidden2, K=K_mix))
                    elif conv == 'GCN':
                        self.skipList1.append(GCNConv(n_mix, n_hidden2))
                    elif conv == 'SAGE':
                        self.skipList1.append(SAGEConv(n_mix, n_hidden2)) 
                    elif conv == 'Graph':
                        self.skipList1.append(GraphConv(n_mix, n_hidden2))
                else:
                    if conv == 'Cheb':
                        self.skipList1.append(ChebConv(n_mix, n_hidden3, K=K_mix))
                    elif conv == 'GCN':
                        self.skipList1.append(GCNConv(n_mix, n_hidden3))
                    elif conv == 'SAGE':
                        self.skipList1.append(SAGEConv(n_mix, n_hidden3)) 
                    elif conv == 'Graph':
                        self.skipList1.append(GraphConv(n_mix, n_hidden3))
                        
        #  Pooled Branch 1 (selected number of convolution blocks)        
        if clustering != 'None':
            self.moduleList2 = torch.nn.ModuleList()
            self.skipList2 = torch.nn.ModuleList()
            for i in range(layers):
                if i == 0:
                    self.moduleList2.append(Kipfblock(n_input=n_input, n_hidden=n_hidden, K=K_block, bn=do_bn, conv = conv))
                    n_mix = n_hidden + n_input
                elif i == 1:
                    self.moduleList2.append(Kipfblock(n_input=n_hidden, n_hidden=n_hidden2, K=K_block, bn=do_bn, conv = conv))
                    n_mix = n_hidden2 + n_hidden
                else:
                    self.moduleList2.append(Kipfblock(n_input=n_hidden2, n_hidden=n_hidden3, K=K_block, bn=do_bn, conv = conv))
                    n_mix = n_hidden3 + n_hidden2
                if midSkip:
                    if i == 0:
                        if conv == 'Cheb':
                            self.skipList2.append(ChebConv(n_mix, n_hidden, K=K_mix))
                        elif conv == 'GCN':
                            self.skipList2.append(GCNConv(n_mix, n_hidden))
                        elif conv == 'SAGE':
                            self.skipList2.append(SAGEConv(n_mix, n_hidden)) 
                        elif conv == 'Graph':
                            self.skipList2.append(GraphConv(n_mix, n_hidden))
                    elif i == 1:
                        if conv == 'Cheb':
                            self.skipList2.append(ChebConv(n_mix, n_hidden2, K=K_mix))
                        elif conv == 'GCN':
                            self.skipList2.append(GCNConv(n_mix, n_hidden2))
                        elif conv == 'SAGE':
                            self.skipList2.append(SAGEConv(n_mix, n_hidden2)) 
                        elif conv == 'Graph':
                            self.skipList2.append(GraphConv(n_mix, n_hidden2))
                    else:
                        if conv == 'Cheb':
                            self.skipList2.append(ChebConv(n_mix, n_hidden3, K=K_mix))
                        elif conv == 'GCN':
                            self.skipList2.append(GCNConv(n_mix, n_hidden3))
                        elif conv == 'SAGE':
                            self.skipList2.append(SAGEConv(n_mix, n_hidden3)) 
                        elif conv == 'Graph':
                            self.skipList2.append(GraphConv(n_mix, n_hidden3))
                            
            #  Pooled Branch 2 (selected number of convolution blocks)                
            self.moduleList3 = torch.nn.ModuleList()
            self.skipList3 = torch.nn.ModuleList()
            for i in range(layers):
                if i == 0:
                    self.moduleList3.append(Kipfblock(n_input=n_input, n_hidden=n_hidden, K=K_block, bn=do_bn, conv = conv))
                    n_mix = n_hidden + n_input
                elif i == 1:
                    self.moduleList3.append(Kipfblock(n_input=n_hidden, n_hidden=n_hidden2, K=K_block, bn=do_bn, conv = conv))
                    n_mix = n_hidden2 + n_hidden
                else:
                    self.moduleList3.append(Kipfblock(n_input=n_hidden2, n_hidden=n_hidden3, K=K_block, bn=do_bn, conv = conv))
                    n_mix = n_hidden3 + n_hidden2
                if midSkip:
                    if i == 0:
                        if conv == 'Cheb':
                            self.skipList3.append(ChebConv(n_mix, n_hidden, K=K_mix))
                        elif conv == 'GCN':
                            self.skipList3.append(GCNConv(n_mix, n_hidden))
                        elif conv == 'SAGE':
                            self.skipList3.append(SAGEConv(n_mix, n_hidden)) 
                        elif conv == 'Graph':
                            self.skipList3.append(GraphConv(n_mix, n_hidden))
                    elif i == 1:
                        if conv == 'Cheb':
                            self.skipList3.append(ChebConv(n_mix, n_hidden2, K=K_mix))
                        elif conv == 'GCN':
                            self.skipList3.append(GCNConv(n_mix, n_hidden2))
                        elif conv == 'SAGE':
                            self.skipList3.append(SAGEConv(n_mix, n_hidden2)) 
                        elif conv == 'Graph':
                            self.skipList3.append(GraphConv(n_mix, n_hidden2))
                    else:
                        if conv == 'Cheb':
                            self.skipList3.append(ChebConv(n_mix, n_hidden3, K=K_mix))
                        elif conv == 'GCN':
                            self.skipList3.append(GCNConv(n_mix, n_hidden3))
                        elif conv == 'SAGE':
                            self.skipList3.append(SAGEConv(n_mix, n_hidden3)) 
                        elif conv == 'Graph':
                            self.skipList3.append(GraphConv(n_mix, n_hidden3))
        
        # Input size for final convolution
        if layers == 1:
            n_mix = n_hidden
        elif layers == 2:
            n_mix = n_hidden2
        else:
            n_mix = n_hidden3
        
        if clustering != 'None':
            n_mix = n_mix * 3
            
        if skipconv:
            n_mix = n_mix + n_input
        
        # Output size depends on heading channel
        if includeHeading:
            n_output = 9
        else:
            n_output = 6
        
        # Select convolution type
        if conv == 'Cheb':
            self.conv_mix = ChebConv(n_mix, n_output, K=K_mix)
        elif conv == 'GCN':
            self.conv_mix = GCNConv(n_mix, n_output)
        elif conv == 'SAGE':
            self.conv_mix = SAGEConv(n_mix, n_output)  
        elif conv == 'Graph':
            self.conv_mix = GraphConv(n_mix, n_output)  
            
        self.xIdx = []
        self.yIdx = []
        self.weights = []
        
        
    def forward(self, data, final, start):
        x, edge_index, pos, batch = data.x, data.edge_index, data.pos, data.batch
        
        x_start = x
        
        # Only street based pooling
        if self.clustering == 'Street':
            batchClusters1 = self.clusters1
            batchCat = self.categories
            batchClusters2 = self.clusters2
            
            batch_size = torch.max(batch) + 1
            
            # Divide clusters and categories from different batches
            for i in range(1,batch_size):
                batchClusters1 = torch.cat((batchClusters1, self.clusters1 + i*self.maxCluster1))
                batchCat = torch.cat((batchCat, self.categories + i * 5))
                batchClusters2 = torch.cat((batchClusters2, self.clusters2 + i*self.maxCluster2))
                
            batchCat = batchCat.long()
            data.batch = batchCat
            
            data2 = data
            
            # Both pooled branches, max pooling
            data = max_pool(batchClusters1, data)
            x_t, edge_index_t, pos_t, batchCat_t = data.x, data.edge_index, data.pos, data.batch
            
            data2 = max_pool(batchClusters2, data2)
            x_t2, edge_index_t2, pos_t2, batchCat_t2 = data2.x, data2.edge_index, data2.pos, data2.batch
            
            edge_index_t, temp = add_self_loops(edge_index_t)
            edge_index_t2, temp = add_self_loops(edge_index_t2)
            
            
            # Add coordinates and categories to input
            if self.coords:
                cats = (batchCat % 5).float()
                catsT = (batchCat_t % 5).float()
                catsT2 = (batchCat_t2 % 5).float()
        
                normPos = pos / torch.max(pos)
                normPos_t = pos_t / torch.max(pos_t)
                normPos_t2 = pos_t2 / torch.max(pos_t2)
                normCat = (cats / 4).view(batchCat.size(0),1)
                normCat_t = (catsT / 4).view(batchCat_t.size(0),1)
                normCat_t2 = (catsT2 / 4).view(batchCat_t2.size(0),1)
        
                x = torch.cat((x, normPos, normCat),1)
                x_t = torch.cat((x_t, normPos_t, normCat_t),1)
                x_t2 = torch.cat((x_t2, normPos_t2, normCat_t2),1)
    
            # Perform convolution blocks in all 3 branches
            for i in range(self.layers):
                x_temp = x
                x = self.moduleList1[i](x,edge_index)
                if self.midSkip:
                    x = torch.cat((x, x_temp), 1)
                    if i == 0:
                        bn = self.bn
                    elif i == 1:
                        bn = self.bn2
                    else:
                        bn = self.bn3
                        
                    x = F.relu(bn(self.skipList1[i](x,edge_index)))
            
            for i in range(self.layers):
                x_ttemp = x_t
                x_t = self.moduleList2[i](x_t,edge_index_t)
                if self.midSkip:
                    x_t = torch.cat((x_t, x_ttemp), 1)
                    if i == 0:
                        bn = self.bn
                    elif i == 1:
                        bn = self.bn2
                    else:
                        bn = self.bn3
                    x_t = F.relu(bn(self.skipList2[i](x_t,edge_index_t)))
                    
            for i in range(self.layers):
                x_ttemp2 = x_t2
                x_t2 = self.moduleList3[i](x_t2,edge_index_t2)
                if self.midSkip:
                    x_t2 = torch.cat((x_t2, x_ttemp2), 1)
                    if i == 0:
                        bn = self.bn
                    elif i == 1:
                        bn = self.bn2
                    else:
                        bn = self.bn3
                    x_t2 = F.relu(bn(self.skipList3[i](x_t2,edge_index_t2)))
            
            # Calculate knn weights of both pooled branches for first batch (and last, since the size might be different)
            if start:
                sorter = torch.argsort(batchCat)        
                backsorter = torch.argsort(sorter)
                
                pos = pos[sorter]
                batchCat = batchCat[sorter]
            
                pairs = knn(pos_t,pos,self.knn, batch_x = batchCat_t, batch_y = batchCat)
                yIdx, xIdx = pairs
                diff = pos_t[xIdx] - pos[yIdx]
                squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
                weights = 1.0 / torch.clamp(squared_distance, min = 1e-16)
                
                pairs2 = knn(pos_t2,pos,self.knn, batch_x = batchCat_t2, batch_y = batchCat)
                yIdx2, xIdx2 = pairs2
                diff2 = pos_t2[xIdx2] - pos[yIdx2]
                squared_distance2 = (diff2 * diff2).sum(dim=-1, keepdim=True)
                weights2 = 1.0 / torch.clamp(squared_distance2, min = 1e-16)
                
                
                self.weights = weights
                self.xIdx = xIdx
                self.yIdx = yIdx
                
                self.weights2 = weights2
                self.xIdx2 = xIdx2
                self.yIdx2 = yIdx2
                self.backSorter = backsorter
                
                
            if final:
                sorter = torch.argsort(batchCat)        
                backsorter = torch.argsort(sorter)
                
                pos = pos[sorter]
                batchCat = batchCat[sorter]
            
                pairs = knn(pos_t,pos,self.knn, batch_x = batchCat_t, batch_y = batchCat)
                yIdx, xIdx = pairs
                diff = pos_t[xIdx] - pos[yIdx]
                squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
                weights = 1.0 / torch.clamp(squared_distance, min = 1e-16)
                
                pairs2 = knn(pos_t2,pos,self.knn, batch_x = batchCat_t2, batch_y = batchCat)
                yIdx2, xIdx2 = pairs2
                diff2 = pos_t2[xIdx2] - pos[yIdx2]
                squared_distance2 = (diff2 * diff2).sum(dim=-1, keepdim=True)
                weights2 = 1.0 / torch.clamp(squared_distance2, min = 1e-16)
                
                
                self.weights = weights
                self.xIdx = xIdx
                self.yIdx = yIdx
                
                self.weights2 = weights2
                self.xIdx2 = xIdx2
                self.yIdx2 = yIdx2
                self.backSorter = backsorter
                
            
            # Unpool pooled branches
            x_t = scatter_add(x_t[self.xIdx] * self.weights, self.yIdx, dim = 0, dim_size=pos.size(0))
            x_t = x_t / scatter_add(self.weights, self.yIdx, dim = 0, dim_size=pos.size(0))
            
            x_t = x_t[self.backSorter]
            
            x_t2 = scatter_add(x_t2[self.xIdx2] * self.weights2, self.yIdx2, dim = 0, dim_size=pos.size(0))
            x_t2 = x_t2 / scatter_add(self.weights2, self.yIdx2, dim = 0, dim_size=pos.size(0))
            
            x_t2 = x_t2[self.backSorter]
            
            # Input size of final convolution
            if self.skipconv:
                y = torch.cat((x, x_t, x_t2, x_start),1)
            else:
                y = torch.cat((x, x_t, x_t2),1)
                
            # Do final convolution    
            y = self.conv_mix(y, edge_index)
            
            # Add dropout layer
            if self.p != 1:
                y = F.dropout(y, training=self.training, p=self.p)
            
        return y