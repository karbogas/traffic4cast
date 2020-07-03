# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:59:42 2020

@author: karbogas
"""

import geopandas as gpd
from shapely.geometry import Polygon
import osmnx as ox
import numpy as np
import pickle
import os

def getMaskFromOSM(city = 'Berlin', storage = ''):
    """
        Creates a mask for the Preprocessing from OSM data
    
        input:
            city: city for which the mask should be created
            storage: mask storage folder
            
        output:
            mask: created OSM mask
    """
    
    # Coordinates of the bounding boxes
    if city == 'Berlin':
        yMin = 52.359
        yMax = 52.854
        xMin = 13.189
        xMax = 13.625
    elif city == 'Moscow':
        yMin = 55.506
        yMax = 55.942
        xMin = 37.357
        xMax = 37.852
    elif city == 'Istanbul':
        yMin = 40.810
        yMax = 41.305
        xMin = 28.794
        xMax = 29.230
        
    # Load street data
    G = ox.graph_from_bbox(yMax,yMin,xMax,xMin, network_type = 'drive', truncate_by_edge=True)
    
    # Convert street data to GeoDataFrame
    nodes, edges = ox.graph_to_gdfs(G)
    
    # Size of the grid cells
    length = 0.001
    
    # Bounding boxes of the grid cells
    cols = np.arange(xMin,xMax,length)
    rows = np.arange(yMin,yMax,length)
    
    polygons = []
    xList = []
    yList = []
    
    
    # Build grid cells representing the pixels of the traffic4cast images
    for x in cols:
        for y in rows:
            polygons.append( Polygon([(x,y),(x+length,y),(x+length,y+length),(x,y+length)]))
            xList.append(int(np.round(1000*(x - xMin))))
            yList.append(int(np.round(1000*(y - yMin))))
            
    grid = gpd.GeoDataFrame({'geometry': polygons, 'x':xList, 'y': yList})
    
    edges.crs = grid.crs
    
    # intersect road data with grid
    joined = gpd.sjoin(edges,grid, op = 'intersects')
    
    mask = np.zeros((495,436))
    
    # Build a mask from intersections (+ rotate data to fit desired output)
    if city == 'Moscow':
        for idx, row in joined.iterrows():
            mask[row.x,row.y] = 1
        mask = np.flip(mask)
    else:
        for idx, row in joined.iterrows():
            mask[row.y,row.x] = 1
        mask = np.flip(mask,0)
        
    mask = (mask > 0)
    
    #Save mask
    path = os.path.join(storage, city + '.mask')
    pickle.dump(mask, open(path,'wb'))
    
    return mask

def getStreetClusters(city='Berlin', storage = '', k = 4):
    
    """
        Creates cluster files for the selected city to be used for street based pooling
    
        input:
            city: city for which the cluster files should be created
            storage: mask storage folder
            k: kernel size of the cluster
    """
    
    #Load mask of the city (needed in case the OSM data changed since it was created)
    mainMask = pickle.load(open(os.path.join(storage, city + '.mask'),'rb'))
    
    # Coordinates of the bounding boxes
    if city == 'Berlin':
        yMin = 52.359
        yMax = 52.854
        xMin = 13.189
        xMax = 13.625
    elif city == 'Moscow':
        yMin = 55.506
        yMax = 55.942
        xMin = 37.357
        xMax = 37.852
    elif city == 'Istanbul':
        yMin = 40.810
        yMax = 41.305
        xMin = 28.794
        xMax = 29.230
    
    # Load street data
    G = ox.graph_from_bbox(yMax,yMin,xMax,xMin, network_type = 'drive', truncate_by_edge=True)
    
    print('Graph loaded for ' + city)
    
    # Convert street data to GeoDataFrame
    nodes, edges = ox.graph_to_gdfs(G)    
    
    # Size of the grid cells
    length = 0.001

    # Bounding boxes of the grid cells
    cols = np.arange(xMin,xMax,length)
    rows = np.arange(yMin,yMax,length)
    
    polygons = []
    xList = []
    yList = []
    
    
    # Build grid cells representing the pixels of the traffic4cast images
    for x in cols:
        for y in rows:
            polygons.append( Polygon([(x,y),(x+length,y),(x+length,y+length),(x,y+length)]))
            xList.append(int(np.round(1000*(x - xMin))))
            yList.append(int(np.round(1000*(y - yMin))))
            
          
    grid = gpd.GeoDataFrame({'geometry': polygons, 'x':xList, 'y': yList})
    
    edges.crs = grid.crs
    
    # intersect road data with grid
    joined = gpd.sjoin(edges,grid, op = 'intersects')
    
    print('Intersections built for ' + city)  
    
    
    mask = np.zeros((495,436))
    
    # Create different masks for different road categories
    if city == 'Moscow':
        
        split1 = joined[joined.highway == 'motorway']
        split2 = joined[joined.highway == 'trunk']
        split3 = joined[joined.highway == 'trunk_link']
        
        mask1 = mask.copy()
        for idx, row in split1.iterrows():
            mask1[row.x,row.y] = 1
        for idx, row in split2.iterrows():
            mask1[row.x,row.y] = 1
        for idx, row in split3.iterrows():
            mask1[row.x,row.y] = 1
        mask1 = np.flip(mask1)
        mask1[mainMask == 0] = 0
        
        split4 = joined[joined.highway == 'primary']
        split5 = joined[joined.highway == 'primary-link']
        
        mask2 = mask.copy()
        
        for idx, row in split4.iterrows():
            mask2[row.x,row.y] = 1
        for idx, row in split5.iterrows():
            mask2[row.x,row.y] = 1
        mask2 = np.flip(mask2)
        
        mask2 = mask2 - mask1
        mask2[mask2 < 0] = 0
        mask2[mainMask == 0] = 0
        
        split6 = joined[joined.highway == 'secondary']
        split7 = joined[joined.highway == 'secondary-link']
        
        mask3 = mask.copy()
        
        for idx, row in split6.iterrows():
            mask3[row.x,row.y] = 1
        for idx, row in split7.iterrows():
            mask3[row.x,row.y] = 1
        mask3 = np.flip(mask3)
        
        mask3 = mask3 - mask2 - mask1
        mask3[mask3 < 0] = 0
        mask3[mainMask == 0] = 0
        
        split8 = joined[joined.highway == 'tertiary']
        split9 = joined[joined.highway == 'tertiary-link']
        
        mask4 = mask.copy()
        
        for idx, row in split8.iterrows():
            mask4[row.x,row.y] = 1
        for idx, row in split9.iterrows():
            mask4[row.x,row.y] = 1
        mask4 = np.flip(mask4)
        
        mask4 = mask4 - mask3 - mask2 - mask1
        mask4[mask4 < 0] = 0
        mask4[mainMask == 0] = 0
        
        mask5 = mainMask - mask4 - mask3 - mask2 - mask1
        mask5[mask5 < 0] = 0
        
    else:
        split1 = joined[joined.highway == 'motorway']
        split2 = joined[joined.highway == 'motorway_link']
        
        mask = np.zeros((495,436))
        mask1 = mask.copy()
        for idx, row in split1.iterrows():
            mask1[row.y,row.x] = 1
        for idx, row in split2.iterrows():
            mask1[row.y,row.x] = 1
        mask1 = np.flip(mask1,0)
        mask1[mainMask == 0] = 0
        
        split3 = joined[joined.highway == 'primary']
        split4 = joined[joined.highway == 'primary-link']
        
        mask2 = mask.copy()
        
        for idx, row in split3.iterrows():
            mask2[row.y,row.x] = 1
        for idx, row in split4.iterrows():
            mask2[row.y,row.x] = 1
        mask2 = np.flip(mask2,0)
        
        mask2 = mask2 - mask1
        mask2[mask2 < 0] = 0
        mask2[mainMask == 0] = 0
        
        split5 = joined[joined.highway == 'secondary']
        split6 = joined[joined.highway == 'secondary-link']
        
        mask3 = mask.copy()
        
        for idx, row in split5.iterrows():
            mask3[row.y,row.x] = 1
        for idx, row in split6.iterrows():
            mask3[row.y,row.x] = 1
        mask3 = np.flip(mask3,0)
        
        mask3 = mask3 - mask2 - mask1
        mask3[mask3 < 0] = 0
        mask3[mainMask == 0] = 0
        
        split7 = joined[joined.highway == 'tertiary']
        split8 = joined[joined.highway == 'tertiary-link']
        
        mask4 = mask.copy()
        
        for idx, row in split7.iterrows():
            mask4[row.y,row.x] = 1
        for idx, row in split8.iterrows():
            mask4[row.y,row.x] = 1
        mask4 = np.flip(mask4,0)
        
        mask4 = mask4 - mask3 - mask2 - mask1
        mask4[mask4 < 0] = 0
        mask4[mainMask == 0] = 0
        
        mask5 = mainMask - mask4 - mask3 - mask2 - mask1
        mask5[mask5 < 0] = 0
        
    maskMix = mask.copy()

    # Put different categories together
    maskMix[mask1.astype(int) == 1] = 1
    maskMix[mask2.astype(int) == 1] = 2
    maskMix[mask3.astype(int) == 1] = 3
    maskMix[mask4.astype(int) == 1] = 4
    maskMix[mask5.astype(int) == 1] = 5
    
    clusterNr = 0
    clusters = {}
    
    # Cluster points in the same category and kernel together
    for i in range (1, 6):
        maskpos = np.where(maskMix == i)
        arr = np.asarray(maskpos)
        maskCluster = arr // k
        clusters[i] = clusterNr + maskCluster[0]*int(np.ceil(435/k)) + maskCluster[1]
        clusterNr = max(clusters[i])
    
    maskpos = np.where(maskMix == 1)
    
    maskCluster = mask.copy()
    for i in range (1, 6):
        maskpos = np.where(maskMix == i)
        maskCluster[maskpos] = clusters[i]
        
    # Calculate cluster centers   
    j = 0
    clusterCenters = np.zeros((len(np.unique(maskCluster)) - 1,2))
    for i in np.unique(maskCluster):
        if i != 0:
            pos = np.where(maskCluster == i)
            x = pos[0]
            y = pos[1]
            meanX = np.mean(x)
            meanY = np.mean(y)
            clusterCenters[j][0] = meanX
            clusterCenters[j][1] = meanY
            
            j += 1 
    
    # Get cluster and category information
    index = np.where(maskMix>0)
    cats = maskMix[index].astype(int)
    clusters = maskCluster[index].astype(int)
    
    # Save cluster, category and cluster center files
    pickle.dump(clusterCenters, open(os.path.join(storage, city + str(k) + '.centers'),'wb'))
    pickle.dump(cats, open(os.path.join(storage, city + str(k)  + '.cats'),'wb'))
    pickle.dump(clusters, open(os.path.join(storage, city + str(k)  + '.clusters'),'wb'))
    print('Process done for ' + city) 
    
if __name__ == '__main__':
    #getMaskFromOSM(city = 'Moscow', storage = r'D:\Masterarbeit\OSM')
    getMaskFromOSM(city = 'Istanbul', storage = r'D:\Masterarbeit\OSM')
    
    
    
    