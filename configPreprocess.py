# -*- coding: utf-8 -*-
"""
@author: karbogas

Config file containing the settings for PreProcessing.py
"""


# Working on the server or locally?
on_server = False

# Working with Spyder?
in_spyder = True

config = dict()

# Different directories if local or server.
if on_server:
    config['source_root'] = '/data/traffic4cast/data_raw'
    config['target_root'] = '/data/traffic4cast/data'
    config['masks_target'] = '/data/karbogas/masks'
    config['graph_target'] = '/data/karbogas/graphs'
    config['num_workers'] = 4
    config['batch_size'] = 1
else:
    config['source_root'] = r'D:\Masterarbeit\traffic4cast-master\data_raw'
    config['target_root'] = r'D:\Masterarbeit\traffic4cast-master\data'
    config['masks_target'] = r'D:\Masterarbeit\traffic4cast-master\masks'
    config['graph_target'] = r'D:\Masterarbeit\traffic4cast-master\graphs'
    config['batch_size'] = 1
    
    # Spyder can't work with num_workers > 0
    if in_spyder:
        config['num_workers'] = 0
    else:
        config['num_workers'] = 8
   

# Preprocessing for these cities
config['cities'] = ['Berlin']

# Only preprocess with test times (small subset for testing)?
config['filter_test_times'] = False

# Shuffle the dataset (No real effect here)?
config['shuffle'] = False

# Threshold to use when threshold mask is built
config['threshold'] = 200000

# Creation of cluster, categories and cluster center files for the following kernel sizes
config['k_list'] = [3,6]

# Type of mask to be built: 'OSM' for OSM mask or 'Sum' for threshold mask
config['mask_type'] = 'OSM'

# Overwrite available masks or graph data? 
config['delete_masks'] = False
config['delete_graphs'] = False