# -*- coding: utf-8 -*-
"""
@author: karbogas

Config file containing the settings for main.py
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
    config['log_target'] = '/home/arbogast/runs'
    config['num_workers'] = 8
    config['batch_size'] = 4
else:
    config['source_root'] = r'D:\Masterarbeit\traffic4cast-master\data_raw'
    config['target_root'] = r'D:\Masterarbeit\traffic4cast-master\data'
    config['masks_target'] = r'D:\Masterarbeit\traffic4cast-master\masks'
    config['graph_target'] = r'D:\Masterarbeit\traffic4cast-master\graphs'
    config['log_target'] = r'D:\Masterarbeit\traffic4cast-master\runs' 
    config['batch_size'] = 4
    
    # Spyder can't work with num_workers > 0
    if in_spyder:
        config['num_workers'] = 0
    else:
        config['num_workers'] = 4 
    
# city the model should train on
config['city'] = 'Berlin'

# Only use the test times for training (small subset for testing)?
config['filter_test_times'] = True

# Shuffle the dataset (to prevent overfitting)?
config['shuffle'] = True

# Include the heading channel for the training?
config['includeHeading'] = False

# Which GPU should be used?
config['device'] = 0

# Learning rate and number of epochs for training
config['learning_rate'] = 0.01
config['epochs'] = 1
