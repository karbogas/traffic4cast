# -*- coding: utf-8 -*-
"""
@author: karbogas

Config file containing the settings for MSEWholeImages.py
"""


config = {}

# Location of the trained model
config['run_folder'] =  r'C:\Users\konst\Desktop\runs\Berlin_ChebConv_DoubleCluster_3Layers'

# Where the data is stored
config['image_folder'] = r'D:\Masterarbeit\traffic4cast-master\data'
config['graph_folder'] = r'D:\Masterarbeit\traffic4cast-master\graphs'
config['mask_folder'] = r'D:\Masterarbeit\traffic4cast-master\masks'

# Apply model to which cities?
config['cities'] = ['Berlin', 'Moscow', 'Istanbul']

# Only use test times? Was heading included for training?
config['filter_test_times'] = False
config['includeHeading'] = False