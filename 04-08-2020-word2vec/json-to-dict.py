# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 01:09:36 2020

@author: strik
"""

import json
import numpy as np

import pandas as pd
# data = None
# with open(r'C:\Users\strik\Desktop\comm_use_subset\pmc_json\finalfile.xml.json') as file:
#     data = json.load(file)
    


# df = pd.read_csv(r'C:\Users\strik\Desktop\comm_use_subset\CORD-19-research-challenge\metadata.csv')
df2 = pd.read_csv(r'C:\Users\strik\Desktop\comm_use_subset\CORD-19-research-challenge\metadata.csv', usecols=['abstract'])
df2.dropna(inplace=True)

# df2.to_csv('all-text.csv', header=False, index=False)
df2.to_csv('all-text.txt', header=False, index=False)


# data = df2.to_numpy()
