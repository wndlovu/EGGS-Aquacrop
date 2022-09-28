# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 22:17:13 2022

@author: w610n091
"""

!pip install aquacrop==2.2
!pip install numba==0.55
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
from aquacrop.utils import prepare_weather, get_filepath
from os import chdir, getcwd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



## Aquacrop Model
wd=getcwd() # set working directory
chdir(wd)
path = get_filepath(wd + '/data/hydrometeorology/gridMET/gridMET_1381151.txt') #replace folder name from folder name with file path
wdf = prepare_weather(path)
sim_start = '2000/01/01' #dates to match crop data
sim_end = '2015/12/31'
soil= Soil('Loam')
crop = Crop('Maize',planting_date='05/01')
initWC = InitialWaterContent(value=['FC'])


# get date variable from the wdf
wdf_date = wdf[["Date"]]

# run aquacrop water flux model
model = AquaCropModel(sim_start,sim_end,wdf,soil,crop,initWC)
model.run_model(till_termination=True)
#model_results = model2.get_simulation_results().head()
model_results = model._outputs.water_flux

# add the date variable and jon by index
model_results = model_results.join(wdf_date)

## add ET data from 








