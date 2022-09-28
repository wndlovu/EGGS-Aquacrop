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

## add ET data from online models
disalexi = pd.read_csv(wd + '/data/hydrometeorology/openET/ET_monthly_disalexi_FieldsAroundSD6KS_20220708.csv')
eemetric = pd.read_csv(wd + "/data/hydrometeorology/openET/ET_monthly_eemetric_FieldsAroundSD6KS_20220708.csv")
enseble = pd.read_csv(wd + "/data/hydrometeorology/openET/ET_monthly_ensemble_FieldsAroundSD6KS_20220708.csv")
geesebal = pd.read_csv(wd + "/data/hydrometeorology/openET/ET_monthly_geesebal_FieldsAroundSD6KS_20220708.csv")
ptjpl = pd.read_csv(wd + "/data/hydrometeorology/openET/ET_monthly_ptjpl_FieldsAroundSD6KS_20220708.csv")
sims = pd.read_csv(wd + "/data/hydrometeorology/openET/ET_monthly_sims_FieldsAroundSD6KS_20220708.csv")
ssebop = pd.read_csv(wd + "/data/hydrometeorology/openET/ET_monthly_ssebop_FieldsAroundSD6KS_20220708.csv")

# filter for site 1381151
disalexi = disalexi[disalexi['UID'] == 1381151]
eemetric = eemetric[eemetric['UID'] == 1381151]
enseble = enseble[enseble['UID'] == 1381151]
geesebal = geesebal[geesebal['UID'] == 1381151]
ptjpl = ptjpl[ptjpl['UID'] == 1381151]
sims = sims[sims['UID'] == 1381151]
ssebop = ssebop[ssebop['UID'] == 1381151]

# set time as index to allow for joining dfs later
disalexi = disalexi.set_index('time')
eemetric = eemetric.set_index('time')
enseble = enseble.set_index('time')
geesebal = geesebal.set_index('time')
ptjpl = ptjpl.set_index('time')
sims = sims.set_index('time')
ssebop = ssebop.set_index('time')

# add method identifier to every column
disalexi.columns = [str(col) + '_disalexi' for col in disalexi.columns]
enseble.columns = [str(col) + '_ensemble' for col in enseble.columns]
eemetric.columns = [str(col) + '_eemetric' for col in eemetric.columns]
geesebal.columns = [str(col) + '_geesebal' for col in geesebal.columns]
ptjpl.columns = [str(col) + '_ptjpl' for col in ptjpl.columns]
ssebop.columns = [str(col) + '_ssebop' for col in ssebop.columns]
sims.columns = [str(col) + '_sims' for col in sims.columns]


fullET = pd.concat([disalexi, enseble, eemetric, geesebal, ptjpl, sims, ssebop], axis=1)
fullET.reset_index(inplace=True) # make time a column

# 

