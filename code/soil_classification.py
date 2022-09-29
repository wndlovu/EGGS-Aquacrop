#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 10:21:14 2022

@author: wayne
"""
#https://soilwater.github.io/pynotes-agriscience/notebooks/soil_textural_class.html
def soiltexturalclass(sand,clay):
    """Function that returns the USDA 
    soil textural class given 
    the percent sand and clay.
    
    Inputs = Percetnage of sand and clay
    """
    
    silt = 100 - sand - clay
    
    if sand + clay > 100 or sand < 0 or clay < 0:
        raise Exception('Inputs adds over 100% or are negative')

    elif silt + 1.5*clay < 15:
        textural_class = 'Sand'

    elif silt + 1.5*clay >= 15 and silt + 2*clay < 30:
        textural_class = 'Loamy Sand'

    elif (clay >= 7 and clay < 20 and sand > 52 and silt + 2*clay >= 30) or (clay < 7 and silt < 50 and silt + 2*clay >= 30):
        textural_class = 'Sandy Loam'

    elif clay >= 7 and clay < 27 and silt >= 28 and silt < 50 and sand <= 52:
        textural_class = 'Loam'

    elif (silt >= 50 and clay >= 12 and clay < 27) or (silt >= 50 and silt < 80 and clay < 12):
        textural_class = 'Silt Loam'

    elif silt >= 80 and clay < 12:
        textural_class = 'Silt'

    elif clay >= 20 and clay < 35 and silt < 28 and sand > 45:
        textural_class = 'Sandy Clay Loam'

    elif clay >= 27 and clay < 40 and sand > 20 and sand <= 45:
        textural_class = 'Clay Loam'

    elif clay >= 27 and clay < 40 and sand <= 20:
        textural_class = 'Silty Clay Loam'

    elif clay >= 35 and sand > 45:
        textural_class = 'Sandy Clay'

    elif clay >= 40 and silt >= 40:
        textural_class = 'Silty Clay'

    elif clay >= 40 and sand <= 45 and silt < 40:
        textural_class = 'Clay'

    else:
        textural_class = 'na'

    return textural_class


soils_df_full = pd.read_csv(wd + '/data/agricLand/soils/Soil_FieldsAroundSD6KS_POLARIS_AGrinstead_20220706.csv')
soils_df = soils_df_full[soils_df_full['UID'] == 1381151]
soils_df = soils_df[soils_df['depth_cm'] == '0-5']

# soil texture for one site
soils_df['soil_class'] = soils_df.apply(lambda x: soiltexturalclass(x['sand_prc'], x['clay_prc']), 
                        axis=1)


# soil texture for all sites
soil_txt =soils_df_full[soils_df_full['depth_cm'] == '0-5']
soil_txt['soil_class'] = soil_txt.apply(lambda x: soiltexturalclass(x['sand_prc'], x['clay_prc']), 
                        axis=1)

# create a function that takes all the sites and uses soil type defined by soiltextureclass()




!!pip install aquacrop==2.2
!pip install numba==0.55
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
from aquacrop.utils import prepare_weather, get_filepath
from os import chdir, getcwd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime




wd=getcwd() # set working directory
chdir(wd)
path = get_filepath(wd + '/data/hydrometeorology/gridMET/gridMET_1381151.txt') #replace folder name from folder name with file path
wdf = prepare_weather(path)
sim_start = '2016/01/01' #dates to match crop data
sim_end = '2021/12/01'
soil= soils_df['soil_class']
crop = Crop('Maize',planting_date='05/01')
initWC = InitialWaterContent(value=['FC'])


# get date variable from the wdf
wdf_date = wdf[["Date"]]
wdf_date = wdf_date[wdf_date['Date'] > '2015/12/31']
wdf_date = wdf_date.reset_index() # reset index to start from 0
wdf_date = wdf_date[['Date']] # select date variable and drop second index column

# run aquacrop water flux model
model = AquaCropModel(sim_start,sim_end,wdf,soil,crop,initWC)
model.run_model(till_termination=True)
#model_results = model2.get_simulation_results().head()
model_results = model._outputs.water_flux
