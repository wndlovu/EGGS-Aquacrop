#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 05:45:53 2022

@author: wayne
"""

!pip install aquacrop==0.2    
#from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
#from aquacrop.utils import prepare_weather, get_filepath
from aquacrop.classes import    *
from aquacrop.core import       *
from os import chdir, getcwd
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import os
import glob
_=[sys.path.append(i) for i in ['.', '..']]


wd=getcwd() # set working directory
chdir(wd)
soils_df_full = pd.read_csv(wd + '/data/agricLand/soils/Soil_FieldsAroundSD6KS_POLARIS_AGrinstead_20220706.csv')
soils_df = soils_df_full[soils_df_full['UID'] == 1381151] # filter for one site
soils_df = soils_df[soils_df['depth_cm'] == '0-5']


soils = pd.DataFrame(soils_df_full)
soils = soils[soils['depth_cm'] == '0-5'] # use upper 0.5cm
soils = soils.head(1)


def soil_cl(x):
        #result = []
        ts = x["thetaS_m3m3"]
        ks= x["logKsat_cmHr"]
        tp = x["thetaR_m3m3"]
        custom = SoilClass('custom')
        custom.add_layer(thickness=0.1,thS=ts, # assuming soil properties are the same in the upper 0.1m
                     Ksat=ks,thWP =tp , 
                     thFC = .4, penetrability = 100)
        return(custom)
    #custom_soil.append(custom)
 
result = []    
def soil_cl(x):
        ts = x["thetaS_m3m3"]
        ks= x["logKsat_cmHr"]
        tp = x["thetaR_m3m3"]
        custom = SoilClass('custom')
        custom.add_layer(thickness=0.1,thS=ts, # assuming soil properties are the same in the upper 0.1m
                     Ksat=ks,thWP =tp , 
                     thFC = .4, penetrability = 100)
        #return(custom)
        result.append(custom)
        return(result)
  
#177799,  177806   
 
custom_soil2 = []
for i in range(0, len(soils)):
     y = soils.apply(soil_cl, axis = 1)
     custom_soil2.append(soils)
     
     
     
     
     
     
     
     
     
     
 