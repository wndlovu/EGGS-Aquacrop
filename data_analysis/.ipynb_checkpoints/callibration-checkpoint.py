#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 07:12:33 2022

@author: wayne
"""

!pip install aquacrop==2.2
!pip install numba==0.55
!pip install statsmodels==0.13.2
!pip install scikit-learn
!pip install pyemu
!pip install flopy
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
from aquacrop.utils import prepare_weather, get_filepath
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
#from aquacrop.entities import IrrigationManagement
from os import chdir, getcwd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import csv



import os 
import datetime
import shapefile as shp
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle 
from sklearn.metrics import r2_score
import pyemu 
from pyemu import *
import flopy


wd=getcwd() # set working directory
chdir(wd)


with open(wd+'/data/hydrometeorology/gridMET/ks_gridMET.pickle', 'rb') as met: 
    gridMET_county = pickle.load(met)
   
    
with open(wd+'/data/groupings/ks_ccm.pickle', 'rb') as info: 
    grouped_info = pickle.load(info)   
    
# soils
with open(wd+'/data/agricLand/soils/ks_soil.pickle', 'rb') as sl: 
    soil_data = pickle.load(sl) 

# sheridan gridMET
sher_gridMET = gridMET_county[10]
sher_gridMET = sher_gridMET.assign(year = sher_gridMET['Date'].dt.year) # create year variable
#sher_gridMET = sher_gridMET[sher_gridMET['year'].some_date.between(2000, 2015)]
sher_gridMET = sher_gridMET[(sher_gridMET['year'] >= 2000) & (sher_gridMET['year'] <= 2015)]
#sher_gridMET = sher_gridMET[sher_gridMET['year'] == 2012] # filter for 2012
sher_gridMET = sher_gridMET.drop(['year'], axis=1) # drop year variable


# sheridan soils
soil_df = soil_data[10]


for i, row in soil_df.iterrows():   #soil_df.itertuples():
     ids = soil_df['UID'][i] #create soil_df with UID from the soils file used - fix this
     #id_list.append(ids)
     if row['depth_cm'] == "0-5":
         pred_thWP_5 = ((-0.024*((soil_df['sand_prc'][i])/100))) + ((0.487*((soil_df['clay_prc'][i])/100))) + ((0.006*((soil_df['om'][i])/100))) + ((0.005*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.013*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.068*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.031
         wp_5 = pred_thWP_5 + (0.14 * pred_thWP_5) - 0.02
         pred_thFC_5 = ((-0.251*((soil_df['sand_prc'][i])/100))) + ((0.195*((soil_df['clay_prc'][i])/100)))+ ((0.011*((soil_df['om'][i])/100))) + ((0.006*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.027*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.452*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.299
         fc_5 = pred_thFC_5 + (1.283 * (np.power(pred_thFC_5, 2))) - (0.374 * pred_thFC_5) - 0.015
         #fc = pred_thFC + (1.283 * (pred_thFC*pred_thFC)) - (0.374 * pred_thFC) - 0.015
         ts_5 =soil_df["thetaS_m3m3"][i]
         ks_5=(soil_df['Ksat_cmHr'][i])*240
     if row['depth_cm'] == "5-15":
         pred_thWP_15 = ((-0.024*((soil_df['sand_prc'][i])/100))) + ((0.487*((soil_df['clay_prc'][i])/100))) + ((0.006*((soil_df['om'][i])/100))) + ((0.005*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.013*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.068*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.031
         wp_15 = pred_thWP_15 + (0.14 * pred_thWP_15) - 0.02
         pred_thFC_15 = ((-0.251*((soil_df['sand_prc'][i])/100))) + ((0.195*((soil_df['clay_prc'][i])/100)))+ ((0.011*((soil_df['om'][i])/100))) + ((0.006*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.027*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.452*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.299
         fc_15 = pred_thFC_15 + (1.283 * (np.power(pred_thFC_15, 2))) - (0.374 * pred_thFC_15) - 0.015
         #fc = pred_thFC + (1.283 * (pred_thFC*pred_thFC)) - (0.374 * pred_thFC) - 0.015
         ts_15 =soil_df["thetaS_m3m3"][i]
         ks_15=(soil_df['Ksat_cmHr'][i])*240
     if row['depth_cm'] == "15-30":
         pred_thWP_30 = ((-0.024*((soil_df['sand_prc'][i])/100))) + ((0.487*((soil_df['clay_prc'][i])/100))) + ((0.006*((soil_df['om'][i])/100))) + ((0.005*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.013*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.068*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.031
         wp_30 = pred_thWP_30 + (0.14 * pred_thWP_30) - 0.02
         pred_thFC_30 = ((-0.251*((soil_df['sand_prc'][i])/100))) + ((0.195*((soil_df['clay_prc'][i])/100)))+ ((0.011*((soil_df['om'][i])/100))) + ((0.006*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.027*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.452*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.299
         fc_30 = pred_thFC_30 + (1.283 * (np.power(pred_thFC_30, 2))) - (0.374 * pred_thFC_30) - 0.015
         #fc = pred_thFC + (1.283 * (pred_thFC*pred_thFC)) - (0.374 * pred_thFC) - 0.015
         ts_30 =soil_df["thetaS_m3m3"][i]
         ks_30=(soil_df['Ksat_cmHr'][i])*240
     if row['depth_cm'] == "30-60":
         pred_thWP_60 = ((-0.024*((soil_df['sand_prc'][i])/100))) + ((0.487*((soil_df['clay_prc'][i])/100))) + ((0.006*((soil_df['om'][i])/100))) + ((0.005*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.013*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.068*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.031
         wp_60 = pred_thWP_60 + (0.14 * pred_thWP_60) - 0.02
         pred_thFC_60 = ((-0.251*((soil_df['sand_prc'][i])/100))) + ((0.195*((soil_df['clay_prc'][i])/100)))+ ((0.011*((soil_df['om'][i])/100))) + ((0.006*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.027*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.452*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.299
         fc_60 = pred_thFC_60 + (1.283 * (np.power(pred_thFC_60, 2))) - (0.374 * pred_thFC_60) - 0.015
         #fc = pred_thFC + (1.283 * (pred_thFC*pred_thFC)) - (0.374 * pred_thFC) - 0.015
         ts_60 =soil_df["thetaS_m3m3"][i]
         ks_60=(soil_df['Ksat_cmHr'][i])*240
     if row['depth_cm'] == "60-100":
         pred_thWP_100 = ((-0.024*((soil_df['sand_prc'][i])/100))) + ((0.487*((soil_df['clay_prc'][i])/100))) + ((0.006*((soil_df['om'][i])/100))) + ((0.005*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.013*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.068*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.031
         wp_100 = pred_thWP_100 + (0.14 * pred_thWP_100) - 0.02
         pred_thFC_100 = ((-0.251*((soil_df['sand_prc'][i])/100))) + ((0.195*((soil_df['clay_prc'][i])/100)))+ ((0.011*((soil_df['om'][i])/100))) + ((0.006*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.027*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.452*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.299
         fc_100 = pred_thFC_100 + (1.283 * (np.power(pred_thFC_100, 2))) - (0.374 * pred_thFC_100) - 0.015
         #fc = pred_thFC + (1.283 * (pred_thFC*pred_thFC)) - (0.374 * pred_thFC) - 0.015
         ts_100 =soil_df["thetaS_m3m3"][i]
         ks_100=(soil_df['Ksat_cmHr'][i])*240
     if row['depth_cm'] == "100-200":
         pred_thWP_200 = ((-0.024*((soil_df['sand_prc'][i])/100))) + ((0.487*((soil_df['clay_prc'][i])/100))) + ((0.006*((soil_df['om'][i])/100))) + ((0.005*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.013*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.068*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.031
         wp_200 = pred_thWP_200 + (0.14 * pred_thWP_200) - 0.02
         pred_thFC_200 = ((-0.251*((soil_df['sand_prc'][i])/100))) + ((0.195*((soil_df['clay_prc'][i])/100)))+ ((0.011*((soil_df['om'][i])/100))) + ((0.006*((soil_df['sand_prc'][i])/100))*((soil_df['om'][i])/100))- ((0.027*((soil_df['clay_prc'][i])/100))*((soil_df['om'][i])/100))+ ((0.452*((soil_df['sand_prc'][i])/100))*((soil_df['clay_prc'][i])/100))+ 0.299
         fc_200 = pred_thFC_200 + (1.283 * (np.power(pred_thFC_200, 2))) - (0.374 * pred_thFC_200) - 0.015
         #fc = pred_thFC + (1.283 * (pred_thFC*pred_thFC)) - (0.374 * pred_thFC) - 0.015
         ts_200 =soil_df["thetaS_m3m3"][i]
         ks_200=(soil_df['Ksat_cmHr'][i])*240
     

         # create soil compartments
         custom = Soil('custom',cn=46,rew=7, dz=[0.025]*2+[0.05]*2+[0.075]*2+[0.15]*2+[0.2]*2+[0.5]*2)

         custom.add_layer(thickness=0.05,thS=ts_5, # assuming soil properties are the same in the upper 0.1m
              Ksat=ks_5,thWP =wp_5 , 
              thFC = fc_5, penetrability = 100.0)
         custom.add_layer(thickness=0.15,thS=ts_15, # assuming soil properties are the same in the upper 0.1m
                      Ksat=ks_15,thWP =wp_15 , 
                      thFC = fc_15, penetrability = 100.0)
         custom.add_layer(thickness=0.3,thS=ts_30, # assuming soil properties are the same in the upper 0.1m
                      Ksat=ks_30,thWP =wp_30 , 
                      thFC = fc_30, penetrability = 100.0)
         custom.add_layer(thickness=0.3,thS=ts_60, # assuming soil properties are the same in the upper 0.1m
                      Ksat=ks_60,thWP =wp_60 , 
                      thFC = fc_60, penetrability = 100.0)
         custom.add_layer(thickness=0.4,thS=ts_100, # assuming soil properties are the same in the upper 0.1m
                      Ksat=ks_100,thWP =wp_100 , 
                      thFC = fc_100, penetrability = 100.0)
         custom.add_layer(thickness=1,thS=ts_200, # assuming soil properties are the same in the upper 0.1m
                      Ksat=ks_200,thWP =wp_200 , 
                      thFC = fc_200, penetrability = 100.0)




# run model 

wdf = sher_gridMET
sim_start = '2000/01/01' #dates to match crop data
sim_end = '2015/12/31'
custom_soil = custom # use custom layer for 1 site
crop = Crop('Maize', planting_date='05/01') 
initWC = InitialWaterContent(value=['FC'])
irr_mngt = IrrigationManagement(irrigation_method=1,SMT=[80]*4)
#irr_mngt = IrrigationManagement(irrigation_method = 0) # no irrigation



# run model
model = AquaCropModel(sim_start,sim_end,wdf,custom_soil,crop,initWC, irr_mngt)
model.run_model(till_termination=True) # run model till the end
model_df_et = model._outputs.water_flux
model_df_irr = model._outputs.final_stats
#model_df_water_storage = model._outputs.water_storage
#model_df_crp_grwth = model._outputs.crop_growth


# yield results
yield_obs = model_df_irr[['Season', 'Yield (tonne/ha)']]
yield_obs = yield_obs.rename(columns={
                   'yield': 'Yield (tonne/ha)'})

# export files
sher_soils.to_csv(r'./data/calibration_files/sheridan_corn/sher_soils.csv', sep=',', encoding='utf-8', header='true')
sher_gridMET.to_csv(r'./data/calibration_files/sheridan_corn/sher_gridMET.csv', sep=',', encoding='utf-8', header='true')
yield_obs.to_csv(r'./data/calibration_files/sheridan_corn/yield_obs.txt', sep=' ', index=False, header='true')


# path to pest input files
inputs_path = wd+'/data/calibration_files/sheridan_corn'
pest_path = wd+'/data/calibration_files/pest_sheridan_corn'
params = wd+'/data/calibration_files/sheridan_corn/sher_pest_params.txt'
yield_df = wd+'/data/calibration_files/sheridan_corn/yield_obs.txt'

pf = pyemu.utils.pst_from.PstFrom(inputs_path,pest_path, remove_existing=True)

pf.add_parameters(yield_df)





pf.add_observations(params)



pf.add_observations("heads.csv")
pf.build_pst("pest.pst")
pe = pf.draw(100)
pe.to_csv("prior.csv")






# define the model and observed data
model = AquaCrop()
observed_data = ...

# create a parameter estimation object 
pe = pyemu.ParameterEstimation(model, observed_data)

# define the parameter bounds
param_bounds = {'Kc': (0.1, 0.9),
                'ETc': (0.1, 0.9),
                'LAI': (0.1, 0.9)}

# run the calibration 
pe.run(param_bounds)

# get the calibrated parameters
calibrated_params = pe.get_calibrated_parameters()




import pyemu
from pyemu import PstFrom

# create a pest control file object
pest_control_file = PstFrom("my_model.pst")

# add parameters to the pest control file object 
pest_control_file.add_parameters(["par1", "par2", "par3"])

# add observations to the pest control file object 
pest_control_file.add_observations(["obs1", "obs2", "obs3"])

# write the pest control file to disk
pest_control_file.write("my_model.pst")





