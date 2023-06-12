#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:41:50 2023

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
import sklearn.metrics as metrics
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
sher_gridMET = sher_gridMET[(sher_gridMET['year'] >= 2000) & (sher_gridMET['year'] <= 2020)]
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
sim_end = '2020/12/31'
custom_soil = custom # use custom layer for 1 site
crop = Crop('Maize', planting_date='05/01') 
initWC = InitialWaterContent(value=['FC'])
#irr_mngt = IrrigationManagement(irrigation_method=1,SMT=[80]*4)
irr_mngt = IrrigationManagement(irrigation_method=1,SMT=[80]*4) # no irrigation



# run model
model = AquaCropModel(sim_start,sim_end,wdf,custom_soil,crop,initWC, irr_mngt)
model.run_model(till_termination=True) # run model till the end
model_df_et = model._outputs.water_flux
model_df_irr = model._outputs.final_stats
#model_df_water_storage = model._outputs.water_storage
#model_df_crp_grwth = model._outputs.crop_growth

## Create year variable
model_df_irr = model_df_irr.assign(Year =  model_df_irr['Harvest Date (YYYY/MM/DD)'].dt.year)


### YIELD

# Yield Data - Irrigated 
# yield data from usda nass https://quickstats.nass.usda.gov/#D93A3218-8B77-31A6-B57C-5A5D97A157D8
yield_Irrig = pd.read_csv(wd + "/data/agricLand/yield/Kansas/CornYield_GMD4_WNdlovu_v1_20230117.csv") #CORN, GRAIN, IRRIGATED - YIELD, MEASURED IN BU / ACRE
#yield_noIrrig = pd.read_csv(wd.replace('code',"data/agricLand/gridMET/sheridanYield_noIrrig.csv")) #CORN, GRAIN, NON-  IRRIGATED - YIELD, MEASURED IN BU / ACRE

# Select year and irrigation value
yield_Irrig = yield_Irrig[(yield_Irrig['County'] == 'SHERIDAN') & (yield_Irrig['Irrig_status'] == 'irrigated')]
yield_Irrig = yield_Irrig[['Year', 'Value']]

# df with USDS NASS yield and Aquacrop yield
yield_df = model_df_irr
yield_df = pd.merge(yield_df, yield_Irrig, on=["Year", "Year"])
yield_df = yield_df.assign(YieldUSDA = yield_df['Value']*0.0673) # convert yield from bushels/acre to tonne/ha
yield_df  = yield_df[['Year', 'YieldUSDA','Yield (tonne/ha)']]

# rename columns
yield_df = yield_df.rename(columns={
                   'YieldUSDA': 'USDA Yield (t/ha)',
                   'Yield (tonne/ha)': 'AquaCrop Yield (t/ha)'})


### IRRIGATION
## get irrigation values from Aquacrop

irrig_aqc = model_df_irr
                                                            

irrig_aqc = irrig_aqc[irrig_aqc['Year'].between(2000, 2014)] # filter for to match Sheridan irrigated yield data
irrig_aqc = irrig_aqc[['Year', 'Seasonal irrigation (mm)']]

## Corn Water Use WIMAS
irrig_wimas = pd.read_csv(wd + "/data/water/Kansas/IrrigationDepth_GMD4_WNdlovu_v1_20230123.csv")
irrig_wimas = irrig_wimas[(irrig_wimas['county_abrev'] == 'SD') & (irrig_wimas['crop_name'] == 'Corn') & (irrig_wimas['WUA_YEAR'].between(2000, 2014))]

# calculate county ave irrigation # better method??
irrig_wimas = irrig_wimas.groupby(['WUA_YEAR']).agg({'irrig_depth': lambda x: x.median(skipna=True)})
irrig_wimas = irrig_wimas.reset_index()


# WIMAS and Aquacrop irrigation df
irrig_df = pd.merge(irrig_wimas, irrig_aqc, left_on = "WUA_YEAR", right_on = "Year")
#irrig_df  = irrig_df[['UID', 'Year','irrig_depth', 'Seasonal irrigation (mm)']]
irrig_df = irrig_df.rename(columns={
                   'irrig_depth': 'WIMAS Irrigation (mm)',
                   'Seasonal irrigation (mm)': 'AquaCrop Irrigation (mm)'})
irrig_df  = irrig_df[['Year', 'WIMAS Irrigation (mm)','AquaCrop Irrigation (mm)']]


uncal_corn_irrig = pd.merge(yield_df, irrig_df, on=["Year", "Year"])

from sklearn.linear_model import LinearRegression
# calculate fit params
def model_fit(y, yhat):
    mae = metrics.mean_absolute_error(y, yhat)
    mse = metrics.mean_squared_error(y, yhat)
    rmse = np.sqrt(mse) # or mse**(0.5) 
    model = LinearRegression().fit(y, yhat)
    #r2 = LinearRegression().fit(y,yhat)
    r2 = model.score(y, yhat)
    
    sum_stats = {'var_name': ['mae', 'mse', 'r2'],
                 'value': [mae, mse, r2]}
                
    sum_stats_df = pd.DataFrame(sum_stats)
    
    return(sum_stats_df)

yield_sumstats = model_fit(uncal_corn_irrig[['USDA Yield (t/ha)']], uncal_corn_irrig[['AquaCrop Yield (t/ha)']])
irrig_sumstats = model_fit(uncal_corn_irrig[['WIMAS Irrigation (mm)']], uncal_corn_irrig[['AquaCrop Irrigation (mm)']])


uncal_corn_irrig.to_csv(r'./data/analysis_results/uncalib_cornsimul_rainfed_sheridan.csv', sep=',', encoding='utf-8', header='true')








