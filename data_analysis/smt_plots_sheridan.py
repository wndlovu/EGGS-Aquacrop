#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:03:42 2023

@author: wayne
"""
!pip install aquacrop==2.2
!pip install sklearn
from os import chdir, getcwd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from matplotlib.ticker import MaxNLocator
from matplotlib import patches
from os import chdir, getcwd
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
import pandas as pd
import sys
import seaborn as sns
import os
import glob
import pickle 
import numpy as np
_=[sys.path.append(i) for i in ['.', '..']]
wd=getcwd() # set working directory
chdir(wd)


# read in input data 
# corn yield
ks_corn = pd.read_csv(wd + '/data/agricLand/yield/Kansas/CornYield_GMD4_WNdlovu_v1_20230117.csv')
ks_irrig = pd.read_csv(wd + '/data/water/Kansas/IrrigationDepth_GMD4_WNdlovu_v1_20230123.csv')


# meteorological
with open(wd+'/data/hydrometeorology/gridMET/ks_gridMET.pickle', 'rb') as met: 
    gridMET_county = pickle.load(met)
   
    
with open(wd+'/data/groupings/ks_ccm.pickle', 'rb') as info: 
    grouped_info = pickle.load(info)   
    
# soils
with open(wd+'/data/agricLand/soils/ks_soil.pickle', 'rb') as sl: 
    soil_data = pickle.load(sl) 
    
 
# sheridan irrigated corn
sher_irrig_corn = ks_corn[(ks_corn['County'] == 'SHERIDAN') & (ks_corn['Irrig_status'] == 'irrigated')]
sher_irrig_corn = sher_irrig_corn[['Year', 'Value']] 

sher_wimas = ks_irrig[(ks_irrig['county_abrev'] == 'SD') & (ks_irrig['crop_name'] == 'Corn') & (ks_irrig['WUA_YEAR'].between(2000, 2014))]

# calculate county median irrigation # better method??
sher_wimas = sher_wimas.groupby(['WUA_YEAR']).agg({'irrig_depth': lambda x: x.median(skipna=True)})
sher_wimas = sher_wimas.reset_index()
sher_wimas = sher_wimas.rename(columns={
                   'WUA_YEAR': 'Year'
                   })
 
custom_soil = []
for soil_df in soil_data:
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
            custom_soil.append(custom)




# sheridan gridMET 
sher_gridMET = gridMET_county[10]
sher_gridMET = sher_gridMET.assign(year = sher_gridMET['Date'].dt.year) # create year variable
#sher_gridMET = sher_gridMET[sher_gridMET['year'] == 2012] # filter for 2012
sher_gridMET = sher_gridMET.drop(['year'], axis=1) # drop year variable
wdf = sher_gridMET

#wdf['MinTemp'].min()

# sheridan soils
sher_soils = custom_soil[10]
custom = sher_soils

sim_start = '2000/01/01' #dates to match crop data
sim_end = '2020/12/31'
crop = Crop('Maize', planting_date='05/01') 
initWC = InitialWaterContent(value=['FC'])



## Aquacrop Model

labels=[]
outputs=[]
for smt in range(0,110,20):
    crop.Name = str(smt) # add helpfull label
    labels.append(str(smt))
    irr_mngt = IrrigationManagement(irrigation_method=1,SMT=[smt]*4) # specify irrigation management [40,60,70,30]*4
    model = AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC,irr_mngt) # create model
    model.run_model(till_termination=True) # run model till the end
    outputs.append(model._outputs.final_stats) # save results
all_outputs = pd.concat(outputs)

all_outputs = all_outputs.assign(Year =  all_outputs['Harvest Date (YYYY/MM/DD)'].dt.year)


irrig_aqc = all_outputs.pivot(index= 'Year', # show irrigation vals for each year horizontally
                             columns='crop Type', 
                             values='Seasonal irrigation (mm)')


irrig_aqc.reset_index(inplace=True)  
irrig_aqc = irrig_aqc[irrig_aqc['Year'] <= 2014] # filter for yrs>2006 to match USDA NASS


# WIMAS and Aquacrop irrigation df
irrig_df = pd.merge(sher_wimas, irrig_aqc, on=["Year", "Year"])
irrig_df  = irrig_df[['Year', 'irrig_depth', '0', '20', '40', '60', '80', '100']]

# USDA NASS yield
yield_df = all_outputs.pivot(index= 'Year', # show irrigation vals for each year horizontally
                             columns='crop Type', 
                             values='Yield (tonne/ha)')

yield_df = pd.merge(yield_df, sher_irrig_corn, on=["Year", "Year"])
yield_df = yield_df.assign(YieldUSDA = yield_df['Value']*0.0673) # convert yield from bushels/acre to tonne/ha


# plots 
irrig_colors = ['#FFA500', '#8abbdb', '#69a6d0', '#4892c6', '#367bac', '#356384', '#344b5b']

# get the upper and lower quatile values for the WIMAS irrigation . Will use to add horizontal strip
sher_wimas.quantile([0.25,0.5,0.75])

sns.set(font_scale = 1.5)
sns.set_style("white")
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.boxplot(data=irrig_df, 
                 color = 'black',
                 palette=sns.color_palette(irrig_colors),
                 linewidth=1.2, 
                 fliersize=2, 
                 order=['irrig_depth', '0', '20', '40', '60', '80', '100'],
                 flierprops=dict(marker='o', markersize=4)) 
#ax.tick_params(bottom=False, left=True)
#ax.ayvspan(min_irrig, max_irrig, color='grey', alpha=0.5, lw=0)
rect = patches.Rectangle(
    xy=(ax.get_xlim()[0], 310),  # lower left corner of box: beginning of x-axis range & y coord)
    width=ax.get_xlim()[1]-ax.get_xlim()[0],  # width from x-axis range
    height=140,
    color='grey', alpha=0.2,
    zorder=1) #ec='red'
ax.add_patch(rect)
ax.set_xticklabels(['WIMAS', '0', '20', '40', '60', '80', '100'])
#ax.set_yticklabels(ax.get_yticks(), y.astype(int), size = 15)
ax.set_xlabel('Soil Moisture Threshold (%TAW)', size=25, weight='bold')
ax.set_ylabel('Irrigation (mm)', size = 25, weight='bold')


# yield
yield_df['20'].describe()


#print(sns.light_palette("seagreen").as_hex())
yield_colors = ['#FFA500', '#ebf3ed', '#c5decf', '#9fc9b1', '#7ab493', '#54a075', '#2e8b57']


sns.set(font_scale = 1.5)
sns.set_style("white")
fig, ax = plt.subplots(figsize=(10, 8))
bplot = sns.boxplot(data=yield_df, 
                 color = 'black',
                 palette=sns.color_palette(yield_colors),
                 linewidth=1.2, 
                 fliersize=2, 
                 order=['YieldUSDA', '0', '20', '40', '60', '80', '100'],
                 flierprops=dict(marker='o', markersize=4)) 
#ax.tick_params(bottom=False, left=True)
rect = patches.Rectangle(
    xy=(ax.get_xlim()[0], 11),  # lower left corner of box: beginning of x-axis range & y coord)
    width=ax.get_xlim()[1]-ax.get_xlim()[0],  # width from x-axis range
    height=3,
    color='grey', alpha=0.2) #ec='red'
ax.add_patch(rect)         
ax.set_xticklabels(['USDA', '0', '20', '40', '60', '80', '100'])
ax.set_xlabel('Soil Moisture Threshold (%TAW)', size=25, weight='bold')
ax.set_ylabel('Yield (t/ha)', size = 25, weight='bold')
