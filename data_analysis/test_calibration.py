#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:15:08 2023

@author: wayne
"""

!pip install aquacrop==2.2
!pip install numba==0.55
!pip install statsmodels==0.13.2
!pip install SALib
!pip install pyswarms 

import matplotlib.pyplot as plt
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
import pickle 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sklearn.metrics as metrics
from pyswarm import pso
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
initWC = InitialWaterContent(value=['FC'])
irr_mngt = IrrigationManagement(irrigation_method=1,SMT=[80]*4) # no irrigation


# uncalibrated model - yield
crop_uncalibrated = Crop('Maize', planting_date='05/01') 

# run model
model_uc = AquaCropModel(sim_start,sim_end,wdf,custom,crop_uncalibrated,initWC, irr_mngt)
model_uc.run_model(till_termination=True) # run model till the end
#model_uc_et = model_uc._outputs.water_flux
model_uc_irr = model_uc._outputs.final_stats
#model_df_water_storage = model._outputs.water_storage
#model_df_crp_grwth = model._outputs.crop_growth

## Create year variable
model_uc_irr = model_uc_irr.assign(Year =  model_uc_irr['Harvest Date (YYYY/MM/DD)'].dt.year)
model_uc_irr = model_uc_irr.rename(columns={
                   'Yield (tonne/ha)': 'Uncalib Yield (t/ha)',
                   'Seasonal irrigation (mm)': 'Uncalib Irrigation (mm)'
                   })
model_uc_irr = model_uc_irr[['Year', 'Uncalib Yield (t/ha)', 'Uncalib Irrigation (mm)']]


# calibrated model - yield
crop_calibrated = Crop("Maize", planting_date='05/01',
            # growth stages
            CalendarType = 1,
            SwitchGDD = 1,  #Convert calendar to gdd mode if inputs are given in calendar days (0 = No; 1 = Yes)
            ##Crop Development
            Emergence = 15, #15
            Senescence = 42, #42
            Maturity = 131, #131
            MaxRooting = 91, #91
            HIstart = 83,
            Flowering = 67,
            YldForm = 60,
            
            # canopy cover profile
            #CCx = 0.94,
            CGC = 13.7,
            CDC = 1.31,
    
            # after running swam
            CCx = 0.99,
            WPy = 75,
            Wp = 33.3641,
            Kcb = 1.1


            
    )


# run model
model_c = AquaCropModel(sim_start,sim_end,wdf,custom,crop_calibrated,initWC, irr_mngt)
model_c.run_model(till_termination=True) # run model till the end
model_c_et = model_c._outputs.water_flux
model_c_irr = model_c._outputs.final_stats
#model_c_water_storage = model_c._outputs.water_storage
#model_c_crp_grwth = model_c._outputs.crop_growth

 # make year a column 
model_c_irr = model_c_irr.assign(Year =  model_c_irr['Harvest Date (YYYY/MM/DD)'].dt.year)
model_c_irr = model_c_irr.rename(columns={
                   'Yield (tonne/ha)': 'Calib Yield (t/ha)',
                   'Seasonal irrigation (mm)': 'Calib Irrigation (mm)'
                   })

model_c_irr = model_c_irr[['Year', 'Calib Yield (t/ha)', 'Calib Irrigation (mm)']]


# compare models
model_comp = model_uc_irr.merge(model_c_irr,on='Year').merge(sher_irrig_corn,on='Year').merge(sher_wimas, on = 'Year')
model_comp = model_comp.assign(YieldUSDA = model_comp['Value']*0.0673)


#model_comp = model_comp[model_comp['Year'] != 2009]
# comparison - time series plot
irrig_colors = ['#FFA500', '#8abbdb', '#69a6d0', '#4892c6', '#367bac', '#356384', '#344b5b']

plt.style.use('default')
plt.plot(model_comp['Year'], model_comp['Calib Yield (t/ha)'],  label='Calibrated AqC', color = 'goldenrod', linewidth=3)
plt.plot(model_comp['Year'], model_comp['Uncalib Yield (t/ha)'],  label='Uncalibrated AqC', color = 'grey', linewidth=3)
plt.plot(model_comp['Year'], model_comp['YieldUSDA'],  label='USDA-NASS', color = '#2e8b57', linestyle='dashed')
plt.ylabel('Yield (t/ha)', fontsize=15)
plt.legend(loc='lower right', fontsize=12, frameon=False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.savefig('results/visuals/yieldmodelcal_40SMT.png', format='png', dpi=600)




# 1:1 plot
plt.scatter(model_comp['YieldUSDA'], model_comp['Calib Yield (t/ha)'],  label='Calibrated AqC', color = 'goldenrod', edgecolors = 'black', s = 50)
plt.scatter(model_comp['YieldUSDA'], model_comp['Uncalib Yield (t/ha)'],  label='Uncalibrated AqC', color = 'grey', edgecolors = 'black', s=50)
plt.plot([10,15],[10,15], color = 'black')
plt.legend(loc='lower right', fontsize=12, frameon=False)
plt.ylabel('AquaCrop Yield (t/ha)', fontsize=15)
plt.xlabel('USDA-NASS (t/ha)', fontsize=15)
plt.legend(loc='lower right', fontsize=12, frameon=False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.savefig('results/visuals/yieldmodelcal_40SMT_11.png', format='png', dpi=600)




# calculate fit params
from statistics import mean
from sklearn.metrics import r2_score 

def model_fit(y, yhat):
    mae = metrics.mean_absolute_error(y, yhat)
    mse = metrics.mean_squared_error(y, yhat)
    rmse = np.sqrt(mse) # or mse**(0.5) 
    model = LinearRegression().fit(y, yhat)
    r2 = model.score(y, yhat)
  
    #r2 = r2_score(y, yhat)
    d_num = ((yhat.squeeze() - y.squeeze())**2).sum()
    d_denom1 = np.abs(yhat.squeeze()-y.squeeze().mean())
    d_denom2 = np.abs(y.squeeze()-y.squeeze().mean())
    d = 1 - (d_num/((d_denom1+d_denom2)**2).sum())
    pe = ((yhat.squeeze() - y.squeeze())/y.squeeze()) * 100
    sum_stats = {'var_name': ['mae', 'rmse', 'r2', 'd'],
                 'value': [mae, rmse, r2, d]}
                
    sum_stats_df = pd.DataFrame(sum_stats)
    
    return(sum_stats_df)

#model_comp = model_comp[model_comp['Year'] != 2009]

uncalib_yield = model_fit(model_comp[['YieldUSDA']], model_comp[['Uncalib Yield (t/ha)']])
uncalib_yield = uncalib_yield.rename(columns={"value": "uncalibrated"})

calib_yield = model_fit(model_comp[['YieldUSDA']], model_comp[['Calib Yield (t/ha)']])
calib_yield = calib_yield.rename(columns={"value": "calibrated"})

yield_fit = uncalib_yield.merge(calib_yield,on='var_name')








yield_fit.to_csv(r'./data/analysis_results/yield_fit_3.csv', sep=',', encoding='utf-8', header='true')




!pip install tabulate
from tabulate import tabulate
print(tabulate(yield_fit))



a = ((model_comp[['Uncalib Yield (t/ha)']].squeeze() - model_comp[['YieldUSDA']].squeeze())**2).sum()

b = np.abs(model_comp[['Uncalib Yield (t/ha)']].squeeze()-model_comp[['YieldUSDA']].squeeze().mean())
     
c = np.abs(model_comp[['YieldUSDA']].squeeze()-model_comp[['YieldUSDA']].squeeze().mean())
    
d = a/((b + c)**2).sum()










plt.plot(model_comp['Year'], model_comp['Calib Irrigation (mm)'],  label='Calibrated AquaCrop')
plt.plot(model_comp['Year'], model_comp['Uncalib Irrigation (mm)'],  label='Uncalibrated AquaCrop')
plt.plot(model_comp['Year'], model_comp['irrig_depth'],  label='USDA')
plt.ylabel('Yield (t/ha)', fontsize=25)
plt.legend(loc='lower right', fontsize=12)
#display plot
plt.show()




####### pyswarm calibration
import statistics as stats

def objFun(x):
    
    wpy, ccx, wp, kc = x
    crop_calibrated = Crop("Maize", planting_date='05/01',
                # growth stages
                CalendarType = 1,
                SwitchGDD = 1,  #Convert calendar to gdd mode if inputs are given in calendar days (0 = No; 1 = Yes)
                ##Crop Development
                Emergence = 15, #15
                Senescence = 42, #42
                Maturity = 131, #131
                MaxRooting = 91, #91
                HIstart = 83,
                Flowering = 67,
                YldForm = 60,
                
                # canopy cover profile
                CCx = ccx,
                CGC = 13.7,
                CDC = 1.31,
                
                
                WPy = wpy,
                Kcb = kc,
                WP = wp
        )


    # run model
    model_c = AquaCropModel(sim_start,sim_end,wdf,custom,crop_calibrated,initWC, irr_mngt)
    model_c.run_model(till_termination=True) # run model till the end
    model_c_et = model_c._outputs.water_flux
    model_c_irr = model_c._outputs.final_stats
    #model_c_water_storage = model_c._outputs.water_storage
    #model_c_crp_grwth = model_c._outputs.crop_growth

     # make year a column 
    model_c_irr = model_c_irr.assign(Year =  model_c_irr['Harvest Date (YYYY/MM/DD)'].dt.year)
    model_c_irr = model_c_irr.rename(columns={
                       'Yield (tonne/ha)': 'Calib Yield (t/ha)',
                       'Seasonal irrigation (mm)': 'Calib Irrigation (mm)'
                       })

    model_c_irr = model_c_irr[['Year', 'Calib Yield (t/ha)', 'Calib Irrigation (mm)']]

    #return model_c_irr

    # compare models
    model_comp = model_c_irr.merge(sher_irrig_corn,on='Year').merge(sher_wimas, on = 'Year')
    model_comp = model_comp.assign(YieldUSDA = model_comp['Value']*0.0673)
    
   # return(model_comp)
    
    
    y = model_comp[['YieldUSDA']]
    yhat = model_comp[['Calib Yield (t/ha)']]
    
    #print(y)
    #return(yhat)
    
    var = yhat.var()
    #fitness = ((y[['YieldUSDA']]-yhat[['Calib Yield (t/ha)']])**2).sum() # Calculate fitness score based on AquaCrop performance
    fitness = (((np.array(y)-np.array(yhat))**2).sum()).item()
    return fitness


# upper and lower bounds for wpy and ccx
lb = [75.000, 0.650, 30.00, 1.00]
ub = [125.00, 0.990, 35.00, 1.10]

xopt, fopt = pso(objFun, lb, ub)


objFun([75.000, 0.650])




