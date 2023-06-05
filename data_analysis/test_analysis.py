#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:25:20 2022

@author: wayne
"""


!pip install aquacrop==2.2
!pip install numba==0.55
!pip install statsmodels==0.13.2
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
#from aquacrop.classes import    *
#from aquacrop.core import       *




wd=getcwd() # set working directory
chdir(wd)
soils_df_full = pd.read_csv(wd + '/data/agricLand/soils/Soil_FieldsAroundSD6KS_POLARIS_AGrinstead_20220706.csv')
soils_df = soils_df_full[soils_df_full['UID'] == 1381151] # filter for one site
soils_df = soils_df[soils_df['depth_cm'] == '0-5']


soils = pd.DataFrame(soils_df_full)
soils = soils[soils['depth_cm'] == '0-5'] # use upper 0.5cm
soils = soils.assign(om = (10**(soils['logOm_%'])),
                     Ksat_cmHr = (10**(soils['logKsat_cmHr'])))
soils = soils[['UID', 'depth_cm', 'silt_prc', 'sand_prc',
               'clay_prc', 'thetaS_m3m3', 'thetaR_m3m3',
               'Ksat_cmHr', 'lambda', 'logHB_kPa', 'n',
               'logAlpha_kPa1', 'om']]
#soils = soils.head(1)

# creating custom soil profile for all soils (run only for full model)
id_list = []
custom_soil = []
for i in range(0, len(soils)): # full model replace with soils
    ids = soils['UID'][i] #create df with UID from the soils file used - fix this
    id_list.append(ids)
    pred_thWP = ((-0.024*((soils['sand_prc'][i])/100))) + ((0.487*((soils['clay_prc'][i])/100))) + ((0.006*((soils['om'][i])/100))) + ((0.005*((soils['sand_prc'][i])/100))*((soils['om'][i])/100))- ((0.013*((soils['clay_prc'][i])/100))*((soils['om'][i])/100))+ ((0.068*((soils['sand_prc'][i])/100))*((soils['clay_prc'][i])/100))+ 0.031
    wp = pred_thWP + (0.14 * pred_thWP) - 0.02
    pred_thFC = ((-0.251*((soils['sand_prc'][i])/100))) + ((0.195*((soils['clay_prc'][i])/100)))+ ((0.011*((soils['om'][i])/100))) + ((0.006*((soils['sand_prc'][i])/100))*((soils['om'][i])/100))- ((0.027*((soils['clay_prc'][i])/100))*((soils['om'][i])/100))+ ((0.452*((soils['sand_prc'][i])/100))*((soils['clay_prc'][i])/100))+ 0.299
    fc = pred_thFC + (1.283 * (np.power(pred_thFC, 2))) - (0.374 * pred_thFC) - 0.015
    #fc = pred_thFC + (1.283 * (pred_thFC*pred_thFC)) - (0.374 * pred_thFC) - 0.015
    ts =soils["thetaS_m3m3"][i]
    ks=(soils['Ksat_cmHr'][i])*240
    #tp = soils['thetaR_m3m3'][i]
    custom = Soil('custom', dz=[0.1]*30)
    custom.add_layer(thickness=custom.zSoil,thS=ts, # assuming soil properties are the same in the upper 0.1m
                     Ksat=ks,thWP =wp , 
                     thFC = fc, penetrability = 100.0)
    custom_soil.append(custom)



# make dictionary with id as key and custom soils properties as value
soil_dict=dict(zip(id_list,custom_soil))

# save disctionary
#soil_dict_fl = csv.writer(open("output.csv", "w"))

# loop over dictionary keys and values
for key, val in soil_dict.items():

    # write every key and value to file
    soil_dict_fl.writerow([key, val])


# test to see if the dictionaries are working
#print(list(soil_dict.keys())[2])   
#print(list(soil_dict.values())[2])

# filter for dictionary with 1381151 test site
test_site = {k: v for k, v in soil_dict.items() if k == 1381151}  # filter for given site number
test_site = list(test_site.values())



## run 


## run model params
path = get_filepath(wd + '/data/hydrometeorology/gridMET/gridMET_1381151.txt') #replace folder name from folder name with file path
wdf = prepare_weather(path)
sim_start = '2000/01/01' #dates to match crop data
sim_end = '2020/12/31'
custom = test_site[0] # use custom layer for 1 site
crop = Crop('Maize', planting_date='05/01') 
initWC = InitialWaterContent(value=['FC'])
irr_mngt = IrrigationManagement(irrigation_method=1,SMT=[80]*4)
#irr_mngt = IrrigationManagement(irrigation_method = 0) # no irrigation



# run model
model = AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC, irr_mngt)
model.run_model(till_termination=True) # run model till the end
model_df_et = model._outputs.water_flux
model_df_irr = model._outputs.final_stats
model_df_water_storage = model._outputs.water_storage
model_df_crp_grwth = model._outputs.crop_growth


### IRRIGATION

## get irrigation values from Aquacrop
model_df_irr = model_df_irr.assign(Year =  model_df_irr['Harvest Date (YYYY/MM/DD)'].dt.year)

irrig_aqc = model_df_irr
irrig_aqc.reset_index(inplace=True)       # make year a column 
                                                                 

irrig_aqc = irrig_aqc[irrig_aqc['Year'].between(2006, 2018)] # filter for 2006>yrs>2018 to match WIMAS
irrig_aqc = irrig_aqc[['Year', 'Seasonal irrigation (mm)']]

## Water Rights WIMAS
wr_groups = pd.read_csv(wd + "/data/water/WRgroups_FieldByYear.csv")
water_use = pd.read_csv(wd + "/data/water/WRgroups_UseByWRG.csv")

# merge water right groups and water use
irrig_wimas = pd.merge(wr_groups, water_use, on=["WR_GROUP", "Year"]) # 
irrig_wimas = irrig_wimas[irrig_wimas['UID'] == 	1381151] # filter for field
irrig_wimas = irrig_wimas.assign(irrig_wimas = (irrig_wimas['Irrigation_m3']/(irrig_wimas['TRGT_ACRES']*4046.86))*1000)

# WIMAS and Aquacrop irrigation df
irrig_df = pd.merge(irrig_wimas, irrig_aqc, on=["Year", "Year"])
irrig_df  = irrig_df[['UID', 'Year','irrig_wimas', 'Seasonal irrigation (mm)']]



# Yield Data
# yield data from usda nass https://quickstats.nass.usda.gov/#D93A3218-8B77-31A6-B57C-5A5D97A157D8
yield_Irrig = pd.read_csv(wd + "/data/agricLand/yield/sheridanYield_Irrig.csv") #CORN, GRAIN, IRRIGATED - YIELD, MEASURED IN BU / ACRE
#yield_noIrrig = pd.read_csv(wd.replace('code',"data/agricLand/gridMET/sheridanYield_noIrrig.csv")) #CORN, GRAIN, NON-  IRRIGATED - YIELD, MEASURED IN BU / ACRE

# Select year and irrigation value
yield_Irrig = yield_Irrig[['Year', 'Value']]

# df with USDS NASS yield and Aquacrop yield
yield_df = model_df_irr
yield_df = pd.merge(yield_df, yield_Irrig, on=["Year", "Year"])
yield_df = yield_df.assign(YieldUSDA = yield_df['Value']*0.0673) # convert yield from bushels/acre to tonne/ha
yield_df  = yield_df[['Year', 'YieldUSDA','Yield (tonne/ha)']]

#### EVAPOTRANSPIRATION
# get date variable from the wdf
wdf_date = wdf[["Date"]]
wdf_date = wdf_date[wdf_date['Date'].between('2000/01/01','2020/12/31')] # filter for 
wdf_date = wdf_date[['Date']] # select date variable and drop second index column
wdf_date = wdf_date.reset_index() # reset index to start from 0


# add the date variable and jon by index
model_df_et = model_df_et.join(wdf_date)

# calculate monthly average ET value
model_df_et['yearmon'] = pd.to_datetime(model_df_et['Date']).dt.strftime('%Y-%m') # create yearmonth variable
model_df_et = model_df_et.assign(Et = model_df_et['Es'] + model_df_et['Tr'])
ave_et = model_df_et.groupby('yearmon')['Et'].sum()


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
fullET['date'] = pd.to_datetime(fullET['time'])
fullET['yearmon'] = pd.to_datetime(fullET['date']).dt.strftime('%Y-%m') # create year mon

# create df with aquacrop mean ET and other online models 
et_means = fullET[['time', 
                   'yearmon',
                   'et_mean_disalexi',
                   'et_mean_ensemble',
                   'et_mean_eemetric',
                   'et_mean_geesebal',
                   'et_mean_ptjpl',
                   'et_mean_sims',
                   'et_mean_ssebop']] # select mean ETs 

# rename colums
et_means = et_means.rename(columns={
                   'et_mean_disalexi': 'disalexi',
                   'et_mean_ensemble': 'ensemble',
                   'et_mean_eemetric': 'eemetric',
                   'et_mean_geesebal': 'geesebal',
                   'et_mean_ptjpl': 'ptjpl',
                   'et_mean_sims': 'sims',
                   'et_mean_ssebop': 'ssebop'}) 


et_means = et_means.merge(ave_et, left_on = 'yearmon', right_on = "yearmon")
et_means = et_means[['time', 'disalexi', 
                     'ensemble', 'geesebal', 
                     'ptjpl', 'sims', 'ssebop', 'Et']]



### SUMMARY STATS TABLE COMPARING MODELS

# r2
#r2_score was not working, so used function

def rsquared(x, y):
    model = LinearRegression().fit(x,y)
    return(model.score(x, y))

#r^2
r2_yield = rsquared(np.array(yield_df_test['USDA-NASS']).reshape(-1, 1), np.array(yield_df_test['Aquacrop']))
r2_irrigation = rsquared(np.array(irrig_df_test['WIMAS']).reshape(-1, 1), np.array(irrig_df_test['Aquacrop']))
r2_disalexi = rsquared(np.array(et_means_test['disalexi']).reshape(-1, 1), np.array(et_means_test['aquacrop']))
r2_ensemble = rsquared(np.array(et_means_test['ensemble']).reshape(-1, 1), np.array(et_means_test['aquacrop']))
r2_geesebal = rsquared(np.array(et_means_test['geesebal']).reshape(-1, 1), np.array(et_means_test['aquacrop']))
r2_ptjpl = rsquared(np.array(et_means_test['ptjpl']).reshape(-1, 1), np.array(et_means_test['aquacrop']))
r2_sims = rsquared(np.array(et_means_test['sims']).reshape(-1, 1), np.array(et_means_test['aquacrop']))
r2_ssebop = rsquared(np.array(et_means_test['ssebop']).reshape(-1, 1), np.array(et_means_test['aquacrop']))


# mean biased error
mbe_yield = (np.sum((yield_df_test['Aquacrop'] - yield_df_test['USDA-NASS'])))/len(yield_df_test['USDA-NASS'])
mbe_irrigation = (np.sum((irrig_df_test['Aquacrop']- irrig_df_test['WIMAS'])))/len(irrig_df_test['WIMAS'])
mbe_disalexi = (np.sum((et_means_test['aquacrop'] -et_means_test['disalexi'])))/len(et_means_test['aquacrop'])
mbe_ensemble = (np.sum((et_means_test['aquacrop'] - et_means_test['ensemble'])))/len(et_means_test['aquacrop'])
mbe_geesebal = (np.sum((et_means_test['aquacrop'] - et_means_test['geesebal'])))/len(et_means_test['aquacrop'])
mbe_ptjpl = (np.sum((et_means_test['aquacrop'] - et_means_test['ptjpl'])))/len(et_means_test['aquacrop'])
mbe_sims = (np.sum((et_means_test['aquacrop'] - et_means_test['sims'])))/len(et_means_test['aquacrop'])
mbe_ssebop = (np.sum((et_means_test['aquacrop'] - et_means_test['ssebop'])))/len(et_means_test['aquacrop'])


# root mean square error
rmse_yield = sqrt(mean_squared_error(yield_df_test['USDA-NASS'],yield_df_test['Aquacrop']))
rmse_irrigation = sqrt(mean_squared_error(irrig_df_test['WIMAS'],irrig_df_test['Aquacrop']))
rmse_disalexi = sqrt(mean_squared_error(et_means_test['disalexi'],et_means_test['aquacrop']))
rmse_ensemble = sqrt(mean_squared_error(et_means_test['ensemble'],et_means_test['aquacrop']))
rmse_geesebal = sqrt(mean_squared_error(et_means_test['geesebal'],et_means_test['aquacrop']))
rmse_ptjpl = sqrt(mean_squared_error(et_means_test['ptjpl'],et_means_test['aquacrop']))
rmse_sims = sqrt(mean_squared_error(et_means_test['sims'],et_means_test['aquacrop']))
rmse_ssebop = sqrt(mean_squared_error(et_means_test['ssebop'],et_means_test['aquacrop']))

# normalised rmse
nrmse_yield = sqrt(mean_squared_error(yield_df_test['USDA-NASS'],yield_df_test['Aquacrop']))/np.mean(yield_df_test['USDA-NASS'])
nrmse_irrigation = sqrt(mean_squared_error(irrig_df_test['WIMAS'],irrig_df_test['Aquacrop']))/np.mean(irrig_df_test['WIMAS'])
nrmse_disalexi = sqrt(mean_squared_error(et_means_test['disalexi'],et_means_test['aquacrop']))/np.mean(et_means_test['disalexi'])
nrmse_ensemble = sqrt(mean_squared_error(et_means_test['ensemble'],et_means_test['aquacrop']))/np.mean(et_means_test['ensemble'])
nrmse_geesebal = sqrt(mean_squared_error(et_means_test['geesebal'],et_means_test['aquacrop']))/np.mean(et_means_test['geesebal'])
nrmse_ptjpl = sqrt(mean_squared_error(et_means_test['ptjpl'],et_means_test['aquacrop']))/np.mean(et_means_test['ptjpl'])
nrmse_sims = sqrt(mean_squared_error(et_means_test['sims'],et_means_test['aquacrop']))/np.mean(et_means_test['sims'])
nrmse_ssebop = sqrt(mean_squared_error(et_means_test['ssebop'],et_means_test['aquacrop']))/np.mean(et_means_test['ssebop'])

# mean absolute error
mae_yield = mean_absolute_error(yield_df_test['USDA-NASS'],yield_df_test['Aquacrop'])
mae_irrigation = mean_absolute_error(irrig_df_test['WIMAS'],irrig_df_test['Aquacrop'])
mae_disalexi = mean_absolute_error(et_means_test['disalexi'],et_means_test['aquacrop'])
mae_ensemble = mean_absolute_error(et_means_test['ensemble'],et_means_test['aquacrop'])
mae_geesebal = mean_absolute_error(et_means_test['geesebal'],et_means_test['aquacrop'])
mae_ptjpl = mean_absolute_error(et_means_test['ptjpl'],et_means_test['aquacrop'])
mae_sims = mean_absolute_error(et_means_test['sims'],et_means_test['aquacrop'])
mae_ssebop = mean_absolute_error(et_means_test['ssebop'],et_means_test['aquacrop'])

# index of agreement
ia_yield = 1-(np.sum((yield_df_test['Aquacrop']- yield_df_test['USDA-NASS'])**2))/(np.sum(np.abs(yield_df_test['Aquacrop']-np.mean(yield_df_test['USDA-NASS']))+(np.abs(yield_df_test['USDA-NASS']-np.mean(yield_df_test['USDA-NASS']))))**2)      
ia_irrigation = 1-(np.sum((irrig_df_test['Aquacrop']-irrig_df_test['WIMAS'])**2))/(np.sum(np.abs(irrig_df_test['Aquacrop']-np.mean(irrig_df_test['WIMAS']))+(np.abs(irrig_df_test['WIMAS']-np.mean(irrig_df_test['WIMAS']))))**2)                     
ia_disalexi = 1-(np.sum((et_means_test['aquacrop']- et_means_test['disalexi'])**2))/(np.sum(np.abs(et_means_test['aquacrop']-np.mean(et_means_test['disalexi']))+(np.abs(et_means_test['disalexi']-np.mean(et_means_test['disalexi']))))**2)                     
ia_ensemble = 1-(np.sum((et_means_test['aquacrop']-et_means_test['ensemble'])**2))/(np.sum(np.abs(et_means_test['aquacrop']-np.mean(et_means_test['ensemble']))+(np.abs(et_means_test['ensemble']-np.mean(et_means_test['ensemble']))))**2)                          
ia_geesebal = 1-(np.sum((et_means_test['aquacrop']- et_means_test['geesebal'])**2))/(np.sum(np.abs(et_means_test['aquacrop']-np.mean(et_means_test['geesebal']))+(np.abs(et_means_test['geesebal']-np.mean(et_means_test['geesebal']))))**2)                    
ia_ptjpl  = 1-(np.sum((et_means_test['aquacrop'] -et_means_test['ptjpl'])**2))/(np.sum(np.abs(et_means_test['aquacrop']-np.mean(et_means_test['ptjpl']))+(np.abs(et_means_test['ptjpl']-np.mean(et_means_test['ptjpl']))))**2)     
ia_sims  = 1-(np.sum((et_means_test['aquacrop']-et_means_test['sims'])**2))/(np.sum(np.abs(et_means_test['aquacrop']-np.mean(et_means_test['sims']))+(np.abs(et_means_test['sims']-np.mean(et_means_test['sims']))))**2)
ia_ssebop = 1-(np.sum((et_means_test['aquacrop']-et_means_test['ssebop'])**2))/(np.sum(np.abs(et_means_test['aquacrop']-np.mean(et_means_test['ssebop']))+(np.abs(et_means_test['ssebop']-np.mean(et_means_test['ssebop']))))**2)



# make df with summary stats
sum_stats = {'var_name': ['yield', 'irrigation', 'et_disalexi', 'et_ensemble', 
                          'et_geesebal', 'et_ptjpl', 'et_sims', 'et_ssebop'],
        'range (observed)': ['10-14(t/ha)', '258-533(mm)', '8-184(mm)', '11-202(mm)',
                        '3-184(mm)', '9-187(mm)', '10-222(mm)', '20-231(mm)'],
        'mean (observed)': ['12(mm)', '384(mm)', '57(mm)', '71(mm)', '60(mm)', '66(mm)', '82(mm)', '96(mm)'],
        'r2': [r2_yield, r2_irrigation, r2_disalexi, r2_ensemble, r2_geesebal,
                  r2_ptjpl, r2_sims, r2_ssebop],   # not sure why r2 is used since it shows relationship between predictor and response variables
        'rmse': [rmse_yield, rmse_irrigation, rmse_disalexi, rmse_ensemble, rmse_geesebal,
                  rmse_ptjpl, rmse_sims, rmse_ssebop],
        'nrmse': [nrmse_yield, nrmse_irrigation, nrmse_disalexi, nrmse_ensemble, nrmse_geesebal,
                  nrmse_ptjpl, nrmse_sims, nrmse_ssebop],
        'mae':  [mae_yield, mae_irrigation, mae_disalexi, mae_ensemble, mae_geesebal,
                  mae_ptjpl, mae_sims, mae_ssebop],
        'mbe':  [mbe_yield, mbe_irrigation, mbe_disalexi, mbe_ensemble, mbe_geesebal,
                  mbe_ptjpl, mbe_sims, mbe_ssebop],
        'ia': [ia_yield, ia_irrigation, ia_disalexi, ia_ensemble, ia_geesebal,
                  ia_ptjpl, ia_sims, ia_ssebop]}

# save df
sum_stats_df = pd.DataFrame(sum_stats)



## WATER STORAGE
# add date column from the wdf_date df
model_df_water_storage = model_df_water_storage.join(wdf_date)
model_df_water_storage = model_df_water_storage.drop(columns=['index'])



### CROP GROWTH
model_df_crp_grwth = model_df_crp_grwth.join(wdf_date)
model_df_crp_grwth = model_df_crp_grwth.drop(columns=['index'])
model_df_crp_grwth = model_df_crp_grwth.assign(biomass_stress = model_df_crp_grwth['biomass_ns'] - model_df_crp_grwth['biomass']) # difference in biomass vals
model_df_crp_grwth = model_df_crp_grwth[model_df_crp_grwth['Date'].between('2000/01/01','2014/12/31')] # filter to match USDA dates (up to 2014)



## save new dfs in analysis_results - might change this to results/tables if file sizes are reasonable
irrig_df.to_csv(r'./data/analysis_resultsirrig_df_test.csv', sep=',', encoding='utf-8', header='true')
yield_df.to_csv(r'./data/analysis_results/yield_df_test.csv', sep=',', encoding='utf-8', header='true')
et_means.to_csv(r'./data/analysis_results/et_df_test.csv', sep=',', encoding='utf-8', header='true')
 



 
    