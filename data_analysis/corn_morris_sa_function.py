#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:12:37 2023

@author: wayne
"""

!pip install aquacrop==2.2
!pip install numba==0.55
!pip install statsmodels==0.13.2
!pip install SALib
from SALib.analyze import morris
from SALib.sample.morris import sample
from SALib.test_functions import Sobol_G
from SALib.util import read_param_file
from SALib.plotting.morris import (
    horizontal_bar_plot,
    covariance_plot,
    sample_histograms,
)
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
from SALib.util import read_param_file
import statistics
import matplotlib.transforms as mtransforms

wd=getcwd() # set working directory
chdir(wd)


# read in input data 
# sensitivity analysis params
corn_params = read_param_file(wd+"/data/sa_params/CornSA_GMD4_WNdlovu_v1_20230125.txt") # read in corn parameters

# parameter names
param_names = dict((k, corn_params[k]) for k in ['names'] # get key with names
           if k in corn_params)
param_names = param_names.get('names') # get values from names dictionary

# meteorological
with open(wd+'/data/hydrometeorology/gridMET/ks_gridMET.pickle', 'rb') as met: 
    gridMET_county = pickle.load(met)
  
    
with open(wd+'/data/analysis_results/sensitivity_analysis_runs/ks_irrig_corn_morris_r4.pickle', 'rb') as fgd: 
    irrig_morris3 = pickle.load(fgd)
    
  
    
with open(wd+'/data/groupings/ks_ccm.pickle', 'rb') as info: 
    grouped_info = pickle.load(info)   
    
# soils
with open(wd+'/data/agricLand/soils/ks_soil.pickle', 'rb') as sl: 
    soil_data = pickle.load(sl) 
    
 
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



param_values_sample = sample(corn_params, N=100, num_levels=10, optimal_trajectories=None) # 100,6,5generate array with param values
param_values = pd.DataFrame(param_values_sample) #transform array to df
param_values.columns = param_names # add column names


# sheridan gridMET 
#sher_data = [('Sheridan', 'Corn', 'irrigated')]
#h = [i for i in grouped_info if any(i for j in sher_data if str(j) in i)]

sher_gridMET = gridMET_county[10]
sher_gridMET = sher_gridMET.assign(year = sher_gridMET['Date'].dt.year) # create year variable

# get wettest and driest years
wet_dry = sher_gridMET.groupby(['year'])[['Precipitation']].sum() # dry = 2002, LEMA year 2012 and wet = 2019
wet_dry = wet_dry.reset_index()
median = wet_dry['Precipitation'].median()
normal =wet_dry[wet_dry['Precipitation'].astype('int').isin(range(int(median-10),int(median+10)))] # uneven num of year , but normal year is either 2004 or 2021, so use 2021


# sheridan soils
sher_soils = custom_soil[10]
custom = sher_soils

def MorrisCornIrrig(df, year):
    x = df  # make it generic
    x = x[x['year']== year]  # filter for yr

    wdf = x.drop(['year'], axis=1) # drop year variable
    #wdf = x
    sim_start = f'{year}/01/01'#'2021/01
    sim_end = f'{year}/12/31'
    initWC = InitialWaterContent(value=['FC'])
    #irr_mngt = IrrigationManagement(irrigation_meth
    model_irrig = []
    #while True:
    #model_irrig2 = pd.DataFrame([])  
   
    for i in range(0, len(param_values)):
        #pre_LEMA_stats = pd.DataFrame()
        #LEMA_stats = pd.DataFrame()
            #irr_mngt = IrrigationManagement(irrigation_method=1,
                                       #SMT=[param_values['smt'][i]]*4)
                                        
            #irr_mngt = IrrigationManagement(irrigation_method=1,
                                       #SMT=[param_values['smt'][i]]*4)#irrigation_irrigated # rainfed
            irr_mngt = IrrigationManagement(irrigation_method=1,
                                            SMT=[param_values['smt'][i]]*4)
            crop = Crop("Maize", planting_date='05/01', 
                        ##Crop Development
                        Tbase = param_values['tb'][i], 
                        Tupp = param_values['tu'][i], 
                        SeedSize = param_values['ccs'][i], 
                        PlantPop = param_values['den'][i],
                        Emergence = param_values['eme'][i],
                        CGC = param_values['cgc'][i], 
                        CCx = param_values['ccx'][i], 
                        Senescence = param_values['sen'][i],
                        CDC = param_values['cdc'][i], 
                        Maturity = param_values['mat'][i],
                        Zmin = param_values['rtm'][i],
                         
                        Flowering = param_values['flolen'][i],
                        Zmax = param_values['rtx'][i],
                        fshape_r = param_values['rtshp'][i],
                        MaxRooting = param_values['root'][i],
                        SxTopQ = param_values['rtexup'][i],
                        SxBotQ = param_values['rtexlw'][i],
                         #Crop Transpiration
                        Kcb = param_values['kc'][i],
                        fage = param_values['kcdcl'][i],
                         #Biomass and Yield
                        WP = param_values['wp'][i],
                        WPy = param_values['wpy'][i],
                        HI0 = param_values['hi'][i],
                        dHI_pre = param_values['hipsflo'][i],
                        exc = param_values['exc'][i],
                        a_HI = param_values['hipsveg'][i],
                        b_HI = param_values['hingsto'][i],
                        dHI0 = param_values['hinc'][i],
                        YldForm = param_values['hilen'][i],
                         #Water and Temperature Stress
                        Tmin_up = param_values['polmn'][i],
                        Tmax_up = param_values['polmx'][i],
                        p_up = param_values['pexup'][i],
                        p_lo = param_values['pexlw'][i],
                        fshape_w = param_values['pexshp'][i])
            #print(crop)           
                       ## run model params
            model = AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC, irr_mngt)
            #model_irrig.append(model)
            model.run_model(till_termination=True) # run model till the end
            model_df = model._outputs.final_stats
            #print(model_df)
           # model_df = model_df.reset_index()
            model_irrig.append(model_df)
            
    model_df_full = pd.concat(model_irrig)
    
    # irrigation sa
    irrig_df = model_df_full[['Seasonal irrigation (mm)']] # select yield variable
    # yield sa
    if (irrig_df['Seasonal irrigation (mm)'].max()) == 0:
        Si_irrig_df = pd.DataFrame()
        Si_irrig_infl = Si_irrig_df
        
        
        
        yield_df = model_df_full[['Yield (tonne/ha)']] # select yield variable
        yield_vals = yield_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
        Si_yield = morris.analyze(
            corn_params,
            param_values_sample,
            yield_vals,
            conf_level=0.95,
            print_to_console=True,
            num_levels=10,
            num_resamples=100,
            )

        Si_yield_df = pd.DataFrame(Si_yield)
        Si_yield_infl = Si_yield_df[Si_yield_df['mu_star'] > 0.3]
        Si_yield_infl = Si_yield_infl.sort_values(by=['mu_star'], ascending=False)
    
   
    
    
    else:
        irrig_vals = irrig_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
    

        Si_irrig = morris.analyze(
            corn_params,
            param_values_sample,
            irrig_vals,
            conf_level=0.95,
            print_to_console=True,
            num_levels=10,
            num_resamples=100,
            )

  
        Si_irrig_df = pd.DataFrame(Si_irrig)
        Si_irrig_infl = Si_irrig_df[Si_irrig_df['mu_star'] > 20] # max is 207
        Si_irrig_infl = Si_irrig_infl.sort_values(by=['mu_star'], ascending=False)
        
        
        yield_df = model_df_full[['Yield (tonne/ha)']] # select yield variable
        yield_vals = yield_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
        Si_yield = morris.analyze(
            corn_params,
            param_values_sample,
            yield_vals,
            conf_level=0.95,
            print_to_console=True,
            num_levels=10,
            num_resamples=100,
            )

        Si_yield_df = pd.DataFrame(Si_yield)
        Si_yield_infl = Si_yield_df[Si_yield_df['mu_star'] > 0.3] #max around 3.2
        Si_yield_infl = Si_yield_infl.sort_values(by=['mu_star'], ascending=False)
    
    return (Si_yield_df, Si_yield_infl, Si_irrig_df, Si_irrig_infl)



def MorrisCornRainfed(df, year):
    x = df  # make it generic
    x = x[x['year']== year]  # filter for yr

    wdf = x.drop(['year'], axis=1) # drop year variable
    #wdf = x


# sheridan soils
    sher_soils = custom_soil[10]
    custom = sher_soils

    sim_start = f'{year}/01/01'#'2021/01
    sim_end = f'{year}/12/31'
    initWC = InitialWaterContent(value=['FC'])
    #irr_mngt = IrrigationManagement(irrigation_meth
    model_irrig = []
    #while True:
    #model_irrig2 = pd.DataFrame([])  
   
    for i in range(0, len(param_values)):
        #pre_LEMA_stats = pd.DataFrame()
        #LEMA_stats = pd.DataFrame()
            #irr_mngt = IrrigationManagement(irrigation_method=1,
                                       #SMT=[param_values['smt'][i]]*4)
                                        
            #irr_mngt = IrrigationManagement(irrigation_method=1,
                                       #SMT=[param_values['smt'][i]]*4)#irrigation_irrigated # rainfed
            irr_mngt = IrrigationManagement(irrigation_method=0)
            crop = Crop("Maize", planting_date='05/01', 
                        ##Crop Development
                        Tbase = param_values['tb'][i], 
                        Tupp = param_values['tu'][i], 
                        SeedSize = param_values['ccs'][i], 
                        PlantPop = param_values['den'][i],
                        Emergence = param_values['eme'][i],
                        CGC = param_values['cgc'][i], 
                        CCx = param_values['ccx'][i], 
                        Senescence = param_values['sen'][i],
                        CDC = param_values['cdc'][i], 
                        Maturity = param_values['mat'][i],
                        Zmin = param_values['rtm'][i],
                         
                        Flowering = param_values['flolen'][i],
                        Zmax = param_values['rtx'][i],
                        fshape_r = param_values['rtshp'][i],
                        MaxRooting = param_values['root'][i],
                        SxTopQ = param_values['rtexup'][i],
                        SxBotQ = param_values['rtexlw'][i],
                         #Crop Transpiration
                        Kcb = param_values['kc'][i],
                        fage = param_values['kcdcl'][i],
                         #Biomass and Yield
                        WP = param_values['wp'][i],
                        WPy = param_values['wpy'][i],
                        HI0 = param_values['hi'][i],
                        dHI_pre = param_values['hipsflo'][i],
                        exc = param_values['exc'][i],
                        a_HI = param_values['hipsveg'][i],
                        b_HI = param_values['hingsto'][i],
                        dHI0 = param_values['hinc'][i],
                        YldForm = param_values['hilen'][i],
                         #Water and Temperature Stress
                        Tmin_up = param_values['polmn'][i],
                        Tmax_up = param_values['polmx'][i],
                        p_up = param_values['pexup'][i],
                        p_lo = param_values['pexlw'][i],
                        fshape_w = param_values['pexshp'][i])
            #print(crop)           
                       ## run model params
            model = AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC, irr_mngt)
            #model_irrig.append(model)
            model.run_model(till_termination=True) # run model till the end
            model_df = model._outputs.final_stats
            #print(model_df)
           # model_df = model_df.reset_index()
            model_irrig.append(model_df)
            
    model_df_full = pd.concat(model_irrig)
    
    # irrigation sa
    #irrig_df = model_df_full[['Seasonal irrigation (mm)']] # select yield variable
    # yield sa
    #if (irrig_df['Seasonal irrigation (mm)'].max()) == 0:
        #Si_irrig_df = pd.DataFrame(columns=np.arange(5)) # create empty df 
        #col_names = ['names', 'mu', 'mu_star', 'sigma', 'mu_star_conf']
        #Si_irrig_df.columns = [col_names] # rename columns to match those from morris.analyse
        #Si_irrig_infl = Si_irrig_df
        
        
    yield_df = model_df_full[['Yield (tonne/ha)']] # select yield variable
    yield_vals = yield_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
    Si_yield = morris.analyze(
            corn_params,
            param_values_sample,
            yield_vals,
            conf_level=0.95,
            print_to_console=True,
            num_levels=10,
            num_resamples=100,
            )

    Si_yield_df = pd.DataFrame(Si_yield)
    Si_yield_infl = Si_yield_df[Si_yield_df['mu_star'] > 0.1] # smaller values ( max yield mu_sta is 1.3 (2019))
    Si_yield_infl = Si_yield_infl.sort_values(by=['mu_star'], ascending=False)
    
   
    # create empty dataframes as irrigation placeholders to make the plot making function work
    irrig_ph1 = pd.DataFrame(columns=np.arange(5)) # create empty df 
    col_names = ['names', 'mu', 'mu_star', 'sigma', 'mu_star_conf']
    irrig_ph1.columns = [col_names]
    irrig_ph2 = pd.DataFrame()
    #else:
        #irrig_vals = irrig_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
    

       # Si_irrig = morris.analyze(
            #corn_params,
            #param_values_sample,
            #irrig_vals,
            #conf_level=0.95,
            #print_to_console=True,
            #num_levels=10,
            #num_resamples=100,
            #)

  
        #Si_irrig_df = pd.DataFrame(Si_irrig)
        #Si_irrig_infl = Si_irrig_df[Si_irrig_df['mu_star'] > 20]
        #Si_irrig_infl = Si_irrig_infl.sort_values(by=['mu_star'], ascending=False)
        
        
        #yield_df = model_df_full[['Yield (tonne/ha)']] # select yield variable
        #yield_vals = yield_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
        #Si_yield = morris.analyze(
            #corn_params,
            #param_values_sample,
            #yield_vals,
            #conf_level=0.95,
            #print_to_console=True,
            #num_levels=10,
            #num_resamples=100,
            #)

        #Si_yield_df = pd.DataFrame(Si_yield)
        #Si_yield_infl = Si_yield_df[Si_yield_df['mu_star'] > 0.1] # smaller values ( max yield mu_sta is 1.3 (2019))
        #Si_yield_infl = Si_yield_infl.sort_values(by=['mu_star'], ascending=False)
    
    return (Si_yield_df, Si_yield_infl)



# treatments and years
sim_years = [2002, 2019, 2021]
weather = [sher_gridMET]
#irrig_strategy = [IrrigationManagement(irrigation_method=1,
                           #SMT=[param_values['smt'][i]]*4)] 
#rainfed_strategy= [IrrigationManagement(irrigation_method=0)]


irrig_corn_morris = list(map(MorrisCornIrrig, weather*3, sim_years))


rainfed_corn_morris = list(map(MorrisCornRainfed, weather*3, sim_years)) 



# min max temp btween May and October (growing season)

mean_climate = sher_gridMET[sher_gridMET['year'].isin(sim_years)]
mean_climate = mean_climate.assign(month = sher_gridMET['Date'].dt.month)

mean_climate = mean_climate[mean_climate['month'].between(4, 10)]

# get the number of days with precipitation
rainfall_days = mean_climate[mean_climate['Precipitation'] > 0]
rainfall_days = rainfall_days.groupby(['year'])['Precipitation'].count()

# get average stats
mean_climate = mean_climate.groupby(['year'])['Precipitation','ReferenceET', 'MinTemp', 'MaxTemp'].agg(['sum', 'max','min', 'mean', 'median'])


# make individual morris plots
# make plots first coz saving first messes up the analysis
def morris_plots(mo_list):
    if len(mo_list) < 3: # irrigated simulation
    #yield  
    #Si_yield.sort_values(by=['mu_star'], inplace = True)
        Si_yield = mo_list[0]
        
        Si_yield_t5 = Si_yield.nlargest(3, ['mu_star'])
        
        #horizontal bar
        ##fig1, (ax1) = plt.subplots()
        ##plt.barh(y=Si_yield.names, width=Si_yield.mu_star, edgecolor='black', color = 'lightgrey')
        #plt.errorbar(y=Si_yield.names, width=Si_yield.mu_star, yerr=0.5, fmt="o", color="r")
        ##plt.xlabel('μ∗ [mm]')
        #horizontal_bar_plot(ax1, Si_yield, sortby="mu_star", unit=r"yield (t/ha)")
        #covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
        ##fig1.tight_layout(pad=1.2)
        #fig.savefig('results/visuals/sheridan_corn_yield_sa_irrig_2019_n100.png', format='png', dpi=1000, orientation = 'landscape')
        
        fig2, ax2 = plt.subplots(figsize=(7,5))
        ax2.scatter(Si_yield['mu_star'],Si_yield['sigma'], s=120, c = "lightgrey", edgecolors = 'grey')
        ax2.scatter(Si_yield_t5['mu_star'],Si_yield_t5['sigma'], s=120, c = "darkred", edgecolors = 'black')
        ax2.set_xlim(-0.01, (Si_yield['mu_star'].max()+0.15))
        ax2.set_ylim(-0.01, (Si_yield['sigma'].max()+0.1))
        ax2.xaxis.set_tick_params(width=1.5)
        plt.setp(ax2.spines.values(), linewidth=1.5)
        plt.xlabel('μ * [t/ha]', fontsize=25)
        plt.ylabel('σ [t/ha]', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0, 1.5)
        plt.xlim(0, 1.5)
        for j, txt in Si_yield_t5.iterrows():
            ax2.annotate(txt['names'], (txt['mu_star'], txt['sigma']), size = 22,
                         textcoords='offset points',  xytext=(4,4), ha='left', color = 'black')
        
        
        return fig2 # just yield 
        
    else:
        Si_yield = mo_list[0]
        
        Si_yield_t5 = Si_yield.nlargest(3, ['mu_star'])
        
        #horizontal bar
        ##fig1, (ax1) = plt.subplots()
        ##plt.barh(y=Si_yield.names, width=Si_yield.mu_star, edgecolor='black', color = 'lightgrey')
        #plt.errorbar(y=Si_yield.names, width=Si_yield.mu_star, yerr=0.5, fmt="o", color="r")
        ##plt.xlabel('μ∗ [mm]')
        #horizontal_bar_plot(ax1, Si_yield, sortby="mu_star", unit=r"yield (t/ha)")
        #covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
        ##fig1.tight_layout(pad=1.2)
        #fig.savefig('results/visuals/sheridan_corn_yield_sa_irrig_2019_n100.png', format='png', dpi=1000, orientation = 'landscape')
        
        fig2, ax2 = plt.subplots(figsize=(7,5))
        ax2.scatter(Si_yield['mu_star'],Si_yield['sigma'], s=120, c = "lightgrey", edgecolors = 'grey')
        ax2.scatter(Si_yield_t5['mu_star'],Si_yield_t5['sigma'], s=120, c = "darkred", edgecolors = 'black')
        ax2.set_xlim(-0.05, (Si_yield['mu_star'].max()+0.6))
        ax2.set_ylim(-0.05, (Si_yield['sigma'].max()+0.05))
        ax2.xaxis.set_tick_params(width=1.5)
        plt.setp(ax2.spines.values(), linewidth=1.5)
        plt.xlabel('μ * [t/ha]', fontsize=25)
        plt.ylabel('σ [t/ha]', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0, 1.4)
        for j, txt in Si_yield_t5.iterrows():
            ax2.annotate(txt['names'], (txt['mu_star'], txt['sigma']), size = 22,
                         textcoords='offset points',  xytext=(5,5), ha='right', color = 'black')
        
        
        # irrigation
        Si_irrig = mo_list[2]
        
        
        Si_irrig_t5 = Si_irrig.nlargest(3, ['mu_star'])
    
        #horizontal bar
        ##fig3, (ax3) = plt.subplots()
        ##plt.barh(y=Si_irrig.names, width=Si_irrig.mu_star, edgecolor='black', color = 'lightgrey')
        #plt.errorbar(y=Si_yield.names, width=Si_yield.mu_star, yerr=0.5, fmt="o", color="r")
        ##plt.xlabel('μ∗ [mm]')
        #horizontal_bar_plot(ax1, Si_yield, sortby="mu_star", unit=r"yield (t/ha)")
        #covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
        ##fig3.tight_layout(pad=1.2)
    
    
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.scatter(Si_irrig['mu_star'],Si_irrig['sigma'], s=120, c = "lightgrey", edgecolors = 'grey')
        ax4.scatter(Si_irrig_t5['mu_star'],Si_irrig_t5['sigma'], s=120, c = "darkred", edgecolors = 'black')
        ax4.set_xlim(-3, (Si_irrig['mu_star'].max()+30))
        ax4.set_ylim(-3, (Si_irrig['sigma'].max()+10))
        ax4.xaxis.set_tick_params(width=1.5)
        plt.setp(ax4.spines.values(), linewidth=1.5)
        plt.xlabel('μ * [mm]', fontsize=25)
        plt.ylabel('σ [mm]', fontsize=25)
        plt.ylim(0, 100)
        plt.xlim(0, 250)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        for j, txt in Si_irrig_t5.iterrows():
            ax4.annotate(txt['names'], (txt['mu_star'], txt['sigma']), size = 22,
                         textcoords='offset points',  xytext=(5,5), ha='right', color = 'black')
            
        return fig2, fig4 # only return scatterplots
    


def morris_plots(mo_list):
    if len(mo_list) < 3: # irrigated simulation
    #yield  
    #Si_yield.sort_values(by=['mu_star'], inplace = True)
        Si_yield = mo_list[0]
        
        Si_yield_t5 = Si_yield.nlargest(5, ['mu_star'])
        
        conditions = [
            Si_yield_t5['names'].isin(['tb',
                               'tu',
                               'ccs',
                                'den',
                                'eme',
                                'cgc',
                                'ccx',
                                'sen',
                                'cdc',
                                'mat',
                                'rtm',
                                'flolen',
                                'rtx',
                                'rtshp',
                                'root',
                                'rtexup',
                                'rtexlw',
                                 'kc',
                                 'kcdcl']),
            Si_yield_t5['names'].isin(['wp',
                               'wpy',
                               'hi',
                               'hipsflo',
                               'exc',
                               'hipsveg',
                               'hingsto',
                                'hinc',
                                'hilen']),
            Si_yield_t5['names'].isin(['smt'])
        ]
        
        choices = ['crop dvpt', 'biomass', 'mngt']
        Si_yield_t5['group'] = np.select(conditions, choices, default=np.nan)
        
        
        colors_bottom = {'crop dvpt': '#377B2B', 'biomass': '#C68642', 'mngt': '#113a9b'} #peachpuff   #blue
        #horizontal bar
        ##fig1, (ax1) = plt.subplots()
        ##plt.barh(y=Si_yield.names, width=Si_yield.mu_star, edgecolor='black', color = 'lightgrey')
        #plt.errorbar(y=Si_yield.names, width=Si_yield.mu_star, yerr=0.5, fmt="o", color="r")
        ##plt.xlabel('μ∗ [mm]')
        #horizontal_bar_plot(ax1, Si_yield, sortby="mu_star", unit=r"yield (t/ha)")
        #covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
        ##fig1.tight_layout(pad=1.2)
        #fig.savefig('results/visuals/sheridan_corn_yield_sa_irrig_2019_n100.png', format='png', dpi=1000, orientation = 'landscape')
        
        fig2, ax2 = plt.subplots(figsize=(7,5))
        ax2.scatter(Si_yield['mu_star'],Si_yield['sigma'], s=120, c = "lightgrey", edgecolors = 'grey')
        ax2.scatter(Si_yield_t5['mu_star'],Si_yield_t5['sigma'], s=120, color = [colors_bottom[f] for f in Si_yield_t5['group']], edgecolors = 'black')
        ax2.set_xlim(-0.01, (Si_yield['mu_star'].max()+0.15))
        ax2.set_ylim(-0.01, (Si_yield['sigma'].max()+0.1))
        ax2.xaxis.set_tick_params(width=1.5)
        plt.setp(ax2.spines.values(), linewidth=1.5)
        plt.xlabel('μ * [t/ha]', fontsize=25)
        plt.ylabel('σ [t/ha]', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0, 1.8)
        plt.xlim(0, 1.8)
        for j, txt in Si_yield_t5.iterrows():
            ax2.annotate(txt['names'], (txt['mu_star'], txt['sigma']), size = 21,
                         textcoords='offset points',  xytext=(4,4), ha='left', color = 'black')
        
        
        return fig2 # just yield 
        
    else:
        Si_yield = mo_list[0]
        
        Si_yield_t5 = Si_yield.nlargest(5, ['mu_star'])
        
        conditions = [
            Si_yield_t5['names'].isin(['tb',
                               'tu',
                               'ccs',
                                'den',
                                'eme',
                                'cgc',
                                'ccx',
                                'sen',
                                'cdc',
                                'mat',
                                'rtm',
                                'flolen',
                                'rtx',
                                'rtshp',
                                'root',
                                'rtexup',
                                'rtexlw',
                                 'kc',
                                 'kcdcl']),
            Si_yield_t5['names'].isin(['wp',
                               'wpy',
                               'hi',
                               'hipsflo',
                               'exc',
                               'hipsveg',
                               'hingsto',
                                'hinc',
                                'hilen']),
            Si_yield_t5['names'].isin(['smt'])
        ]
        
        choices = ['crop dvpt', 'biomass', 'mngt']
        Si_yield_t5['group'] = np.select(conditions, choices, default=np.nan)
        
        
        colors_bottom = {'crop dvpt': '#377B2B', 'biomass': '#C68642', 'mngt': '#113a9b'} #peachpuff   #blue
        
        #horizontal bar
        ##fig1, (ax1) = plt.subplots()
        ##plt.barh(y=Si_yield.names, width=Si_yield.mu_star, edgecolor='black', color = 'lightgrey')
        #plt.errorbar(y=Si_yield.names, width=Si_yield.mu_star, yerr=0.5, fmt="o", color="r")
        ##plt.xlabel('μ∗ [mm]')
        #horizontal_bar_plot(ax1, Si_yield, sortby="mu_star", unit=r"yield (t/ha)")
        #covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
        ##fig1.tight_layout(pad=1.2)
        #fig.savefig('results/visuals/sheridan_corn_yield_sa_irrig_2019_n100.png', format='png', dpi=1000, orientation = 'landscape')
        
        fig2, ax2 = plt.subplots(figsize=(7,5))
        ax2.scatter(Si_yield['mu_star'],Si_yield['sigma'], s=120, c = "lightgrey", edgecolors = 'grey')
        ax2.scatter(Si_yield_t5['mu_star'],Si_yield_t5['sigma'], s=120, color = [colors_bottom[f] for f in Si_yield_t5['group']], edgecolors = 'black')
        ax2.set_xlim(-0.05, (Si_yield['mu_star'].max()+0.6))
        ax2.set_ylim(-0.05, (Si_yield['sigma'].max()+0.05))
        ax2.xaxis.set_tick_params(width=1.5)
        plt.setp(ax2.spines.values(), linewidth=1.5)
        plt.xlabel('μ * [t/ha]', fontsize=25)
        plt.ylabel('σ [t/ha]', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0, 1.6)
        for j, txt in Si_yield_t5.iterrows():
            ax2.annotate(txt['names'], (txt['mu_star'], txt['sigma']), size = 21,
                         textcoords='offset points',  xytext=(5,5), ha='right', color = 'black')
        
        
        # irrigation
        Si_irrig = mo_list[2]
        
        
        Si_irrig_t5 = Si_irrig.nlargest(5, ['mu_star'])
        
        conditions = [
            Si_irrig_t5['names'].isin(['tb',
                               'tu',
                               'ccs',
                                'den',
                                'eme',
                                'cgc',
                                'ccx',
                                'sen',
                                'cdc',
                                'mat',
                                'rtm',
                                'flolen',
                                'rtx',
                                'rtshp',
                                'root',
                                'rtexup',
                                'rtexlw',
                                 'kc',
                                 'kcdcl']),
            Si_irrig_t5['names'].isin(['wp',
                               'wpy',
                               'hi',
                               'hipsflo',
                               'exc',
                               'hipsveg',
                               'hingsto',
                                'hinc',
                                'hilen']),
            Si_irrig_t5['names'].isin(['smt'])
        ]
        
        choices = ['crop dvpt', 'biomass', 'mngt']
        Si_irrig_t5['group'] = np.select(conditions, choices, default=np.nan)
        
        
        colors_bottom = {'crop dvpt': '#377B2B', 'biomass': '#C68642', 'mngt': '#113a9b'} #peachpuff   #blue
        
    
    
    
    
        #horizontal bar
        ##fig3, (ax3) = plt.subplots()
        ##plt.barh(y=Si_irrig.names, width=Si_irrig.mu_star, edgecolor='black', color = 'lightgrey')
        #plt.errorbar(y=Si_yield.names, width=Si_yield.mu_star, yerr=0.5, fmt="o", color="r")
        ##plt.xlabel('μ∗ [mm]')
        #horizontal_bar_plot(ax1, Si_yield, sortby="mu_star", unit=r"yield (t/ha)")
        #covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
        ##fig3.tight_layout(pad=1.2)
    
    
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.scatter(Si_irrig['mu_star'],Si_irrig['sigma'], s=120, c = "lightgrey", edgecolors = 'grey')
        ax4.scatter(Si_irrig_t5['mu_star'],Si_irrig_t5['sigma'], s=120, color = [colors_bottom[f] for f in Si_irrig_t5['group']], edgecolors = 'black')
        ax4.set_xlim(-3, (Si_irrig['mu_star'].max()+30))
        ax4.set_ylim(-3, (Si_irrig['sigma'].max()+10))
        ax4.xaxis.set_tick_params(width=1.5)
        plt.setp(ax4.spines.values(), linewidth=1.5)
        plt.xlabel('μ * [mm]', fontsize=25)
        plt.ylabel('σ [mm]', fontsize=25)
        plt.ylim(0, 100)
        plt.xlim(0, 250)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        for j, txt in Si_irrig_t5.iterrows():
            ax4.annotate(txt['names'], (txt['mu_star'], txt['sigma']), size = 21,
                         textcoords='offset points',  xytext=(5,5), ha='right', color = 'black')
            
        return fig2, fig4 # only return scatterplots
    



# test function - plots look just ok - need more work
with open(wd+'/data/analysis_results/sensitivity_analysis_runs/ks_irrig_corn_morris_n1000.pickle', 'rb') as cr22: 
    irrig_corn_morris = pickle.load(cr22)

with open(wd+'/data/analysis_results/sensitivity_analysis_runs/ks_rainfed_corn_morris_n1000.pickle', 'rb') as cr23: 
    rainfed_corn_morris = pickle.load(cr23)


irrg2002,irr2019, irr2021= map(morris_plots, irrig_corn_morris)
rain2002, rain2019, rain2021= map(morris_plots, rainfed_corn_morris)


# yield params 
fig5, ax5 = plt.subplots(nrows=2, ncols=3, figsize=(24, 11))
ax5[0,0].imshow(irrg2002[0].canvas.renderer.buffer_rgba())
ax5[0,1].imshow(irr2021[0].canvas.renderer.buffer_rgba())
ax5[0,2].imshow(irr2019[0].canvas.renderer.buffer_rgba())
ax5[1,0].imshow(rain2002.canvas.renderer.buffer_rgba())
ax5[1,1].imshow(rain2021.canvas.renderer.buffer_rgba())
ax5[1,2].imshow(rain2019.canvas.renderer.buffer_rgba())
#ax5[0,0].set_title('Dry-Irrigated', fontweight ="bold", size = 17, pad = 8)
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig5.dpi_scale_trans)
ax5[0,0].text(0.55, 0.9, s = 'Dry-Irrigated', transform=ax5[0,0].transAxes + trans,#size = 20,
            fontsize=20, verticalalignment='top', weight = 'bold',
            bbox=dict(facecolor='white', edgecolor='white', pad=3.0))
ax5[0,1].text(0.42, 0.9, s = 'Normal-Irrigated', transform=ax5[0,1].transAxes + trans,#size = 20,
            fontsize=20, verticalalignment='top', weight = 'bold',
            bbox=dict(facecolor='white', edgecolor='white', pad=3.0))
ax5[0,2].text(0.52, 0.9, s = 'Wet-Irrigated', transform=ax5[0,2].transAxes + trans,#size = 20,
            fontsize=20, verticalalignment='top', weight = 'bold',
            bbox=dict(facecolor='white', edgecolor='white', pad=3.0))
ax5[1,0].text(0.57, 0.9, s = 'Dry-Rainfed', transform=ax5[1,0].transAxes + trans,#size = 20,
            fontsize=20, verticalalignment='top', weight = 'bold',
            bbox=dict(facecolor='white', edgecolor='white', pad=3.0))
ax5[1,1].text(0.46, 0.9, s = 'Normal-Rainfed', transform=ax5[1,1].transAxes + trans,#size = 20,
            fontsize=20, verticalalignment='top', weight = 'bold',
            bbox=dict(facecolor='white', edgecolor='white', pad=3.0))
ax5[1,2].text(0.57, 0.9, s = 'Wet-Rainfed', transform=ax5[1,2].transAxes + trans,#size = 20,
            fontsize=20, verticalalignment='top', weight = 'bold',
            bbox=dict(facecolor='white', edgecolor='white', pad=3.0))
#ax5[0,1].set_title('Normal-Irrigated', ha='center', va='center', size = 20)
#ax5[0,1].set_title('Wet-Irrigated', ha='center', va='center', size = 20)
#ax5[1,0].set_title('Dry-Rainfed', ha='center', va='center', size = 20)
#ax5[1,1].set_title('Normal-Rainfed', ha='center', va='center', size = 20)
#ax5[1,2].set_title('Wet-Rainfed', ha='center', va='center', size = 20)
#ax5[0,0].legend(loc='lower right', fontsize=12, frameon=False)
ax5[0,0].axis('off')
ax5[0,1].axis('off')
ax5[0,2].axis('off')
ax5[1,0].axis('off')
ax5[1,1].axis('off')
ax5[1,2].axis('off')
fig5.subplots_adjust(wspace=.09, hspace=0.09)
plt.savefig('results/visuals/morris_corn_yield.png',format='png')
#plt.savefig('results/visuals/morris_corn_yield.pdf',format='pdf', bbox_inches='tight')




# irrigation plots
fig6, ax6 = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
ax6[0].imshow(irrg2002[1].canvas.renderer.buffer_rgba())
ax6[1].imshow(irr2021[1].canvas.renderer.buffer_rgba())
ax6[2].imshow(irr2019[1].canvas.renderer.buffer_rgba())
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig6.dpi_scale_trans)
ax6[0].text(0.78, 0.9, s = 'Dry', transform=ax6[0].transAxes + trans,#size = 20,
            fontsize=16, verticalalignment='top', weight = 'bold',
            bbox=dict(facecolor='white', edgecolor='white', pad=3.0))
ax6[1].text(0.67, 0.9, s = 'Normal', transform=ax6[1].transAxes + trans,#size = 20,
            fontsize=16, verticalalignment='top', weight = 'bold',
            bbox=dict(facecolor='white', edgecolor='white', pad=3.0))
ax6[2].text(0.78, 0.9, s = 'Wet', transform=ax6[2].transAxes + trans,#size = 20,
            fontsize=16, verticalalignment='top', weight = 'bold',
            bbox=dict(facecolor='white', edgecolor='white', pad=3.0))
ax6[0].axis('off')
ax6[1].axis('off')
ax6[2].axis('off')
fig6.subplots_adjust(wspace=.09, hspace=0.09)
plt.savefig('results/visuals/morris_corn_irrigation.png',format='png')





######## final run with the root uptake values that are closer to the ones in aquacrop and N = 1000
# save results in pickle
# irrigated corn 
with open(r'./data/analysis_results/sensitivity_analysis_runs/ks_irrig_corn_morris_n1000.pickle', 'wb') as ci3: 
    pickle.dump(irrig_corn_morris, ci3) 

# rainfed corn
with open(r'./data/analysis_results/sensitivity_analysis_runs/ks_rainfed_corn_morris_N1000.pickle', 'wb') as cr3: 
    pickle.dump(rainfed_corn_morris, cr3) 




######## second run with the root uptake values that are closer to the ones in aquacrop
# save results in pickle
# irrigated corn 
with open(r'./data/analysis_results/sensitivity_analysis_runs/ks_irrig_corn_morris_r4.pickle', 'wb') as ci3: 
    pickle.dump(irrig_corn_morris, ci3) 

# rainfed corn
with open(r'./data/analysis_results/sensitivity_analysis_runs/ks_rainfed_corn_morris_r4.pickle', 'wb') as cr3: 
    pickle.dump(rainfed_corn_morris, cr3) 




#######******************************** IGNORE IGNORE IGNORE

######## second run with the root uptake values that are closer to the ones in aquacrop
# save results in pickle
# irrigated corn 
with open(r'./data/analysis_results/sensitivity_analysis_runs/ks_irrig_corn_morris_r3.pickle', 'wb') as ci2: 
    pickle.dump(irrig_corn_morris, ci2) 

# rainfed corn
with open(r'./data/analysis_results/sensitivity_analysis_runs/ks_rainfed_corn_morris_r3.pickle', 'wb') as cr2: 
    pickle.dump(rainfed_corn_morris, cr2) 



# save results in pickle
# irrigated corn 
#with open(r'./data/analysis_results/sensitivity_analysis_runs/ks_irrig_corn_morris.pickle', 'wb') as ci: 
   # pickle.dump(irrig_corn_morris, ci) 

# rainfed corn
#with open(r'./data/analysis_results/sensitivity_analysis_runs/ks_rainfed_corn_morris.pickle', 'wb') as cr: 
   # pickle.dump(rainfed_corn_morris, cr) 

def morris_plots(df):
    
    #yield
    yield_df = df[['Yield (tonne/ha)']] # select yield variable
    yield_vals = yield_df.to_numpy() # transform yield to numpy array

    # Perform the sensitivity analysis using the model output
    # Specify which column of the output file to analyze (zero-indexed)
    Si_yield_ms = morris.analyze(
        corn_params,
        param_values_sample,
        yield_vals,
        conf_level=0.95,
        print_to_console=True,
         num_levels=10,
         num_resamples=100,
         )
    
    #Si_yield.sort_values(by=['mu_star'], inplace = True)
    Si_yield = pd.DataFrame(Si_yield_ms)
    
    Si_yield_t5 = Si_yield.nlargest(5, ['mu_star'])
    
    #horizontal bar
    ##fig1, (ax1) = plt.subplots()
    ##plt.barh(y=Si_yield.names, width=Si_yield.mu_star, edgecolor='black', color = 'lightgrey')
    #plt.errorbar(y=Si_yield.names, width=Si_yield.mu_star, yerr=0.5, fmt="o", color="r")
    ##plt.xlabel('μ∗ [mm]')
    #horizontal_bar_plot(ax1, Si_yield, sortby="mu_star", unit=r"yield (t/ha)")
    #covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
    ##fig1.tight_layout(pad=1.2)
    #fig.savefig('results/visuals/sheridan_corn_yield_sa_irrig_2019_n100.png', format='png', dpi=1000, orientation = 'landscape')
    
    fig2, ax2 = plt.subplots()
    ax2.scatter(Si_yield['mu_star'],Si_yield['sigma'], s=40, c = "lightgrey", edgecolors = 'black')
    plt.xlabel('μ ∗ [t/ha]', fontsize=15)
    plt.ylabel('σ [t/ha]', fontsize=15)
    #plt.xticks(fontsize=12)
    #plt.yticks(fontsize=12)
    for j, txt in Si_yield_t5.iterrows():
        ax2.annotate(txt['names'], (txt['mu_star'], txt['sigma']), size = 10,
                     textcoords='offset points',  xytext=(1,6), ha='center', color = 'black')
    

    # irrigation
    irrig_df = df[['Seasonal irrigation (mm)']] # select yield variable
    irrig_vals = irrig_df.to_numpy() # transform yield to numpy array

    Si_irrig_ms = morris.analyze(
        corn_params,
        param_values_sample,
        irrig_vals,
        conf_level=0.95,
        print_to_console=True,
         num_levels=10,
         num_resamples=100,
         )
    
    Si_irrig = pd.DataFrame(Si_irrig_ms)
    
    Si_irrig_t5 = Si_irrig.nlargest(5, ['mu_star'])

    #horizontal bar
    ##fig3, (ax3) = plt.subplots()
    ##plt.barh(y=Si_irrig.names, width=Si_irrig.mu_star, edgecolor='black', color = 'lightgrey')
    #plt.errorbar(y=Si_yield.names, width=Si_yield.mu_star, yerr=0.5, fmt="o", color="r")
    ##plt.xlabel('μ∗ [mm]')
    #horizontal_bar_plot(ax1, Si_yield, sortby="mu_star", unit=r"yield (t/ha)")
    #covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
    ##fig3.tight_layout(pad=1.2)


    fig4, ax4 = plt.subplots()
    ax4.scatter(Si_irrig['mu_star'],Si_irrig['sigma'], s=40, c = "lightgrey", edgecolors = 'black')
    plt.xlabel('μ ∗ [mm]', fontsize=15)
    plt.ylabel('σ [mm]', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    for j, txt in Si_irrig_t5.iterrows():
        ax4.annotate(txt['names'], (txt['mu_star'], txt['sigma']), size = 10,
                     textcoords='offset points',  xytext=(8,6), ha='center', color = 'black')
        
    
    return fig2, fig4 # only return scatterplots


# make graphs 
def morris_plots(df):
    
    #yield
    yield_df = df[['Yield (tonne/ha)']] # select yield variable
    yield_vals = yield_df.to_numpy() # transform yield to numpy array

    # Perform the sensitivity analysis using the model output
    # Specify which column of the output file to analyze (zero-indexed)
    Si_yield = morris.analyze(
        corn_params,
        param_values_sample,
        yield_vals,
        conf_level=0.95,
        print_to_console=True,
         num_levels=10,
         num_resamples=100,
         )

    #horizontal bar
    fig1, (ax1) = plt.subplots()
    horizontal_bar_plot(ax1, Si_yield, {}, sortby="mu_star", unit=r"yield (t/ha)")
    #covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
    fig.tight_layout(pad=1.0)
    #fig.savefig('results/visuals/sheridan_corn_yield_sa_irrig_2019_n100.png', format='png', dpi=1000, orientation = 'landscape')
    
    # scatterplot
    fig2, ax2 = plt.subplots()
    ax2.scatter(Si_yield['mu_star'],Si_yield['sigma'], s=20, c = "white", edgecolors = 'black')
    plt.xlabel('mu_star')
    plt.ylabel('sigma')
    for j, txt in enumerate(Si_yield['names']):
        ax2.annotate(txt, (Si_yield['mu_star'][j], Si_yield['sigma'][j]))

    
    # irrigation
    irrig_df = df[['Seasonal irrigation (mm)']] # select yield variable
    irrig_vals = irrig_df.to_numpy() # transform yield to numpy array
    
    Si_irrig = morris.analyze(
     corn_params,
     param_values_sample,
     irrig_vals,
     conf_level=0.95,
     print_to_console=True,
     num_levels=10,
     num_resamples=100,
     )
    
    
    fig3, (ax3) = plt.subplots()
    horizontal_bar_plot(ax3, Si_irrig, {}, sortby="mu_star", unit=r"irrigation (mm)")
    fig.tight_layout(pad=1.0)
    
    fig4, ax4 = plt.subplots()
    ax4.scatter(Si_irrig['mu_star'],Si_irrig['sigma'], s=20, c = "white", edgecolors = 'black')
    plt.xlabel('mu_star')
    plt.ylabel('sigma')
    for j, txt in enumerate(Si_irrig['names']):
        ax4.annotate(txt, (Si_irrig['mu_star'][j], Si_irrig['sigma'][j]))


    return(fig1, fig2, fig3, fig4) # return yield and irrig


list(map(morris_plots, [rainfed_corn_lst[0][0]]))


###########################3

# make individual graphs 
def morris_plots(df):
    
    #yield
    yield_df = df[['Yield (tonne/ha)']] # select yield variable
    yield_vals = yield_df.to_numpy() # transform yield to numpy array

    # Perform the sensitivity analysis using the model output
    # Specify which column of the output file to analyze (zero-indexed)
    Si_yield = morris.analyze(
        corn_params,
        param_values_sample,
        yield_vals,
        conf_level=0.95,
        print_to_console=True,
         num_levels=10,
         num_resamples=100,
         )
    

    #horizontal bar
    fig1, (ax1) = plt.subplots()
    horizontal_bar_plot(ax1, Si_yield, {}, sortby="mu_star", unit=r"yield (t/ha)")
    #covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
    fig.tight_layout(pad=1.0)
    #fig.savefig('results/visuals/sheridan_corn_yield_sa_irrig_2019_n100.png', format='png', dpi=1000, orientation = 'landscape')
    
    # scatterplot
    fig2, ax2 = plt.subplots()
    ax2.scatter(Si_yield['mu_star'],Si_yield['sigma'], s=40, c = "white", edgecolors = 'black')
    plt.xlabel('mu_star')
    plt.ylabel('sigma')
    for j, txt in enumerate(Si_yield['names']):
        ax2.annotate(txt[0:5], (Si_yield['mu_star'][j], Si_yield['sigma'][j]))
   
    # irrigation
    irrig_df = df[['Seasonal irrigation (mm)']] # select yield variable
    irrig_vals = irrig_df.to_numpy() # transform yield to numpy array
    
    Si_irrig = morris.analyze(
     corn_params,
     param_values_sample,
     irrig_vals,
     conf_level=0.95,
     print_to_console=True,
     num_levels=10,
     num_resamples=100,
     )
    
    
    fig3, (ax3) = plt.subplots()
    horizontal_bar_plot(ax3, Si_irrig, {}, sortby="mu_star", unit=r"irrigation (mm)")
    fig.tight_layout(pad=1.0)
    
    fig4, ax4 = plt.subplots()
    ax4.scatter(Si_irrig['mu_star'],Si_irrig['sigma'], s=20, c = "white", edgecolors = 'black')
    plt.xlabel('mu_star')
    plt.ylabel('sigma')
    for j, txt in enumerate(Si_irrig['names']):
        ax4.annotate(txt, (Si_irrig['mu_star'][j], Si_irrig['sigma'][j]))


    return(fig1, fig2, fig3, fig4) # return yield and irrig





list(map(morris_plots, rainfed_corn_lst))


with open(wd+'/data/analysis_results/sensitivity_analysis_runs/ks_irrig_corn_morris_higherRt_vals.pickle', 'rb') as cr: 
    irrig_corn_lst = pickle.load(cr)















df = influ_morris_irrig[1][0]

yield_df = df[['Yield (tonne/ha)']] # select yield variable
yield_vals = yield_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si_yield = morris.analyze(
    corn_params,
    param_values_sample,
    yield_vals,
    conf_level=0.95,
    print_to_console=True,
     num_levels=10,
     num_resamples=100,
     )

#Si_yield.sort_values(by=['mu_star'], inplace = True)

Si_yield_t5 = Si_yield.nlargest(10, ['mu_star'])
#horizontal bar
fig1, (ax1) = plt.subplots()
plt.barh(y=Si_yield.names, width=Si_yield.mu_star, edgecolor='black', color = 'lightgrey')
#plt.errorbar(y=Si_yield.names, width=Si_yield.mu_star, yerr=0.5, fmt="o", color="r")
plt.xlabel('mu_star')
#horizontal_bar_plot(ax1, Si_yield, sortby="mu_star", unit=r"yield (t/ha)")
#covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
fig1.tight_layout(pad=1.2)
#fig.savefig('results/visuals/sheridan_corn_yield_sa_irrig_2019_n100.png', format='png', dpi=1000, orientation = 'landscape')

fig2, ax2 = plt.subplots()
ax2.scatter(Si_yield['mu_star'],Si_yield['sigma'], s=40, c = "lightgrey", edgecolors = 'black')
plt.xlabel('μ∗ [t/ha]', fontsize=15)
plt.ylabel('σ [t/ha]', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
for j, txt in Si_yield_t5.iterrows():
    ax2.annotate(txt['names'], (txt['mu_star'], txt['sigma']), size = 10,
                 textcoords='offset points',  xytext=(0,6), ha='center', color = 'black')



# extract the required data
trajectories = range(10, 110, 10)  # number of trajectories
mean_EEs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # mean EEs for each parameter
std_EEs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]  # standard deviation of EEs for each parameter

# create the plot
plt.errorbar(trajectories, mean_EEs, yerr=std_EEs, fmt='-o')
plt.xlabel('Number of trajectories')
plt.ylabel('Mean elementary effect')
plt.title('Convergence of Morris screening analysis')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from SALib.sample.morris import morris
from SALib.analyze.morris import analyze

# define the problem
problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-1, 1], [-1, 1], [-1, 1]]
}

# generate the samples
n = 1000  # number of samples
n = 1000  # number of samples
sample = morris.sample(problem, n, num_levels=4, optimal_trajectories=None, 
                       local_optimization=False, seed=123)

# run the model and calculate the EEs
Y = np.zeros((n, 1))  # shape correction
for i, X in enumerate(sample):
    Y[i, 0] = (X**2).sum()  # shape correction
ee, _, _ = analyze(problem, sample, Y, conf_level=0.95, print_to_console=False)

# extract the required data for plotting
trajectories = np.arange(2, len(ee['mu_star'])+1)
mean_EEs = ee['mu_star'][1:]
std_EEs = ee['mu_star_conf'][1:]

# create the plot
plt.errorbar(trajectories, mean_EEs, yerr=std_EEs, fmt='-o')
plt.xlabel('Number of trajectories')
plt.ylabel('Mean elementary effect')
plt.title('Convergence of Morris screening analysis')
plt.show()







