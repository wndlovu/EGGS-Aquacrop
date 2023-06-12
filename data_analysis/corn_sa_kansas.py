#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:25:00 2023

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


param_names = param_names['name'].values.tolist() # get the column names



param_values_sample = sample(corn_params, N=100, num_levels=10, optimal_trajectories=None) # 100,6,5generate array with param values
param_values = pd.DataFrame(param_values_sample) #transform array to df
param_values.columns = param_names # add column names


# sheridan gridMET 
sher_gridMET = gridMET_county[10]
sher_gridMET = sher_gridMET.assign(year = sher_gridMET['Date'].dt.year) # create year variable

# get wettest and driest years
wet_dry = sher_gridMET.groupby(['year'])[['Precipitation']].sum() # dry = 2002, LEMA year 2012 and wet = 2019
wet_dry = wet_dry.reset_index()
median = wet_dry['Precipitation'].median()
normal = wet_dry[wet_dry['Precipitation']== median] # uneven num of year , but normal year is either 2001 or 2021, so use 2001


sher_gridMET = sher_gridMET[sher_gridMET['year'] == 2021] # filter for 2012
sher_gridMET = sher_gridMET.drop(['year'], axis=1) # drop year variable
wdf = sher_gridMET

#wdf['MinTemp'].min()

# sheridan soils
sher_soils = custom_soil[10]
custom = sher_soils

sim_start = '2021/01/01' #dates to match crop data
sim_end = '2021/12/31'
initWC = InitialWaterContent(value=['FC'])
#irr_mngt = IrrigationManagement(irrigation_method=1,SMT=[80]*4)

# Morris method
model_irrig = []
#while True:
#model_irrig2 = pd.DataFrame([])    
for i in range(0, len(param_values)):
    #pre_LEMA_stats = pd.DataFrame()
    #LEMA_stats = pd.DataFrame()
        irr_mngt = IrrigationManagement(irrigation_method=1,
                                   SMT=[param_values['smt'][i]]*4)
                                    
        #irr_mngt = IrrigationManagement(irrigation_method=0) # rainfed
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
        

model_df_full = pd.concat(model_irrig) # model results from simul

model_df_full.to_csv(r'./data/analysis_results/SherCorn_SAModel_Irrig_2021_n100.csv', sep=',', encoding='utf-8', header='true') # 100 trajectories, 4 optimal



## Morris Screening

# read in data sets
corn_irrig2002 = pd.read_csv(wd+"/data/analysis_results/SherCorn_SAModel_Irrig_2002_n100.csv")
corn_irrig2012 = pd.read_csv(wd+"/data/analysis_results/SherCorn_SAModel_Irrig_2012_n100.csv")
corn_rain2019 = pd.read_csv(wd+"/data/analysis_results/SherCorn_SAModel_Rainfed_2019_n100.csv")
corn_irrig2019 = pd.read_csv(wd+"/data/analysis_results/SherCorn_SAModel_Irrig_2019_n100.csv")
corn_rain2021 = pd.read_csv(wd+"/data/analysis_results/SherCorn_SAModel_Rainfed_2021_n100.csv")
corn_irrig2021= pd.read_csv(wd+"/data/analysis_results/SherCorn_SAModel_Irrig_2021_n100.csv")

corn_sa_results = [corn_irrig2012, corn_rain2019, corn_irrig2019, corn_rain2021, corn_irrig2021]

# yield screening
def yield_screen(x):
    yield_df = x[['Yield (tonne/ha)']] # select yield variable
    yield_vals = yield_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
    Si = morris.analyze(
     corn_params,
     param_values_sample,
     yield_vals,
     conf_level=0.95,
     print_to_console=True,
     num_levels=10,
     num_resamples=100,
     )


    Si_df = pd.DataFrame(Si)
    Si_df = Si_df[Si_df['mu_star'] > 0.3]
    
    return(Si_df)


#The thresholds used for crop yield and total transpiration were 0.25 t · ha and 10 mm

# create df with the influential params for all combinations

influ_params = []

for i in corn_sa_results:
    

yield_irrig2002_influ = yield_scrn(corn_irrig2002) 
yield_irrig2002_influ = yield_irrig2002_influ[yield_irrig2002_influ['mu_star'] > 0.25]

yield_irrig2019_influ = yield_scrn(corn_irrig2019) 
yield_irrig2019_influ = yield_irrig2019_influ[yield_irrig2019_influ['mu_star'] > 0.3]

yield_irrig2021_influ = yield_scrn(corn_irrig2021) 
yield_irrig2021_influ = yield_irrig2021_influ[yield_irrig2021_influ['mu_star'] > 0.3]


yield_rain2019_influ = yield_scrn(corn_rain2019) 
yield_rain2019_influ = yield_rain2019_influ[yield_rain2019_influ['mu_star'] > 0.3]

yield_rain2021_influ = yield_scrn(corn_rain2021) 
yield_rain2021_influ = yield_rain2021_influ[yield_rain2021_influ['mu_star'] > 0.3]



# the mean of the absolute values of the elementary effects (µ∗

#Large µ∗ values indicate higher influ220 ence on the model output and 
#large σ values indicate more interactions with
# other parameters or the non-linear model response.

# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# e.g. Si['mu_star'] contains the mu* value for each parameter, in the
# same order as the parameter file
yield_df = corn_irrig2002[['Yield (tonne/ha)']] # select yield variable
yield_vals = yield_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = morris.analyze(
    corn_params,
    param_values_sample,
    yield_vals,
    conf_level=0.95,
    print_to_console=True,
     num_levels=10,
     num_resamples=100,
     )


fig, (ax1, ax2) = plt.subplots(1, 2)
horizontal_bar_plot(ax1, Si, {}, sortby="mu_star", unit=r"yield(t/ha)/year")
covariance_plot(ax2, Si, {}, unit=r"yield(t/ha)/year")
fig.tight_layout(pad=1.0)
#fig.savefig('results/visuals/sheridan_corn_yield_sa_irrig_2019_n100.png', format='png', dpi=1000, orientation = 'landscape')


fig2 = plt.figure()
sample_histograms(fig2, param_values_sample, corn_params, {"color": "y"})
plt.show()

# 3rd plot
fig, ax = plt.subplots()
ax.scatter(Si['mu_star'],Si['sigma'], s=20, c = "white", edgecolors = 'black')
#ax.plot(Si['mu_star'],2*Si['sigma']/np.sqrt(number_of_trajectories),'--',alpha=0.5)
#ax.plot(np.array([0,Si['mu_star'][0]]),2*np.array([0,Si['sigma'][0]/np.sqrt(number_of_trajectories)]),'--',alpha=0.5)

#plt.title('Distribution of Elementary effects')
plt.xlabel('mu_star')
plt.ylabel('sigma')
for i, txt in enumerate(Si['names']):
    ax.annotate(txt, (Si['mu_star'][i], Si['sigma'][i]))
fig.savefig('results/visuals/sheridan_corn_yield_irrig_sa2_2019_n100.png', format='png', dpi=1000, orientation = 'landscape')


###############################################
fig, ax = plt.subplots()
ax.scatter(Si['mu_star'],Si['sigma'], s=20, c = "white", edgecolors = 'black')
#ax.plot(Si['mu_star'],2*Si['sigma']/np.sqrt(number_of_trajectories),'--',alpha=0.5)
#ax.plot(np.array([0,Si['mu_star'][0]]),2*np.array([0,Si['sigma'][0]/np.sqrt(number_of_trajectories)]),'--',alpha=0.5)

#plt.title('Distribution of Elementary effects')
plt.xlabel('mu_star')
plt.ylabel('sigma')
for i, txt in enumerate(Si['names']):
    ax.annotate(txt, (Si['mu_star'][i], Si['sigma'][i]))

Si_df2 = Si_df[Si_df['mu_star'] > 0.25]


fig, ax = plt.subplots()
ax.scatter(Si_df['mu_star'],Si_df['sigma'], s=20, c = "white", edgecolors = 'black')
#ax.plot(Si['mu_star'],2*Si['sigma']/np.sqrt(number_of_trajectories),'--',alpha=0.5)
#ax.plot(np.array([0,Si['mu_star'][0]]),2*np.array([0,Si['sigma'][0]/np.sqrt(number_of_trajectories)]),'--',alpha=0.5)

#plt.title('Distribution of Elementary effects')

#fig, ax = plt.subplots()
#plt.errorbar(x=mu_star, y=sigma,  fmt="o", ecolor = "black", color = "white",
            # markeredgecolor = 'black', markersize = 5)
plt.xlabel('mu_star')
plt.ylabel('sigma')

for i in range(0, len(Si_df2)):
    u = Si_df2['names'][i]
    print(u)
    #for i in range(len(mu_star)): 
    #muStar = [i for i in Si_df['mu_star'] if i > 0.25]
    ax.annotate(txt, (Si_df2['mu_star'][i], Si_df2['sigma'][i]))





for i, txt in enumerate(prm):
    if mu
    ax.annotate(txt, (mu_star[i], sigma[i]))


# fix this

mu_star = np.array(Si_df[['mu_star']])
sigma = np.array(Si_df[['sigma']])
prm = Si_df["names"].values.tolist()



plt.errorbar(x=mu_star, y=sigma,   fmt="o", ecolor = "black", color = "white",
             markeredgecolor = 'black', markersize = 10)
plt.xlabel('mu_star')
plt.ylabel('sigma')

#plt.scatter(x, y)
  
# Loop for annotation of all points
for i in range(len(mu_star)):
    for i, txt in enumerate(prm):
        muStar = [i for i in mu_star if i > 0.25]
        plt.annotate(txt, (sigma[i], muStar[i]+ 0.2))
  
plt.show()


############################################################
def irrig_scrn(x):

# irrig screening
    irrig_df = x[['Seasonal irrigation (mm)']] # select yield variable
    irrig_vals = irrig_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
    Si = morris.analyze(
     corn_params,
     param_values_sample,
     irrig_vals,
     conf_level=0.95,
     print_to_console=True,
     num_levels=10,
     num_resamples=100,
     )

  
    Si_df = pd.DataFrame(Si)
     
    return(Si_df)
 
    
irrig2002_influ = irrig_scrn(corn_irrig2002) 
irrig2002_influ = irrig2002_influ[irrig2002_influ['mu_star'] > 20]
 
irrig2019_influ = irrig_scrn(corn_irrig2019) 
irrig2019_influ = irrig2019_influ[irrig2019_influ['mu_star'] > 20]


irrig2021_influ = irrig_scrn(corn_irrig2021) 
irrig2021_influ = irrig2021_influ[irrig2021_influ['mu_star'] > 20]


   
#The thresholds used for crop yield and total transpiration were 0.25 t · ha and 10 mm


# the mean of the absolute values of the elementary effects (µ∗

#Large µ∗ values indicate higher influ220 ence on the model output and 
#large σ values indicate more interactions with
# other parameters or the non-linear model response.

# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# e.g. Si['mu_star'] contains the mu* value for each parameter, in the
# same order as the parameter file

irrig_df = model_df_full[['Seasonal irrigation (mm)']] # select yield variable
irrig_vals = irrig_df.to_numpy() # transform yield to numpy array

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = morris.analyze(
     corn_params,
     param_values_sample,
     irrig_vals,
     conf_level=0.95,
     print_to_console=True,
     num_levels=10,
     num_resamples=100,
     )


fig, (ax1, ax2) = plt.subplots(1, 2)
horizontal_bar_plot(ax1, Si, {}, sortby="mu_star", unit=r"irrigation (mm)")
covariance_plot(ax2, Si, {}, unit=r"irrigation (mm)")
fig.tight_layout(pad=1.0)
fig.savefig('results/visuals/sheridan_corn_irrig_sa_v2_2021.png', format='png', dpi=1000, orientation = 'landscape')


fig2 = plt.figure()
sample_histograms(fig2, param_values_sample, corn_params, {"color": "y"})
plt.show()

# 3rd plot
fig, ax = plt.subplots()
ax.scatter(Si['mu_star'],Si['sigma'], s=20, c = "white", edgecolors = 'black')
#ax.plot(Si['mu_star'],2*Si['sigma']/np.sqrt(number_of_trajectories),'--',alpha=0.5)
#ax.plot(np.array([0,Si['mu_star'][0]]),2*np.array([0,Si['sigma'][0]/np.sqrt(number_of_trajectories)]),'--',alpha=0.5)

#plt.title('Distribution of Elementary effects')
plt.xlabel('mu_star')
plt.ylabel('sigma')
for i, txt in enumerate(Si['names']):
    ax.annotate(txt, (Si['mu_star'][i], Si['sigma'][i]))
fig.savefig('results/visuals/sheridan_corn_irrig_sa2_v2_2002.png', format='png', dpi=1000, orientation = 'landscape')


########## Sobol
























