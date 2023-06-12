# -*- coding: utf-8 -*-

import sys

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




wd=getcwd() # set working directory
chdir(wd)


soils_df_full = pd.read_csv(wd + '/data/agricLand/soils/Soil_FieldsAroundSD6KS_POLARIS_AGrinstead_20220706.csv')
soils_df = soils_df_full[soils_df_full['UID'].isin(s_list)] # filter for one site
soils_df = soils_df[soils_df['depth_cm'] == '0-5']


soils = pd.DataFrame(soils_df_full)
soils = soils[soils['depth_cm'] == '0-5'] # use upper 0.5cm
soils = soils.assign(om = (10**(soils['logOm_%'])),
                     Ksat_cmHr = (10**(soils['logKsat_cmHr'])))
soils = soils[['UID', 'depth_cm', 'silt_prc', 'sand_prc',
               'clay_prc', 'thetaS_m3m3', 'thetaR_m3m3',
               'Ksat_cmHr', 'lambda', 'logHB_kPa', 'n',
               'logAlpha_kPa1', 'om']]


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


# test to see if the dictionaries are working
#print(list(soil_dict.keys())[2])   
#print(list(soil_dict.values())[2])

# filter for dictionary with 1381151 test site
test_site = {k: v for k, v in soil_dict.items() if k == 1381151}  # filter for given site number
test_site = list(test_site.values())



## run model params
path = get_filepath(wd + '/data/hydrometeorology/gridMET/gridMET_1381151.txt') #replace folder name from folder name with file path
wdf = prepare_weather(path)
sim_start = '2000/01/01' #dates to match crop data
sim_end = '2020/12/31'
custom = test_site[0] # use custom layer for 1 site
#crop = Crop('Maize', planting_date='05/01') 
initWC = InitialWaterContent(value=['FC'])
irr_mngt = IrrigationManagement(irrigation_method=1,SMT=[80]*4)
#irr_mngt = IrrigationManagement(irrigation_method = 0) # no irrigation



# run model ignore for now
for i in range(0, len(problem)):
    crop = Crop("Maize", Tbase = tb[i], Tupp = tu[i], SeedSize = ccs[i], PlantPop = den[i])
    
model = AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC, irr_mngt)
model.run_model(till_termination=True) # run model till the end
model_df_et = model._outputs.water_flux
model_df_irr = model._outputs.final_stats
model_df_water_storage = model._outputs.water_storage
model_df_crp_grwth = model._outputs.crop_growth


# Morris method

# Define the model inputs
# Define the model inputs

problem = {
    'num_vars': 27,
    'names': ['To_crop',  # crop development
              'Tmax_crop', 
              'ccs',  
              'eme', 
              'cgc', 
              'ccx', 
              'sen', 
              'cdc', 
              'mat', 
              'flolen',
              'rtm', # removed - ZeroDivisionError: division by zero
              'rtx',
              'rtshp',
              'root',
              'rtexup',
              'rtexlw',
              # Crop Transpiration
              'kc',
              'kcdcl'
              # Biomass and Yield
              'wp',
              'wpy',
              'hi',
              'hipsflo',
              'exc',
              'hipsveg',
              'hingsto',
              'hinc',
              'hilen'],
    'groups': None,
    'bounds': [#Crop Development
               [-1.0, -0.5],  # bounds for winter corn (look for corn bounds)
               [21.0, 26.0],
               [1.0, 2.0],
               [90.0, 230.0],
               [0.004, 0.008],
               [0.75,	1],
               [1090.0, 2250.0],
               [0.006, 0.012],
               [1590.0, 3150.0],
               [100, 300],
               [0.15,	0.3], ############
               [1.0, 2.4],
               [20.0, 40.0],
               [1200.0, 2011],
               [0.02, 0.03],
               [0.001, 0.05],
               #Crop Transpiration
               [1.05, 1.15],
               [0.2, 0.4],
               #Biomass and Yield
               [15.0, 20.0],
               [75,	125],
               [40,	55],
               [3, 5],
               [25, 50],
               [4, 6],
               [4.1, 7],
               [20, 30],
               [320, 880]]
}


param_values = sample(problem, 
                      N=100, 
                      num_levels=27, 
                      optimal_trajectories=20)
param_values2 = pd.DataFrame(param_values)
param_values2.columns = [#Crop Development
                         'To_crop', 
                         'Tmax_crop', 
                         'ccs', 
                         'eme', 
                         'cgc', 
                         'ccx', 
                         'sen', 
                         'cdc', 
                         'mat',
                         'flolen',
                         'rtm', ##########
                         'rtx',
                         'rtshp',
                         'root',
                         'rtexup',
                         'rtexlw',
                         #Crop Transpiration
                         'kc',
                         'kcdcl',
                         # Biomass and Yield
                         'wp',
                         'wpy',
                         'hi',
                         'hipsflo',
                         'exc',
                         'hipsveg',
                         'hingsto',
                         'hinc',
                         'hilen']
#param_values2.den = param_values2.den.astype(int)# convert to int

# matrix rows = (num params + 1) * optimal_trajectories or N if optimal_traj not specified

#model_list = [] #Y
#model_df_full = pd.DataFrame()
model_pre_LEMA = pd.DataFrame()
model_LEMA = pd.DataFrame()
for i in range(0, len(param_values2)):
    #pre_LEMA_stats = pd.DataFrame()
    #LEMA_stats = pd.DataFrame()
    crop = Crop("Maize", planting_date='05/01', 
                ##Crop Development
                Tbase = param_values2['To_crop'][i], 
                Tupp = param_values2['Tmax_crop'][i], 
                SeedSize = param_values2['ccs'][i], 
                Emergence = param_values2['eme'][i],
                CGC = param_values2['cgc'][i], 
                CCx = param_values2['ccx'][i], 
                Senescence = param_values2['sen'][i],
                CDC = param_values2['cdc'][i], 
                Maturity = param_values2['mat'][i],
                Zmin = param_values2['rtm'][i],
                Flowering = param_values2['flolen'][i],
                Zmax = param_values2['rtx'][i],
                fshape_r = param_values2['rtshp'][i],
                MaxRooting = param_values2['root'][i],
                SxTopQ = param_values2['rtexup'][i],
                SxBotQ = param_values2['rtexlw'][i],
                #Crop Transpiration
                Kcb = param_values2['kc'][i],
                fage = param_values2['kcdcl'][i],
                #Biomass and Yield
                WP = param_values2['wp'][i],
                WPy = param_values2['wpy'][i],
                HI0 = param_values2['hi'][i],
                dHI_pre = param_values2['hipsflo'][i],
                exc = param_values2['exc'][i],
                a_HI = param_values2['hipsveg'][i],
                b_HI = param_values2['hingsto'][i],
                dHI0 = param_values2['hinc'][i],
                YldForm = param_values2['hilen'][i]) 
    model = AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC, irr_mngt)
    model.run_model(till_termination=True) # run model till the end
           # model_df_et = model._outputs.water_flux
    model_df = model._outputs.final_stats
    model_df = model_df.reset_index()
    #model_list.append(model_df)
    model_df = model_df.assign(year = model_df['Harvest Date (YYYY/MM/DD)'].dt.year)
    pre_LEMA = model_df[model_df['year'] < 2013] # filter for preLEMA year
    #pre_LEMA = pre_LEMA.assign(mean_yield = pre_LEMA['Yield (tonne/ha)'].mean(), # calculate ave yield
                                     #mean_irrig = pre_LEMA['Seasonal irrigation (mm)'].mean())
    pre_LEMA = pre_LEMA.groupby(['crop Type'])['Yield (tonne/ha)', 'Seasonal irrigation (mm)'].mean()
    #pre_LEMA = pre_LEMA.[['Yield (tonne/ha)', 'Seasonal irrigation (mm)']].mean()
    LEMA = model_df[model_df['year'] > 2012]
    #LEMA_stats = LEMA_stats.assign(mean_yield = LEMA['Yield (tonne/ha)'].mean(), # calculate ave yield
                                     #mean_irrig = LEMA['Seasonal irrigation (mm)'].mean())
    LEMA = LEMA.groupby(['crop Type'])['Yield (tonne/ha)', 'Seasonal irrigation (mm)'].mean()                                 
    model_pre_LEMA = model_pre_LEMA.append(pre_LEMA)
    model_LEMA = model_LEMA.append(LEMA)
    #model_df_full = model_df_full.append(model_df)

        
# check to see if yield values are different
y1 = model_list[0]
y2 = model_list[6]   




# yield analysis - approach 1 
yield_results_pre_LEMA = model_pre_LEMA[['Yield (tonne/ha)']]
yield_results_pre_LEMA =  yield_results_pre_LEMA.to_numpy()

yield_results_LEMA = model_LEMA[['Yield (tonne/ha)']]
yield_results_LEMA =  yield_results_LEMA.to_numpy() # convert p to array of float




# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = morris.analyze(
    problem,
    param_values,
    yield_results_LEMA,
    conf_level=0.95,
    print_to_console=True,
    num_levels=16,
    num_resamples=100,
)



# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# e.g. Si['mu_star'] contains the mu* value for each parameter, in the
# same order as the parameter file

fig, (ax1, ax2) = plt.subplots(1, 2)
horizontal_bar_plot(ax1, Si, {}, sortby="mu_star", unit=r"yield(t/ha)/year")
covariance_plot(ax2, Si, {}, unit=r"yield(t/ha)/year")

fig2 = plt.figure()
sample_histograms(fig2, param_values, problem, {"color": "y"})
plt.show()







############## APPROACH 2
# yield analysis - approach 2
# approach 2 use original data - different len 
model_df_full = model_df_full.assign(year = model_df_full['Harvest Date (YYYY/MM/DD)'].dt.year)
pre_LEMA2 = model_df_full[model_df['year'] < 2013] 
LEMA2 = model_df_full[model_df['year'] > 2012]


yield_results_pre_LEMA2 = pre_LEMA2[['Yield (tonne/ha)']]
yield_results_pre_LEMA2 =  yield_results_pre_LEMA2.to_numpy()

yield_results_LEMA2 = LEMA2[['Yield (tonne/ha)']]
yield_results_LEMA2 =  yield_results_LEMA2.to_numpy() # convert p to array of float





############################

problem = {
 'num_vars': 8,
 'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
  'groups': None,
 'bounds': [[0.0, 1.0],
             [0.0, 1.0],
            [0.0, 1.0],
             [0.0, 1.0],
             [0.0, 1.0],
             [0.0, 1.0],
             [0.0, 1.0],
             [0.0, 1.0]]
 }

# Generate samples
param_values = sample(problem, N=3, num_levels=4, optimal_trajectories=None)

# To use optimized trajectories (brute force method),
# give an integer value for optimal_trajectories

# Run the "model" -- this will happen offline for external models
Y = Sobol_G.evaluate(param_values)













# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = morris.analyze(
    problem,
    param_values,
    Y,
    conf_level=0.95,
    print_to_console=True,
    num_levels=4,
    num_resamples=100,
)
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# e.g. Si['mu_star'] contains the mu* value for each parameter, in the
# same order as the parameter file

fig, (ax1, ax2) = plt.subplots(1, 2)
horizontal_bar_plot(ax1, Si, {}, sortby="mu_star", unit=r"tCO$_2$/year")
covariance_plot(ax2, Si, {}, unit=r"tCO$_2$/year")

fig2 = plt.figure()
sample_histograms(fig2, param_values, problem, {"color": "y"})
plt.show()



###################
import matplotlib.dates as mdates
import datetime
wd=getcwd() # set working directory
chdir(wd)

# read in df
et_means_test = pd.read_csv(wd + '/data/analysis_results/et_df_test.csv', index_col=0)
yield_df_test = pd.read_csv(wd + '/data/analysis_results/yield_df_test.csv', index_col=0)
irrig_df_test = pd.read_csv(wd + '/data/analysis_results/irrig_df_test.csv', index_col =0)

obs_data = yield_df_test['Yield (tonne/ha)']

k = Project(typ='R2')

simu_ensemble = np.zeros((len(obs_data),len(sample)))
for ii in range(0, len(sample)):
    with io.capture_output() as captured:          # suppress inline output from ResIPy
        # creating mesh
        k.createMesh(res0=10**sample[ii,0])   # need to use more effective method, no need to create mesh every time

        # add region
        k.addRegion(np.array([[2,-0.3],[2,-2],[3,-2],[3,-0.3],[2,-0.3]]), 10**sample[ii,1])
        k.addRegion(np.array([[5,-2],[5,-3.5],[8,-3.5],[8,-2],[5,-2]]), 10**sample[ii,2])

        # forward modelling
        k.forward(noise=0.025, iplot = False)
        out_data = np.loadtxt(os.path.join(fwd_dir, 'R2_forward.dat'),skiprows =1)
        simu_ensemble[:,ii] = out_data[:,6]
    print("Running sample",ii+1)
    
    
    
    
    
    
    