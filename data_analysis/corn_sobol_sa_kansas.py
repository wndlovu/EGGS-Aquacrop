#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:32:56 2023

@author: wayne
"""

from SALib.analyze import sobol
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop import Crop
# Read the parameter range file and generate samples
problem = read_param_file("../../src/SALib/test_functions/params/Ishigami.txt")

# Generate samples
param_values = saltelli.sample(problem, 1024, calc_second_order=True, skip_values=2048)

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Ishigami.evaluate(param_values)

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = sobol.analyze(
    problem, Y, calc_second_order=True, conf_level=0.95, print_to_console=True
)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# e.g. Si['S1'] contains the first-order index for each parameter,
# in the same order as the parameter file
# The optional second-order indices are now returned in keys 'S2', 'S2_conf'
# These are both upper triangular DxD matrices with nan's in the duplicate
# entries.
# Optional keyword arguments parallel=True and n_processors=(int) for parallel execution
# using multiprocessing


corn_params = read_param_file(wd+"/data/sa_params/CornSA_GMD4_WNdlovu_v1_20230125.txt") # read in corn parameters

# parameter names
param_names = dict((k, corn_params[k]) for k in ['names'] # get key with names
           if k in corn_params)
param_names = param_names.get('names') # get values from names dictionary


# influential parameters
with open(wd+'/data/analysis_results/sensitivity_analysis_runs/ks_irrig_corn_morris.pickle', 'rb') as mi: 
    influ_morris_irrig = pickle.load(mi)

with open(wd+'/data/analysis_results/sensitivity_analysis_runs/ks_rainfed_corn_morris.pickle', 'rb') as mr: 
    influ_morris_rainfed = pickle.load(mr)


 
    


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




# morris sensitive parameters

def Parameters(morris_results): 
    yield_params_list = [] 
    irrig_params_list = []
    for item in morris_results:
    
           # yield
           yield_influ = list(item[1]['names'])
           
           #print(yield_params)
           yield_params = {'bounds': [],
                       'dists': corn_params['dists'],
                       'groups': [],
                       'names': [],
                       'num_vars': 0}
        
           yield_names_list = corn_params['names']
           yield_count = 0
            
           for i in range(len(yield_names_list)):
               if yield_names_list[i] in yield_influ:
                   yield_count +=1
                   yield_params['bounds'].append(corn_params['bounds'][i])
                   yield_params['names'].append(yield_names_list[i])
                   yield_params['groups'].append(yield_names_list[i])
                   yield_params['num_vars'] = yield_count
           yield_params_list.append(yield_params)  
           
           # irrigation
           irrig_influ = list(item[2]['names'])
           
           #print(yield_params)
           irrig_params = {'bounds': [],
                       'dists': corn_params['dists'],
                       'groups': [],
                       'names': [],
                       'num_vars': 0}
        
           irrig_names_list = corn_params['names']
           irrig_count = 0
            
           for i in range(len(irrig_names_list)):
               if irrig_names_list[i] in irrig_influ:
                   irrig_count +=1
                   irrig_params['bounds'].append(corn_params['bounds'][i])
                   irrig_params['names'].append(irrig_names_list[i])
                   irrig_params['groups'].append(irrig_names_list[i])
                   irrig_params['num_vars'] = irrig_count
           irrig_params_list.append(irrig_params)  
           
    return yield_params_list, irrig_params_list
     



# create list of dataframes with the influential parameters for each treatment (rainfed, normal and irrigated)

morris_irrig = [influ_morris_irrig]
yield_list = list(map(Parameters, [influ_morris_irrig]))[0][0]

h1_zero = h1[0][0][0]




from functools import partial
h, h2 = list(map(partial(map, Parameters), test)))




# weather data
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
    

# total runs = 2N(p+1)
h1 = yield_list_irrig[1]
param_values_sobol = saltelli.sample(h1, 2, calc_second_order=True, skip_values=4)
param_values_sobol = pd.DataFrame(param_values_sobol)
param_values_sobol.columns = h1['names']


new_names = {
    'tb': 'Tbase', 
    'tu': 'Tupp',
    'ccs': 'SeedSize',
    'den': 'PlantPop',
    'eme': 'Emergence',
    'cgc':'CGC',
    'ccx': 'CCx',
    'sen': 'Senescence',
    'cdc': 'CDC',
    'mat': 'Maturity',
    'rtm': 'Zmin',
    'flolen': 'Flowering',
    'rtx': 'Zmax',
    'rtshp': 'fshape_r',
    'root': 'MaxRooting',
    'rtexup': 'SxTopQ',
    'rtexlw': 'SxBotQ',
     #Crop Transpiration
    'kc': 'Kcb',
    'kcdcl': 'fage',
     #Biomass and Yield
    'wp': 'WP',
    'wpy': 'WPy',
    'hi': 'HI0',
    'hipsflo': 'dHI_pre',
    'exc': 'exc',
    'hipsveg': 'a_HI',
    'hingsto': 'b_HI',
    'hinc': 'dHI0',
    'hilen': 'YldForm',
     #Water and Temperature Stress
    'polmn': 'Tmin_up',
    'polmx': 'Tmax_up',
    'pexup': 'p_up',
    'pexlw': 'p_lo',
    'pexshp': 'fshape_w',
    'smt': 'SMT'
    }

param_values_sobol = param_values_sobol.rename(columns=new_names)




sher_soils = custom_soil[10]
custom = sher_soils
sim_start = '2021/01/01' #dates to match crop data
sim_end = '2021/12/31'
initWC = InitialWaterContent(value=['FC'])


sher_gridMET2 = sher_gridMET[sher_gridMET['year'] == 2021] # filter for 2012
sher_gridMET2 = sher_gridMET2.drop(['year'], axis=1) # drop year variable
wdf = sher_gridMET2



sobol_list_yield = []
sobol_list_irrig = []
model_list = []






# Loop through the rows of the dataframe and extract each row into a new dataframe
for i in range(0,len(param_values_sobol)):
    # Create an empty list to store the ith row
    ith_row = []
    
    # Loop through the columns of the dataframe and extract the ith value
    for col_name in param_values_sobol.columns:
        ith_value = param_values_sobol[col_name][i]
        ith_row.append(ith_value)
    
    # Create a new dataframe with data from the ith row
    new_df = pd.DataFrame([ith_row], columns=param_values_sobol.columns)
    new_df_melt = new_df.melt(ignore_index=True).reset_index()
    #new_df_melt = new_df_melt[new_df_melt.variable != 'SMT']
    
    crop = Crop("Maize", planting_date='05/01')
    
    if 'SMT' in new_df.columns:
    
        irr_mngt = IrrigationManagement(irrigation_method=1,
                                   SMT=[new_df['SMT'][0]]*4)

        for index, row in new_df_melt.iterrows():
        
            
            
            if hasattr(crop, row['variable']):
               setattr(crop, row['variable'], row['value'])
            
            #irr_mngt = IrrigationManagement(irrigation_method=1,
                                           #SMT=[new_df['SMT'][0]]*4)
            model = AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC, irr_mngt)
                #model_irrig.append(model)
            model.run_model(till_termination=True) # run model till the end
            model_df = model._outputs.final_stats
                #print(model_df)
               # model_df = model_df.reset_index()
                #model_irrig.append(model_df)
            
            
            # Append the new dataframe to the list of new dataframes
            
    else:
        irr_mngt = IrrigationManagement(irrigation_method=1,
                                   SMT=[80]*4)
        for index, row in new_df_melt.iterrows():
         
             
             
             if hasattr(crop, row['variable']):
                setattr(crop, row['variable'], row['value'])
             
             #irr_mngt = IrrigationManagement(irrigation_method=1,
                                            #SMT=[new_df['SMT'][0]]*4)
             model = AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC, irr_mngt)
                 #model_irrig.append(model)
             model.run_model(till_termination=True) # run model till the end
             model_df = model._outputs.final_stats
            
    model_list.append(model_df)
   
    
   
model_df_full = pd.concat(model_list)


yield_vals = np.array(model_df_full[['Yield (tonne/ha)']]).ravel() # select yield column and tranform to array (x,)
      

Si_sobol_yield = sobol.analyze( # returns dictionary
        h1, yield_vals, calc_second_order=True, conf_level=0.95, print_to_console=True
        )

sobol_list_yield.append(Si_sobol_yield)


param_values_sobol = pd.DataFrame(param_values_sobol)
param_values_sobol.columns = h1[0][0][0]['names']

# sample of 10
#param_values_sobol = param_values_sobol.head(20)

2**8

# Run the "model" and save the output in a text file
# This will happen offline for external models
#Y_sobol= Ishigami.evaluate(param_values_sobol)


from pandas import DataFrame
 
numbers = {'mynumbers': [51, 52, 53, 54, 55]}
df = DataFrame(numbers, columns =['mynumbers'])
 
df['<= 53'] = df['mynumbers'].apply(lambda x: 'True' if x <= 53 else 'False')




sim_start = f'{2021}/01/01'#'2021/01
sim_end = f'{2021}/12/31'
initWC = InitialWaterContent(value=['FC'])
#irr_mngt = IrrigationManagement(irrigation_meth
model_irrig_sobol = []
#while True:
#model_irrig2 = pd.DataFrame([])  
colnames = ['ccx', 'rtexup', 'rtexlw', 'wp', 'wpy', 'hi', 'smt'] 
newnames = ['CCx', 'SxTopQ', 'SxBotQ', 'WP', 'WPy', 'HI0', 'smt']

param_values_sobol.columns = newnames








new_dfs = []

# Loop through the rows of the dataframe and extract each row into a new dataframe
for i in range(0,len(param_values_sobol)):
    # Create an empty list to store the ith row
    ith_row = []
    
    # Loop through the columns of the dataframe and extract the ith value
    for col_name in param_values_sobol.columns:
        ith_value = param_values_sobol[col_name][i]
        ith_row.append(ith_value)
    
    # Create a new dataframe with data from the ith row
    new_df = pd.DataFrame([ith_row], columns=param_values_sobol.columns)
    new_df_melt = new_df.melt(ignore_index=True).reset_index()
    
    cropz2 = Crop("Maize", planting_date='05/01')
    irr_mngt = IrrigationManagement(irrigation_method=1,
                                   SMT=[new_df['smt'][0]]*4)

    for index, row in new_df_melt.iterrows():
        if hasattr(cropz, row['variable']):
           setattr(cropz, row['variable'], row['value'])
        
        
        model = AquaCropModel(sim_start,sim_end,wdf,custom,cropz2,initWC, irr_mngt)
            #model_irrig.append(model)
        model.run_model(till_termination=True) # run model till the end
        model_df = model._outputs.final_stats
            #print(model_df)
           # model_df = model_df.reset_index()
            #model_irrig.append(model_df)
            
            
            # Append the new dataframe to the list of new dataframes
    new_dfs.append(model_df)

        
#5:23pm - 6:22 stopped (taking too long)

model_df_full = pd.concat(new_dfs)
            
   
yield_vals = np.array(model_df_full[['Yield (tonne/ha)']]).ravel() # select yield column and tranform to array (x,)
      

Si_sobol = sobol.analyze(
        h1_zero, yield_vals, calc_second_order=True, conf_level=0.95, print_to_console=True
        )















problem = read_param_file("/Users/wayne/Downloads/Ishigami.txt")

problem = {
 'num_vars': 8,
 'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
  'groups': None,  'bounds': [[0.0, 1.0],
             [0.0, 1.0],
             [0.0, 1.0],
            [0.0, 1.0],
             [0.0, 1.0],
             [0.0, 1.0],
             [0.0, 1.0],
            [0.0, 1.0]]
 }


param_values = saltelli.sample(problem, 50, calc_second_order=True)

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Ishigami.evaluate(param_values)

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = sobol.analyze(
    problem, Y, calc_second_order=True, conf_level=0.95, print_to_console=True
)














##########
for i in range(len(args)):
    #print(i)
    # Check if the argument is a column name in the dataframe
        if i in param_values_sobol.columns:
            print(i)
            arg_va = param_values_sobol[arg].values
            
            
            
            args[i] = arg_va
            
            crop = Crop2("Maize", planting_date='05/01', args)
 inspect.getfullargspec(Crop2)[0][1:]           
df = pd.DataFrame({'arg1':[ 2], 'arg2':[3]})
#default_args = [1, 2, 3]  # Default values for arguments 'arg1', 'arg2', and 'arg3'  
def default_args(arg1 = 4, arg2 = 5, arg3 = 6):
    fg1 = arg1
    fg2 = arg2
    fg3 = arg3
    return fg1, fg2, fg3
 

test_list = []

for i in inspect.getfullargspec(default_args)[0]:
    # Check if the argument is a column name in the dataframe
    if i in df.columns:
        # Use the value from the dataframe if it's present, otherwise use the default value
        print(i)
        test3 = df[i].iloc[0] if not pd.isna(df[i].iloc[0]) else default_args()[i]
        test_list.append(test3)
    else:
        # Use the default value if the argument is not in the dataframe
        test4 = default_args()[i]




def default_args(arg1=4, arg2=5, arg3=6):
    fg1 = arg1
    fg2 = arg2
    fg3 = arg3
    return fg1, fg2, fg3

df = pd.DataFrame({'arg1':[2], 'arg2':[3]})

for i in inspect.getfullargspec(default_args)[0][:]:
    if i in df.columns:
        arg_value = df[i].values[0]
    else:
        arg_value = None
    
    if i == 'arg1':
        arg1 = arg_value or 4
    elif i == 'arg2':
        arg2 = arg_value or 5
    elif i == 'arg3':
        arg3 = arg_value or 6

test5 = default_args(arg1=arg1, arg2=arg2, arg3=arg3)
print(test5)

Aer= 5.0,
  CCx= 0.96,
  CDC= -9.0,
  CDC_CD= 0.11691,
  CGC= -9.0,
  CGC_CD= 0.16312,
  CalendarType= 1,
  CropType= 3,
  Determinant= 1.0,
  ETadj= 1.0,
  Emergence= -9.0,
  EmergenceCD= 6.0,
  Flowering= -9.0,
  FloweringCD= 13.0,
  GDD_lo= 0,
  GDD_up= 12.0,
  GDDmethod= 3,
  HI0= 0.48,
  HIstart= -9.0,
  HIstartCD= 66.0,
  Kcb= 1.05,
  Maturity= -9.0,
  MaturityCD= 132.0,
  MaxRooting= -9.0,
  MaxRootingCD= 108.0,
  #Name= 'Maize',
  PlantMethod= 1.0,
  PlantPop= 75000.0,
  PolColdStress= 1,
  PolHeatStress= 1,
  SeedSize= 6.5,
  Senescence= -9.0,
  SenescenceCD= 107.0,
  SwitchGDD= 0,
  SxBotQ= 0.011,
  SxTopQ= 0.045,
  Tbase= 8.0,
  Tmax_lo= 45.0,
  Tmax_up= 40.0,
  Tmin_lo= 5.0,
  Tmin_up= 10.0,
  TrColdStress= 1,
  Tupp= 30.0,
  WP= 33.7,
  WPy= 100.0,
  YldForm= -9.0,
  YldFormCD= 61.0,
  Zmax= 2.3,
  Zmin= 0.3,
  a_HI= 7.0,
  b_HI= 3.0,
  dHI0= 15.0,
  dHI_pre= 0.0,
  exc= 50.0,
  fage= 0.3,
  fshape_r= 1.3,
  fshape_w1= 2.9,
  fshape_w2= 6.0,
  fshape_w3= 2.7,
  fshape_w4= 1,
  fsink= 0.5,
  p_lo1= 0.72,
  p_lo2= 1,
  p_lo3= 1,
  p_lo4= 1,
  p_up1= 0.14,
  p_up2= 0.69,
  p_up3= 0.69,
  p_up4= 0.8)

for i in inspect.getfullargspec(Crop2)[0][:]::
    #print(i)
    # Check if the argument is a column name in the dataframe
        if i in param_values_sobol.columns:
            arg_value = param_values_sobol[i].values[0]
        else:
            arg_value = None
        if i == 'Aer':
            Aer = arg_value or 5.0
        if i == 'CCx':
            CCx = arg_value or 0.96
        if i == 'CDC':
            CDC = arg_value or -9.0
        if i == 'CDC_CD':
            CDC_CD = arg_value or 0.11691
        if i == 'CGC':
            CGC = arg_value or 9.0
        if i == 'CGC_CD':
            CGC_CD = arg_value or 0.16312
        if i == 'CalendarType':
            CalendarType = arg_value or 1
        if i == 'CropType':
            CropType = arg_value or 3
        if i == 'Determinant':
            Determinant = arg_value or 1.0
        if i == 'ETadj':
            ETadj = arg_value or 1.0
        if i == 'Emergence':
            Emergence = arg_value or -9.0
        if i == 'EmergenceCD':
            EmergenceCD = arg_value or 6.0
        if i == 'Flowering':
            Flowering = arg_value or -9.0
        if i == 'FloweringCD':
            FloweringCD = arg_value or 13.0
        if i == 'GDD_lo':
            GDD_lo = arg_value or 0
        if i == 'GDD_up':
            GDD_up = arg_value or 12.0
        if i == 'GDDmethod':
            GDDmethod = arg_value or 3.0
        if i == 'GHI0':
            HI0 = arg_value or 0.48
        if i == 'HIstart':
            HIstart = arg_value or 0.48
            
            
       
            
args = {}
for arg in inspect.getfullargspec(Crop2)[0]:
    if arg in param_values_sobol.columns:
        args[arg] = param_values_sobol[arg].values[0]
    else:
        args[arg] = default_args.__defaults__[inspect.getfullargspec(default_args)[0].index(arg)]
        
test = Crop2(**args)
print(test)            
 
           
for i in range(len(inspect.getfullargspec(default_args)[0])):
    arg_name = inspect.getfullargspec(default_args)[0][i]
    if arg_name in df.columns:
        arg_value = df[arg_name][0]
    else:
        arg_value = inspect.getfullargspec(default_args)[3][i]
    locals()[arg_name] = arg_value

result = default_args(arg1, arg2, arg3)
print(result)


for i in range(len(inspect.getfullargspec(Crop2)[0])):
    arg_name = inspect.getfullargspec(Crop2)[0][i]
    if arg_name in param_values_sobol.columns:
        arg_value = param_values_sobol[arg_name][0]
    else:
        arg_value = inspect.getfullargspec(Crop2)[3][i]
    locals()[arg_name] = arg_value

result = Crop2(fshape_b, PctZmin, fshape_ex, ETadj, Aer, LagAer, beta, a_Tr, GermThr, CCmin, MaxFlowPct, HIini, bsted, bface, Name, CropType, PlantMethod, CalendarType, SwitchGDD, planting_date, harvest_date, Emergence, MaxRooting, Senescence, Maturity, HIstart, Flowering, YldForm, GDDmethod, Tbase, Tupp, PolHeatStress, Tmax_up, Tmax_lo, PolColdStress, Tmin_up, Tmin_lo, TrColdStress, GDD_up, GDD_lo, Zmin, Zmax, fshape_r, SxTopQ, SxBotQ, SeedSize, PlantPop, CCx, CDC, CGC, Kcb, fage, WP, WPy, fsink, HI0, dHI_pre, a_HI, b_HI, dHI0, Determinant, exc, p_up1, p_up2, p_up3, p_up4, p_lo1, p_lo2, p_lo3, p_lo4, fshape_w1, fshape_w2, fshape_w3, fshape_w4, CGC_CD, CDC_CD, EmergenceCD, MaxRootingCD, SenescenceCD, MaturityCD, HIstartCD, FloweringCD, YldFormCD)
print(result)


           
            
            
            
for col in param_values_sobol.columns:
    print(Crop(**param_values_sobol[col]))           
            
            
            

            irr_mngt = IrrigationManagement(irrigation_method=0)
            crop1 = Crop2("Maize", planting_date='05/01', 
                        ##Crop Development
                        Tbase = param_values_sobol[i],
                        Tupp = param_values_sobol[i], 
                        SeedSize = param_values_sobol[i], 
                        PlantPop = param_values_sobol[i],
                        Emergence = param_values_sobol[i],
                        CGC = param_values_sobol[i], 
                        CCx = param_values_sobol[i], 
                        Senescence = param_values_sobol[i],
                        CDC = param_values_sobol[i], 
                        Maturity = param_values_sobol[i],
                        Zmin = param_values_sobol[i],
                        Flowering = param_values_sobol[i],
                        Zmax = param_values_sobol[i],
                        fshape_r = param_values_sobol[i],
                        MaxRooting = param_values_sobol[i],
                        SxTopQ = param_values_sobol[i],
                        SxBotQ = param_values_sobol[i],
                         #Crop Transpiration
                        Kcb = param_values_sobol[i],
                        fage = param_values_sobol[i],
                         #Biomass and Yield
                        WP = param_values_sobol[i],
                        WPy = param_values_sobol[i],
                        HI0 = param_values_sobol[i],
                        dHI_pre = param_values_sobol[i],
                        exc = param_values_sobol[i],
                        a_HI = param_values_sobol[i],
                        b_HI = param_values_sobol[i],
                        dHI0 = param_values_sobol[i],
                        YldForm = param_values_sobol[i],
                         #Water and Temperature Stress
                        Tmin_up = param_values_sobol[i],
                        Tmax_up = param_values_sobol[i],
                        p_up = param_values_sobol[i],
                        p_lo = param_values_sobol[i],
                        fshape_w = param_values_sobol[i])
        else:
            
            irr_mngt = IrrigationManagement(irrigation_method=0)
            crop2 = Crop2("Maize", planting_date='05/01',
                         ##Crop Development
                        Aer= 5.0,
                          CCx= 0.96,
                          CDC= -9.0,
                          CDC_CD= 0.11691,
                          CGC= -9.0,
                          CGC_CD= 0.16312,
                          CalendarType= 1,
                          CropType= 3,
                          Determinant= 1.0,
                          ETadj= 1.0,
                          Emergence= -9.0,
                          EmergenceCD= 6.0,
                          Flowering= -9.0,
                          FloweringCD= 13.0,
                          GDD_lo= 0,
                          GDD_up= 12.0,
                          GDDmethod= 3,
                          HI0= 0.48,
                          HIstart= -9.0,
                          HIstartCD= 66.0,
                          Kcb= 1.05,
                          Maturity= -9.0,
                          MaturityCD= 132.0,
                          MaxRooting= -9.0,
                          MaxRootingCD= 108.0,
                          #Name= 'Maize',
                          PlantMethod= 1.0,
                          PlantPop= 75000.0,
                          PolColdStress= 1,
                          PolHeatStress= 1,
                          SeedSize= 6.5,
                          Senescence= -9.0,
                          SenescenceCD= 107.0,
                          SwitchGDD= 0,
                          SxBotQ= 0.011,
                          SxTopQ= 0.045,
                          Tbase= 8.0,
                          Tmax_lo= 45.0,
                          Tmax_up= 40.0,
                          Tmin_lo= 5.0,
                          Tmin_up= 10.0,
                          TrColdStress= 1,
                          Tupp= 30.0,
                          WP= 33.7,
                          WPy= 100.0,
                          YldForm= -9.0,
                          YldFormCD= 61.0,
                          Zmax= 2.3,
                          Zmin= 0.3,
                          a_HI= 7.0,
                          b_HI= 3.0,
                          dHI0= 15.0,
                          dHI_pre= 0.0,
                          exc= 50.0,
                          fage= 0.3,
                          fshape_r= 1.3,
                          fshape_w1= 2.9,
                          fshape_w2= 6.0,
                          fshape_w3= 2.7,
                          fshape_w4= 1,
                          fsink= 0.5,
                          p_lo1= 0.72,
                          p_lo2= 1,
                          p_lo3= 1,
                          p_lo4= 1,
                          p_up1= 0.14,
                          p_up2= 0.69,
                          p_up3= 0.69,
                          p_up4= 0.8)


















     
for i in inspect.getfullargspec(default_args)[0][:]:
    for j in range(0, len(df)):
    #arg = f"arg{i+1}"  # Get the corresponding argument name (e.g. 'arg1' for index 0)
        #print(j)
    # Check if the argument is a column name in the dataframe
        if j in df.columns:
        #for j in range(0, len(df)):
            print(j)
            #print(df[i].values)
            test1 = default_args(
            arg1=df[j],
            arg2=df[j],
            arg3=df[j])
        
        else:
            test2 = default_args(
            arg1=4,
            arg2=5,
            arg3=6)
            



import pandas as pd

# Create dataframe1 with variables a, b, and c
df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

# Create dataframe2 with variables a, b, c, d, and e
df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60], 'c': [70, 80, 90], 'd': [100, 200, 300], 'e': [400, 500, 600]})

# Create dataframe3
df3 = pd.DataFrame()

# Iterate over the columns in dataframe2
for col in df2.columns:
    
    # Check if the column name exists in dataframe1
    if col in df1.columns:
        
        # Replace the values in dataframe3 with the values from dataframe1
        df3[col] = df1[col]
    else:
        
        # Add the variable only in dataframe2 to dataframe3
        df3[col] = df2[col]

# Print the dataframes
print("Dataframe1:\n", df1)
print("\nDataframe2:\n", df2)
print("\nDataframe3:\n", df3)
            
            
            
        # Replace the argument with the corresponding values from the dataframe
        arg_values = df[arg].values
        default_args[i] = arg_values
    else:
        # Use default values for the argument
        default_args[i] = default_args[i]  
print(default_args)        
        
        
        
for i in (args):
    #print(i)
    # Check if the argument is a column name in the dataframe
        if i in param_values_sobol.columns:
            print(param_values_sobol[i])  

            irr_mngt = IrrigationManagement(irrigation_method=0)
            crop1 = Crop2("Maize", planting_date='05/01', 
                        ##Crop Development
                        Tbase = param_values_sobol[i],
                        Tupp = param_values_sobol[i], 
                        SeedSize = param_values_sobol[i], 
                        PlantPop = param_values_sobol[i],
                        Emergence = param_values_sobol[i],
                        CGC = param_values_sobol[i], 
                        CCx = param_values_sobol[i], 
                        Senescence = param_values_sobol[i],
                        CDC = param_values_sobol[i], 
                        Maturity = param_values_sobol[i],
                        Zmin = param_values_sobol[i],
                        Flowering = param_values_sobol[i],
                        Zmax = param_values_sobol[i],
                        fshape_r = param_values_sobol[i],
                        MaxRooting = param_values_sobol[i],
                        SxTopQ = param_values_sobol[i],
                        SxBotQ = param_values_sobol[i],
                         #Crop Transpiration
                        Kcb = param_values_sobol[i],
                        fage = param_values_sobol[i],
                         #Biomass and Yield
                        WP = param_values_sobol[i],
                        WPy = param_values_sobol[i],
                        HI0 = param_values_sobol[i],
                        dHI_pre = param_values_sobol[i],
                        exc = param_values_sobol[i],
                        a_HI = param_values_sobol[i],
                        b_HI = param_values_sobol[i],
                        dHI0 = param_values_sobol[i],
                        YldForm = param_values_sobol[i],
                         #Water and Temperature Stress
                        Tmin_up = param_values_sobol[i],
                        Tmax_up = param_values_sobol[i],
                        p_up = param_values_sobol[i],
                        p_lo = param_values_sobol[i],
                        fshape_w = param_values_sobol[i])
        else:
            
            irr_mngt = IrrigationManagement(irrigation_method=0)
            crop2 = Crop2("Maize", planting_date='05/01',
                         ##Crop Development
                        Aer= 5.0,
                          CCx= 0.96,
                          CDC= -9.0,
                          CDC_CD= 0.11691,
                          CGC= -9.0,
                          CGC_CD= 0.16312,
                          CalendarType= 1,
                          CropType= 3,
                          Determinant= 1.0,
                          ETadj= 1.0,
                          Emergence= -9.0,
                          EmergenceCD= 6.0,
                          Flowering= -9.0,
                          FloweringCD= 13.0,
                          GDD_lo= 0,
                          GDD_up= 12.0,
                          GDDmethod= 3,
                          HI0= 0.48,
                          HIstart= -9.0,
                          HIstartCD= 66.0,
                          Kcb= 1.05,
                          Maturity= -9.0,
                          MaturityCD= 132.0,
                          MaxRooting= -9.0,
                          MaxRootingCD= 108.0,
                          #Name= 'Maize',
                          PlantMethod= 1.0,
                          PlantPop= 75000.0,
                          PolColdStress= 1,
                          PolHeatStress= 1,
                          SeedSize= 6.5,
                          Senescence= -9.0,
                          SenescenceCD= 107.0,
                          SwitchGDD= 0,
                          SxBotQ= 0.011,
                          SxTopQ= 0.045,
                          Tbase= 8.0,
                          Tmax_lo= 45.0,
                          Tmax_up= 40.0,
                          Tmin_lo= 5.0,
                          Tmin_up= 10.0,
                          TrColdStress= 1,
                          Tupp= 30.0,
                          WP= 33.7,
                          WPy= 100.0,
                          YldForm= -9.0,
                          YldFormCD= 61.0,
                          Zmax= 2.3,
                          Zmin= 0.3,
                          a_HI= 7.0,
                          b_HI= 3.0,
                          dHI0= 15.0,
                          dHI_pre= 0.0,
                          exc= 50.0,
                          fage= 0.3,
                          fshape_r= 1.3,
                          fshape_w1= 2.9,
                          fshape_w2= 6.0,
                          fshape_w3= 2.7,
                          fshape_w4= 1,
                          fsink= 0.5,
                          p_lo1= 0.72,
                          p_lo2= 1,
                          p_lo3= 1,
                          p_lo4= 1,
                          p_up1= 0.14,
                          p_up2= 0.69,
                          p_up3= 0.69,
                          p_up4= 0.8)








######################
Tbase = param_values_sobol[i],
Tupp = param_values_sobol['Tupp'][i], 
SeedSize = param_values_sobol['SeedSize'][i], 
PlantPop = param_values_sobol['PlantPop'][i],
Emergence = param_values_sobol['Emergence'][i],
CGC = param_values_sobol['CGC'][i], 
CCx = param_values_sobol['CCx'][i], 
Senescence = param_values_sobol['Senescence'][i],
CDC = param_values_sobol['CDC'][i], 
Maturity = param_values_sobol['Maturity'][i],
Zmin = param_values_sobol['Zmin'][i],
Flowering = param_values_sobol['Flowering'][i],
Zmax = param_values_sobol['Zmax'][i],
fshape_r = param_values_sobol['fshape_r'][i],
MaxRooting = param_values_sobol['MaxRooting'][i],
SxTopQ = param_values_sobol['SxTopQ'][i],
SxBotQ = param_values_sobol['SxBotQ'][i],
 #Crop Transpiration
Kcb = param_values_sobol['Kcb'][i],
fage = param_values_sobol['kcdcl'][i],
 #Biomass and Yield
WP = param_values_sobol['wp'][i],
WPy = param_values_sobol['wpy'][i],
HI0 = param_values_sobol['hi'][i],
dHI_pre = param_values_sobol['hipsflo'][i],
exc = param_values_sobol['exc'][i],
a_HI = param_values_sobol['hipsveg'][i],
b_HI = param_values_sobol['hingsto'][i],
dHI0 = param_values_sobol['hinc'][i],
YldForm = param_values_sobol['hilen'][i],
 #Water and Temperature Stress
Tmin_up = param_values_sobol['polmn'][i],
Tmax_up = param_values_sobol['polmx'][i],
p_up = param_values_sobol['pexup'][i],
p_lo = param_values_sobol['pexlw'][i],
fshape_w = param_values_sobol['pexshp'][i])




        else:
            irr_mngt = IrrigationManagement(irrigation_method=0)
            crop = Crop("Maize", planting_date='05/01')
                     
                       ## run model params
            model_sobol = AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC, irr_mngt)
            #model_irrig.append(model)
            model_sobol.run_model(till_termination=True) # run model till the end
            model_sobol_df = model_sobol._outputs.final_stats
            #print(model_df)
           # model_df = model_df.reset_index()
            model_irrig_sobol.append(model_sobol_df)
            
model_sobol_full = pd.concat(model_irrig_sobol)





# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = sobol.analyze(
   h1[0][0][0], Y_sobol, calc_second_order=True, conf_level=0.95, print_to_console=True
)



dt = list(param_values_sobol)

sim_start = f'{2001}/01/01'#'2021/01
sim_end = f'{2001}/12/31'
initWC = InitialWaterContent(value=['FC'])
x = sher_gridMET[sher_gridMET['year']== 2001]  # filter for yr

wdf = x.drop(['year'], axis=1)
initWC = InitialWaterContent(value=['FC'])
irr_mngt = IrrigationManagement(irrigation_method=1,
                           SMT=[param_values['smt'][i]]*4)
crop

d = map(AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC, irr_mngt), param_values_sobol)

def stars(ccx, rtexup, rtexlw, wp, wpy, hi, smt):
    new_val = ccx* rtexup* rtexlw
    new_val2 = wp + wpy+ hi+ smt
    return(new_val, new_val2)



df_params = h1[0][0][0]['names']
for i in range(len(df_params)):
    if df_params[i] in Crop:
        print(df_params)


for column in param_values_sobol:
    
    
for index, row in  param_values_sobol.iterrows():
    print(row)
        row = list(index)
        #row = [row[i::4] for i in range(row)]
        gh = list(map(stars, row))
        print(row)
    #row = list(map(lambda x: x.split(', '), row))
       #print(row)
        row = [0.8202490234375, 0.019068603515625, 0.00460302734375, 31.234130859375, 115.97900390625, 0.489423828125, 64.193115234375]
        gh = list(map(stars, row))


import inspect 
print(inspect.signature(Crop2)) 

def replace_args_in_dataframe(df, func):
    """
    Replace arguments in a function with values from a Pandas DataFrame
    """

args = inspect.getfullargspec(Crop2)[0][1:]
for i in range(len(args)):
        #return(arg)
        if args[i] in list(param_values_sobol.columns):
            print(i)
            #return(arg)
            z = Crop2.__globals__[arg] = param_values_sobol[arg].values
            


for i in (args):
    #print(i)
    # Check if the argument is a column name in the dataframe
    if i in param_values_sobol.columns:
        print(i)
        #arg_values = param_values_sobol[i].values
        #print(arg_values)
        #args[i] = arg_values
    else:
        # Use default values for the argument
        args[i] = args[i]
print(args.values)
        
df = pd.DataFrame({'arg1': [1, 2, 3], 'b': [4, 5, 6]})
default_args = [1, 2, 3]  # Default values for arguments 'arg1', 'arg2', and 'arg3'

# Loop through the indices of the default_args list
for i in range(len(default_args)):
    arg = f"arg{i+1}"  # Get the corresponding argument name (e.g. 'arg1' for index 0)

    # Check if the argument is a column name in the dataframe
    if arg in df.columns:
        # Replace the argument with the corresponding values from the dataframe
        arg_values = df[arg].values
        default_args[i] = arg_values
    else:
        # Use default values for the argument
        default_args[i] = default_args[i]  # Retain the default value

# Print the updated list
print(default_args)       
        
        
        for j in range(len(arg_values)):
            args[arg][j] = arg_values[j]
        #print(arg)
        # Replace the argument with the corresponding values from the dataframe
        irr_mngt = IrrigationManagement(irrigation_method=0)
        crop = Crop("Maize", planting_date='05/01', 
                    ##Crop Development
                    Tbase = param_values_sobol[arg],

                    Tupp = param_values_sobol[arg])
                    #SeedSize = param_values_sobol[arg],
                    #PlantPop = param_values_sobol[arg],
                    #Emergence = param_values_sobol[arg])
                    #CGC = param_values_sobol[arg])
                    #CCx = param_values_sobol[arg])
                    #Senescence = param_values_sobol[arg])
    else:
        # Use default values for the argument
        pass
        model = AquaCropModel(sim_start,sim_end,wdf,custom,crop,initWC, irr_mngt)
        #model_irrig.append(model)
        model.run_model(till_termination=True) # run model till the end
        model_df = model._outputs.final_stats
        #print(model_df)
       # model_df = model_df.reset_index()
        model_irrig.append(model_df)
        
model_df_full = pd.concat(model_irrig)
        
        
        
                    CDC = param_values_sobol['cdc'][i], 
                    Maturity = param_values_sobol['mat'][i],
                    Zmin = param_values_sobol['rtm'][i],
                     
                    Flowering = param_values_sobol['flolen'][i],
                    Zmax = param_values_sobol['rtx'][i],
                    fshape_r = param_values_sobol['rtshp'][i],
                    MaxRooting = param_values_sobol['root'][i],
                    SxTopQ = param_values_sobol['rtexup'][i],
                    SxBotQ = param_values_sobol['rtexlw'][i],
                     #Crop Transpiration
                    Kcb = param_values_sobol['kc'][i],
                    fage = param_values_sobol['kcdcl'][i],
                     #Biomass and Yield
                    WP = param_values_sobol['wp'][i],
                    WPy = param_values_sobol['wpy'][i],
                    HI0 = param_values_sobol['hi'][i],
                    dHI_pre = param_values_sobol['hipsflo'][i],
                    exc = param_values_sobol['exc'][i],
                    a_HI = param_values_sobol['hipsveg'][i],
                    b_HI = param_values_sobol['hingsto'][i],
                    dHI0 = param_values_sobol['hinc'][i],
                    YldForm = param_values_sobol['hilen'][i],
                     #Water and Temperature Stress
                    Tmin_up = param_values_sobol['polmn'][i],
                    Tmax_up = param_values_sobol['polmx'][i],
                    p_up = param_values_sobol['pexup'][i],
                    p_lo = param_values_sobol['pexlw'][i],
                    fshape_w = param_values_sobol['pexshp'][i])
        arg_values = param_values_sobol[arg].values
        print(arg_values)
        for i in range(len(arg_values)):
            exec(f"{arg}[{i}] = {arg_values[i]}")
    else:
        # Use default values for the argument
        default_value = arg  # Replace with your own function to get the default value
        exec(f"{arg} = {default_value}")


            
            
    return func(*func.__code__.co_varnames[:func.__code__.co_argcount])

sf = replace_args_in_dataframe(param_values_sobol, Crop2)


colnames = ['ccx', 'rtexup', 'rtexlw', 'wp', 'wpy', 'hi', 'smt'] 


for arg in inspect.getfullargspec(Crop2.__init__):
    print(arg)
    if arg in newnames:
        print('true')
    else:
        print('false')

def replace_arguments(df, func):
    # Get the column names of the dataframe
    column_names = list(df.columns)

    # Loop through the arguments of the function
    for i, arg in enumerate(func):
        # Check if the argument matches a column name
        if arg in column_names:
            # Replace the argument with the corresponding value from the dataframe
            func.__defaults__ = func.__defaults__[:i] + (df[arg],) + func.__defaults__[i+1:]

# Example usage
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
def my_func(a, b, c):
    return a + b + c

# Replace arguments in my_func with values from df
result = replace_arguments(param_values_sobol, Crop2)
print(result)

def update_crop_from_df(df, crop):
    for index, row in df.iterrows():
        arg_name = row['arg_name']
        if arg_name in crop:
            crop[arg_name] = row['value']
    return crop

crop_func = update_crop_from_df(Crop, param_values_sobol)

# Call the crop_func to get the result of Crop2 with the updated arguments
result = crop_func()

cropz = Crop("Maize", planting_date='05/01')

for index, row in dfm.iterrows():
    if hasattr(cropz, row['variable']):
        setattr(cropz, row['variable'], row['value'])



dfm = param_values_sobol.melt(ignore_index=True).reset_index()
