# recalculate t0c in celcius and create ymd variables
gridMET = gridMET.assign(Tmin = gridMET.tmmn-273.15,
                    Tmax = gridMET.tmmx-273.15,
                    date = pd.to_datetime(gridMET['date_ymd'], format='%Y%m%d'))


# separiting date
gridMET = gridMET.assign(day =  gridMET['date'].dt.day,
                         month = gridMET['date'].dt.month,
                         year = gridMET['date'].dt.year)#!/usr/bin/env python3
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
soils_soil_df_full = pd.read_csv(wd + '/data/agricLand/soils/Soil_FieldsAroundSD6KS_POLARIS_AGrinstead_20220706.csv')
soils_soil_df = soils_soil_df_full[soils_soil_df_full['UID'] == 1381151] # filter for one site
soils_soil_df = soils_soil_df[soils_soil_df['depth_cm'] == '0-5']


soils = pd.DataFrame(soils_soil_df_full)
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
     
     



#define response variable
y = irrig_soil_df_test['WIMAS']

#define predictor variables
x = irrig_soil_df_test[['Aquacrop']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
#model = model.summary()

results_as_html = model.summary().tables[1].as_html()
pd.read_html(results_as_html, header=0, index_col=0)[0]




def Merge(dict1, dict2):
    for i in dict2.keys():
        dict1[i]=dict2[i]
    return dict1
     
# Driver code
dict1 = {'x': 10, 'y': 8}
dict2 = {'x': 6, 'b': 4}


dict3 = Merge(dict1, dict2)


print(dict3)

# merge two dictionaries
dict1 = {'key1':['value11','value12','value13'] , 'key2':['value21','value22','value23']}
dict2 = {'key1':['value14','value15'] , 'key2':['value24','value25']}

dict3 = {}
for key in set().union(dict1, dict2):
    if key in dict1: dict3.setdefault(key, []).extend(dict1[key])
    if key in dict2: dict3.setdefault(key, []).extend(dict2[key])

print(dict3)


# same projection
fields = fields.to_crs({'init': 'epsg:4326'}) 
thomas = thomas.to_crs({'init': 'epsg:4326'}) 


# visualise
fig, ax = plt.subplots(figsize = (10,10))

# Set the base as the fields shapefile
base = fields.plot(ax=ax, color='white', edgecolor='k', alpha=0.3)
ax.spines['top'].set_visible(False) # remove border around plot
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Plot the boundary and buffer on top of fields, setting the ax = base
# Set color of the line to black
thomas.boundary.plot(ax=base, color='blue')


# clip to fields in thomas
thomas.to_crs(fields.crs)
 = gpd.clip(fields, thomas)


def Gridmet(county_uid_soil_df, gridMet_soil_df):
    ids = gridMET.UID.isin(thomas_cornUID.UID)
    county_gridMET = gridMET[ids]  

    # calculate average values
    county_gridMET = county_gridMET.groupby(['date']).agg({'Tmin':'mean',
                                                               'Tmax':'mean',
                                                               'pr':'mean',
                                                               'eto':'mean'})
    county_gridMET.reset_index(inplace=True) # reset depth to variable                                                          
    
    # split date
    county_gridMET['day'] = county_gridMET['date'].dt.day
    county_gridMET['month'] = county_gridMET['date'].dt.month
    county_gridMET['year'] = county_gridMET['date'].dt.year

    county_gridMET = county_gridMET[["day", "month", "year", "Tmin", "Tmax", "pr", "eto"]] # select variables used in model
    return(county_gridMET)

# gridMet data for thomas
thomas_gridMET = Gridmet(thomas_cornUID, gridMET)




def Crop_fn(soil_df, crop):
    data = soil_df
    crop_soil_df = data[data['cropName'] == crop]
    return crop_soil_df

x = Crop_fn(crops_field, 'Corn')


# irrigation status
irrig_status = pd.read_csv(wd + '/data/agricLand/irrigationStatus/Kansas/FieldsAttributes_FieldsAroundSD6KS_Irrigation_AnnualAIM.csv')

irrig_status = irrig_status[irrig_status['Year'] >= 2000]
irrig_status = irrig_status[irrig_status['IrrigatedPrc'] > 0]




# soils

#### fields used to test function 
id_list = [1381151, 177799]
soils_soil_df = soils_soil_df_full[soils_soil_df_full['UID'].isin(id_list)] # filter for one site
soils = pd.DataFrame(soils_soil_df)
####


soils = pd.DataFrame(soils_soil_df_full)
soils = soils.assign(om = (10**(soils['logOm_%'])), # unit conversion
                     Ksat_cmHr = (10**(soils['logKsat_cmHr'])))

soils[['depth_min', 'depth_min']] = soils.depth_cm.str.split("-", expand = True) # split the depth variable


soils['depth_min'] = soils['depth_min'].astype(str).astype(int) ## create numeric depth variable

soils =soils.sort_values(by=['UID', 'depth_min'], ascending=True) # organise by UID and depth to enable correct looping in the next function

soils = soils[['UID', 'depth_cm', 'silt_prc', 'sand_prc',
               'clay_prc', 'thetaS_m3m3', 'thetaR_m3m3',
               'Ksat_cmHr', 'lambda', 'logHB_kPa', 'n',
               'logAlpha_kPa1', 'om']] # select variables



v = soils.groupby(['depth_cm']).agg({'silt_prc':'mean',
                                     'sand_prc':'mean',
                                     'clay_prc':'mean',
                                     'thetaS_m3m3':'mean',
                                     'thetaR_m3m3':'mean',
                                     'Ksat_cmHr':'mean',
                                     'lambda':'mean',
                                     'logHB_kPa':'mean',
                                     'n':'mean',
                                     'logAlpha_kPa1':'mean',
                                     'om':'mean'})

v.reset_index(inplace=True) # reset index to column

w = soils.groupby(['depth_cm'])['silt_prc', 'sand_prc',
               'clay_prc', 'thetaS_m3m3', 'thetaR_m3m3',
               'Ksat_cmHr', 'lambda', 'logHB_kPa', 'n',
               'logAlpha_kPa1','om'].mean().reset_index()

v = v.assign(UID = 'X')

# function to specify soils properties at the different depths
soils = soils.groupby(['depth_cm']).agg({'silt_prc':'mean',
                                     'sand_prc':'mean',
                                     'clay_prc':'mean',
                                     'thetaS_m3m3':'mean',
                                     'thetaR_m3m3':'mean',
                                     'Ksat_cmHr':'mean',
                                     'lambda':'mean',
                                     'logHB_kPa':'mean',
                                     'n':'mean',
                                     'logAlpha_kPa1':'mean',
                                     'om':'mean'})

soils.reset_index(inplace=True) # reset depth to variable
soils = soils.assign(UID = 'X') # place holder UID
soils[['depth_min', 'depth_min']] = soils.depth_cm.str.split("-", expand = True) # split the depth variable

soils['depth_min'] = soils['depth_min'].astype(str).astype(int) ## create numeric depth variable

soils =soils.sort_values(by=['UID', 'depth_min'], ascending=True) # organise by UID and depth to enable correct looping in the next function


def CustomS(soil_df):
    id_list = []
    custom_soil = []
    
    # calculate average soil characteristics
    soil_df = soil_df.groupby(['depth_cm']).agg({'silt_prc':['mean', 'std'],
                                         'sand_prc':['mean', 'std'],
                                         'clay_prc':['mean', 'std'],
                                         'thetaS_m3m3':['mean', 'std'],
                                         'thetaR_m3m3':['mean', 'std'],
                                         'Ksat_cmHr':['mean', 'std'],
                                         'lambda':['mean', 'std'],
                                         'logHB_kPa':['mean', 'std'],
                                         'n':['mean', 'std'],
                                         'logAlpha_kPa1':['mean', 'std'],
                                         'om':['mean', 'std']})

    soil_df.reset_index(inplace=True) # reset depth to variable
    soil_df = soil_df.assign(UID = 'X') # place holder UID
    soil_df[['depth_min', 'depth_min']] = soil_df.depth_cm.str.split("-", expand = True) # split the depth variable

    soil_df['depth_min'] = soil_df['depth_min'].astype(str).astype(int) ## create numeric depth variable

    soil_df = soil_df.sort_values(by=['UID', 'depth_min'], ascending=True) # organise by UID and depth to enable correct looping in the next function


    
    for i, row in soil_df.iterrows():   #soil_df.itertuples():
        ids = soil_df['UID'][i] #create soil_df with UID from the soils file used - fix this
        id_list.append(ids)
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
        #else:
            #break
    return id_list, custom_soil

# soil data for Thomas County *change name to thomas_soilData
soil_data = CustomS(soils)

thomas_id_list = soil_data[0]
thomas_custom_soil = soil_data[1]



def CustomSoil(soil_df, cropFieldMng_df): # cropFrieldMngt_df is the dataframe with the specified crop and irrigation scheme (rainfed or irrigated)
    id_list = []
    custom_soil = []
    
    soil_df = soil_df[soil_df['UID'].isin(cropFieldMng_df['UID'].tolist())]
    
    # calculate average soil characteristics
    
    # dataframe with the std
    soil_ave = soil_df.groupby(['depth_cm']).agg({'silt_prc':['mean', 'std'],
                                         'sand_prc':['mean', 'std'],
                                         'clay_prc':['mean', 'std'],
                                         'thetaS_m3m3':['mean', 'std'],
                                         'thetaR_m3m3':['mean', 'std'],
                                         'Ksat_cmHr':['mean', 'std'],
                                         'lambda':['mean', 'std'],
                                         'logHB_kPa':['mean', 'std'],
                                         'n':['mean', 'std'],
                                         'logAlpha_kPa1':['mean', 'std'],
                                         'om':['mean', 'std']})

    soil_ave.reset_index(inplace=True) 
    soil_ave[['depth_min', 'depth_min']] = soil_ave.depth_cm.str.split("-", expand = True) # split the depth variable

    soil_ave['depth_min'] = soil_ave['depth_min'].astype(str).astype(int) ## create numeric depth variable
    soil_ave = soil_ave.sort_values(by=['depth_min'], ascending=True) # organise by UID and depth to enable correct looping in the next function

    
    # dataframe with only the mean
    soil_df = soil_df.groupby(['depth_cm']).agg({'silt_prc':'mean',
                                         'sand_prc':'mean',
                                         'clay_prc':'mean',
                                         'thetaS_m3m3':'mean',
                                         'thetaR_m3m3':'mean',
                                         'Ksat_cmHr':'mean',
                                         'lambda':'mean',
                                         'logHB_kPa':'mean',
                                         'n':'mean',
                                         'logAlpha_kPa1':'mean',
                                         'om':'mean'})

    soil_df.reset_index(inplace=True) # reset depth to variable
    soil_df = soil_df.assign(UID = 'X') # place holder UID
    soil_df[['depth_min', 'depth_min']] = soil_df.depth_cm.str.split("-", expand = True) # split the depth variable

    soil_df['depth_min'] = soil_df['depth_min'].astype(str).astype(int) ## create numeric depth variable

    soil_df = soil_df.sort_values(by=['UID', 'depth_min'], ascending=True) # organise by UID and depth to enable correct looping in the next function


    
    for i, row in soil_df.iterrows():   #soil_df.itertuples():
        ids = soil_df['UID'][i] #create soil_df with UID from the soils file used - fix this
        id_list.append(ids)
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
        #else:
            #break
    return id_list, custom_soil, soil_ave


thomas_irrg_soil_data = CustomSoil(soils, thomas_irrig_cornUID)
thomas_rainfed_id_list = thomas_irrg_soil_data[0]
thomas_irrig_custom_soil = thomas_irrg_soil_data[1]
thomas_irrig_ave_soil = thomas_irrg_soil_data[2]


thomas_rainfed_soil_data = CustomSoil(soils, thomas_rainfed_cornUID)
thomas_rainfed_id_list = thomas_rainfed_soil_data[0]
thomas_rainfed_custom_soil = thomas_rainfed_soil_data[1]
thomas_rainfed_ave_soil = thomas_rainfed_soil_data[2]


#extract data from .gbd folder
import geopandas as gpd
from shapely import geometry
fields = gpd.read_file(wd + '/data/agricLand/Property Lines/Kansas/Fields_Around_SD6KS.shp')
     


fields_coords = shp.Reader(wd + '/data/agricLand/Property Lines/Kansas/Fields_Around_SD6KS.shp')
fields_coords.bbox
     

Visualize shapefile


fig, ax = plt.subplots(figsize = (10,10))
fields.plot(ax=ax)
plt.show()

# gmd4 
fields2 = gpd.read_file(wd + '/data/gmd4/gmd4.shp')
     


fields_coords2 = shp.Reader(wd + '/data/gmd4/gmd4.shp')
fields_coords2.bbox
     

#Visualize shapefile


fig, ax = plt.subplots(figsize = (10,10))
fields2.plot(ax=ax)
plt.show()


