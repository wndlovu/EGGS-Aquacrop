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
soils_df_full = pd.read_csv(wd + '/data/agricLand/soils/Soil_FieldsAroundSD6KS_POLARIS_AGrinstead_20220706.csv')
soils_df = soils_df_full[soils_df_full['UID'] == 1381151] # filter for one site
soils_df = soils_df[soils_df['depth_cm'] == '0-5']


soils = pd.DataFrame(soils_df_full)
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
y = irrig_df_test['WIMAS']

#define predictor variables
x = irrig_df_test[['Aquacrop']]

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



def Crop_fn(df, crop):
    data = df
    crop_df = data[data['cropName'] == crop]
    return crop_df

x = Crop_fn(crops_field, 'Corn')


# soils = specify the soil characterics for each depth
custom = Soil('custom',cn=46,rew=7, dz=[0.025]*2+[0.05]*2+[0.075]*2+[0.15]*2+[0.2]*2+[0.5]*2)

custom.add_layer(thickness=0.05,thWP=0.24,
                 thFC=0.40,thS=0.50,Ksat=155,
                 penetrability=100)

custom.add_layer_from_texture(thickness=.15,
                              Sand=10,Clay=35,
                              OrgMat=2.5,penetrability=100)

custom.add_layer(thickness=0.3,thWP=0.24,
                 thFC=0.70,thS=0.50,Ksat=155,
                 penetrability=80)

custom.add_layer(thickness=0.3,thWP=0.24,
                 thFC=0.70,thS=0.50,Ksat=155,
                 penetrability=80)

custom.add_layer(thickness=0.4,thWP=0.24,
                 thFC=0.70,thS=0.50,Ksat=155,
                 penetrability=80)

custom.add_layer(thickness=1,thWP=0.24,
                 thFC=0.70,thS=0.50,Ksat=155,
                 penetrability=80)

a1 = custom.profile







     
     
     
 