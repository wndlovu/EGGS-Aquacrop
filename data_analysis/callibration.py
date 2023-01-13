#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 07:12:33 2022

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
from sklearn.metrics import r2_score


wd=getcwd() # set working directory
chdir(wd)

# SD-6 fields
fields = gpd.read_file(wd + '/data/agricLand/Property Lines/Kansas/Fields_Around_SD6KS.shp')


# KS counties
# https://catalog.data.gov/dataset/tiger-line-shapefile-2019-state-kansas-current-county-subdivision-state-based
# site had wrong coordinates

# https://public.opendatasoft.com/explore/dataset/us-county-boundaries/export/?disjunctive.statefp&disjunctive.countyfp&disjunctive.name&disjunctive.namelsad&disjunctive.stusab&disjunctive.state_name&refine.state_name=Kansas
county_boundary = gpd.read_file(wd + '/data/agricLand/countyBoundaries/Kansas/us-county-boundaries.shp')

# filter for thomas county
thomas = county_boundary[county_boundary['name'] == 'Thomas']

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
thomas_fields = gpd.clip(fields, thomas)
     

# crop data
crops_field = pd.read_csv(wd + '/data/agricLand/landUse_landCover/FieldsAttributesAroundSD6KS_LandCover_AnnualCDL.csv')


# Create lists of crop codes, crop names, and crop groups to match them up in a single dataframe
CropCode = [1, 4, 5, 6, 24, 61, 510, 520, 530, 540, 550, 560, 21, 23, 25, 27, 28, 29, 205, 2, 31, 33, 42, 43, 53, 
             26, 225, 226, 235, 236, 237, 238]

crop_name = ["Corn", 
             "Sorghum", 
             "Soybeans", 
             "Sunflower", 
             "Winter Wheat",
             "Fallow/Idle",
             "Alfalfa/Hay",
             "Grass/Shrub",
             "Forest", 
             "Wetland", 
             "Developed", 
             "Barren/Water", 
             "Barley", 
             "Spring Wheat", 
             "Other Small Grains", 
             "Rye", 
             "Oats", 
             "Millet", 
             "Triticale",
             "Cotton", 
             "Canola", 
             "Safflower", 
             "Dry Beans", 
             "Potatoes", 
             "Peas", 
             "Dbl Crop WinWht/Soybeans", 
             "Dbl_Crop_WinWht/Corn", 
             "Dbl Crop Oats/Corn", 
             "Dbl Crop Barley/Sorghum", 
             "Dbl Crop WinWht/Sorghum", 
             "Dbl_Crop_Barley/Corn", 
             "Dbl Crop WinWht/Cotton"]


crops = pd.DataFrame(zip(CropCode, crop_name), columns=['CropCode', 'cropName'])

# new column with crop name int he crops_fields
crops_field = crops_field.merge(crops, on='CropCode', how='left')

# filter for specific crop
def Crop_fn(df, crop):
    data = df
    crop_df = data[data['cropName'] == crop]
    return crop_df

corn_fields = Crop_fn(crops_field, 'Corn')

# filter for Thomas county
corn_thomas = thomas_fields.merge(corn_fields, on = 'UID', how = )













