#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:03:23 2023

@author: wayne
"""

!pip install aquacrop==2.2
!pip install numba==0.55
!pip install statsmodels==0.13.2
!pip install dfply
!pip install seaborn
!pip install pyshp
!pip install geopandas

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
import pickle 
#from dfply import *


import os 
import datetime
import shapefile as shp
import pandas as pd
import geopandas as gpd
from shapely import geometry



wd=getcwd() # set working directory
chdir(wd)

# SD-6 fields
fields = gpd.read_file(wd + '/data/agricLand/Property Lines/Kansas/Fields_Around_SD6KS.shp')

# soils
soils_df_full = pd.read_csv(wd + '/data/agricLand/soils/Soil_FieldsAroundSD6KS_POLARIS_AGrinstead_20220706.csv')

# KS counties
# https://catalog.data.gov/dataset/tiger-line-shapefile-2019-state-kansas-current-county-subdivision-state-based
# site had wrong coordinates

# https://public.opendatasoft.com/explore/dataset/us-county-boundaries/export/?disjunctive.statefp&disjunctive.countyfp&disjunctive.name&disjunctive.namelsad&disjunctive.stusab&disjunctive.state_name&refine.state_name=Kansas
county_boundary = gpd.read_file(wd + '/data/agricLand/countyBoundaries/Kansas/USCountyBoundaries_v1_WNdlovu_20221220.shp')



# crop data
crops_field = pd.read_csv(wd + '/data/agricLand/landUse_landCover/FieldsAttributesAroundSD6KS_LandCover_AnnualCDL.csv')

# county level yield for different crops (corn, wheat, sorghum and soybeans)
corn_gmd4 = pd.read_csv(wd + '/data/agricLand/yield/Kansas/corn.csv')
corn_gmd4['Irrig_status'] = np.where(corn_gmd4['Data Item'].str.contains("NON-IRRIGATED"), 'rainfed', 'irrigated')
corn_gmd4 = corn_gmd4[["Year", "State", "County", "Commodity", "Data Item", "Irrig_status", "Value"]]

wheat_gmd4 = pd.read_csv(wd + '/data/agricLand/yield/Kansas/winter_wheat.csv')
wheat_gmd4['Irrig_status'] = np.where(wheat_gmd4['Data Item'].str.contains("NON-IRRIGATED"), 'rainfed', 'irrigated')
wheat_gmd4 = wheat_gmd4[["Year", "State", "County", "Commodity", "Data Item", "Irrig_status", "Value"]]

soybeans_gmd4 = pd.read_csv(wd + '/data/agricLand/yield/Kansas/soybeans.csv')
soybeans_gmd4['Irrig_status'] = np.where(soybeans_gmd4['Data Item'].str.contains("NON-IRRIGATED"), 'rainfed', 'irrigated')
soybeans_gmd4 = soybeans_gmd4[["Year", "State", "County", "Commodity", "Data Item", "Irrig_status", "Value"]]

sorghum_gmd4 = pd.read_csv(wd + '/data/agricLand/yield/Kansas/sorghum.csv')
sorghum_gmd4['Irrig_status'] = np.where(sorghum_gmd4['Data Item'].str.contains("NON-IRRIGATED"), 'rainfed', 'irrigated')
sorghum_gmd4 = sorghum_gmd4[["Year", "State", "County", "Commodity", "Data Item", "Irrig_status", "Value"]]



# gridMET data
gridMET = pd.read_csv(wd + '/data/hydrometeorology/gridMET/gridMET_400m_2000_21.csv')



gridMET = gridMET.assign(Tmin = gridMET.tmmn-273.15,
                    Tmax = gridMET.tmmx-273.15,
                    date = pd.to_datetime(gridMET['date_ymd'], format='%Y%m%d'))


# separiting date
#gridMET = gridMET.assign(day =  gridMET['date'].dt.day,
                        # month = gridMET['date'].dt.month,
                         #year = gridMET['date'].dt.year)


# irrigation status
irrig_status = pd.read_csv(wd + '/data/agricLand/irrigationStatus/Kansas/FieldsAttributes_FieldsAroundSD6KS_Irrigation_AnnualAIM.csv')

irrig_status = irrig_status[irrig_status['Year'] >= 2000] # filter for years that match gridMET
irrig_status = irrig_status.assign(irrig_management = np.where(irrig_status['IrrigatedPrc'] <= 0.5, 'rainfed', 'irrigated')) # variable with irrigation status


# soils
soils = pd.DataFrame(soils_df_full)
soils = soils.assign(om = (10**(soils['logOm_%'])), # unit conversion
                     Ksat_cmHr = (10**(soils['logKsat_cmHr'])))


# filter for countys in GMD-4
gmd4 = ['Cheyenne', 'Rawlins', 'Decatur', 'Sherman', 'Gove', 'Thomas', 'Sheridan', 'Graham', 'Wallace', 'Logan'] # list of gmd4 coundties
county_boundary = county_boundary[county_boundary['name'].isin(gmd4)] # filter for counties inside gmd4 list

# county_boundary and fields layers have same projection
fields = fields.to_crs({'init': 'epsg:4326'})
county_boundary = county_boundary.to_crs({'init': 'epsg:4326'})

# Execute spatial join
fields_county = county_boundary.sjoin(fields, how="inner", predicate='intersects') # has 11351 obs vs fields with 11314 obs
dup_fields = fields_county[fields_county['UID'].duplicated(keep= False)] # fields that appear in 2 counties

fields_county = fields_county.drop_duplicates(subset=['UID']) # dropped dupilcated observations based on UID
fields_county = fields_county[['UID', 'name', 'geometry']] # select useful variables


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

# new column with crop name int he crops_fields (full df)
crops_field = crops_field.merge(crops, on='CropCode', how='left')

# one df with field and crop data
crops_irrig = crops_field.merge(irrig_status, on=['UID', 'Year'], how='inner') # create a dataframe which shows the field UID, irrigation status and the crop grown 
crops_irrig = crops_irrig.merge(fields_county, on=['UID'], how='inner') # merge to get df with the county

# filter for crops used in study
model_crops = ["Corn", "Sorghum", "Soybeans", "Winter Wheat"] 
crops_irrig = crops_irrig[crops_irrig['cropName'].isin(model_crops)]



grouped_info = [] # crop combination info (county, crop name and irrigation management (irrig or rainfed))
grouped_crop = [] # dataframe for each combination
gridMET_county = [] # daily average gridment for all fields in a county 
soil_county = []
custom_soil = []
for name in crops_irrig.groupby(['name', 'cropName', 'irrig_management']): #groupby county, crop name and irrigation management (irrig or rainfed)
    #grouped_crop.append(name)
    group_info = name[0] # collect the group combination information
    grouped_info.append(group_info)  
    group_df = name[1] # extract dataframe with the fields 
    group_df = group_df.drop_duplicates(subset=['UID']) # drop duplicated field IDS
    
    ids = gridMET.UID.isin(group_df.UID) # filter gridMET df for fields in the group
    county_gridMET = gridMET[ids] # dataframe with gridMET data for fields of interest
    #grouped_crop.append(county_gridMET) #
    
    # calculate daily average values within each county for a given combination of crop type and irrig management
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
    county_gridMET.columns = str(
        "Day Month Year MinTemp MaxTemp Precipitation ReferenceET").split()

    # put the weather dates into datetime format
    county_gridMET["Date"] = pd.to_datetime(county_gridMET[["Year", "Month", "Day"]])

    # drop the day month year columns
    county_gridMET = county_gridMET.drop(["Day", "Month", "Year"], axis=1)

    # set limit on ET0 to avoid divide by zero errors
    county_gridMET.ReferenceET.clip(lower=0.1, inplace=True)
    gridMET_county.append(county_gridMET)

    
    soil_df = soils[soils['UID'].isin(group_df['UID'].tolist())]
    
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
    soil_county.append(soil_ave)
    
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
    custom_soil.append(soil_df)
    
    
    
# export lists 

# gridMET
with open(r'./data/hydrometeorology/gridMET/ks_gridMET.pickle', 'wb') as met: 
    pickle.dump(gridMET_county, met) 

# soils
with open(r'./data/agricLand/soils/ks_soil.pickle', 'wb') as sl: 
    pickle.dump(custom_soil, sl) 
    
# mean soils by county
with open(r'./data/agricLand/soils/mean_ks_soil.pickle', 'wb') as mean_sl: 
    pickle.dump(soil_county, mean_sl) 

with open(r'./data/groupings/ks_ccm.pickle', 'wb') as info: # county crop managemnt
    pickle.dump(grouped_info, info) 

# re-save yield data - wrangled files will be used to validate model
corn_gmd4.to_csv(r'./data/agricLand/yield/Kansas/CornYield_GMD4_WNdlovu_v1_20230117.csv', sep=',', encoding='utf-8', header='true') 
wheat_gmd4.to_csv(r'./data/agricLand/yield/Kansas/WheatYield_GMD4_WNdlovu_v1_20230117.csv', sep=',', encoding='utf-8', header='true') 
soybeans_gmd4.to_csv(r'./data/agricLand/yield/Kansas/SoybeansYield_GMD4_WNdlovu_v1_20230117.csv', sep=',', encoding='utf-8', header='true') 
sorghum_gmd4.to_csv(r'./data/agricLand/yield/Kansas/SorghumYield_GMD4_WNdlovu_v1_20230117.csv', sep=',', encoding='utf-8', header='true') 



