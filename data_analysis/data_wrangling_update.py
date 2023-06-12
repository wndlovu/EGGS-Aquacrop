#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:03:23 2023

@author: wayne
"""

#!pip install aquacrop==2.2
#!pip install statsmodels==0.13.2
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
import math
from os import listdir
from os.path import isfile, join
import glob



wd=getcwd() # set working directory
chdir(wd)

# GMD4 fields
fields = gpd.read_file(wd + '/data/agricLand/Property Lines/Kansas/gmd4/ModelExtent_GMD4model_BWilson_v1_20230111.shp')

eggs/

# KS counties
# https://catalog.data.gov/dataset/tiger-line-shapefile-2019-state-kansas-current-county-subdivision-state-based
# site had wrong coordinates

# https://public.opendatasoft.com/explore/dataset/us-county-boundaries/export/?disjunctive.statefp&disjunctive.countyfp&disjunctive.name&disjunctive.namelsad&disjunctive.stusab&disjunctive.state_name&refine.state_name=Kansas

county_boundary = gpd.read_file(wd + '/data/agricLand/countyBoundaries/Kansas/USCountyBoundaries_v1_WNdlovu_20221220.shp')


# filter for countys in GMD-4
gmd4 = ['Cheyenne', 'Rawlins', 'Decatur', 'Sherman', 'Gove', 'Thomas', 'Sheridan', 'Graham', 'Wallace', 'Logan'] # list of gmd4 coundties
county_boundary = county_boundary[county_boundary['name'].isin(gmd4)] # filter for counties inside gmd4 list

# county_boundary and fields layers have same projection
fields = fields.to_crs({'init': 'epsg:4326'})
county_boundary = county_boundary.to_crs({'init': 'epsg:4326'})

# Execute spatial join
fields_county = county_boundary.sjoin(fields, how="inner", predicate='intersects') # has 93390 obs vs fields with 176407 obs
dup_fields = fields_county[fields_county['UID'].duplicated(keep= False)] # fields that appear in 2 counties

fields_county = fields_county.drop_duplicates(subset=['UID']) # dropped dupilcated observations based on UID
fields_county = fields_county[['UID', 'name', 'geometry']] # select useful variables


# crop data data from 2006 - 2020
#crops_irrigation = pd.read_csv(wd + '/data/water/Kansas/IrrigationDepth_GMD4_WNdlovu_v1_20230123.csv')
#crops_field = pd.read_csv(wd + '/data/agricLand/landUse_landCover/FieldsAttributesAroundSD6KS_LandCover_AnnualCDL.csv')

# merge irrigation status files

landUse_path = wd + '/data/agricLand/landUse_landCover/GMD4_OpenET-data-cdl_tables_lema' # landouse folder path
landUse_files = [f for f in listdir(landUse_path) if isfile(join(landUse_path, f))] # read files from folder
landUse_dfs_list = []  # List to store the dataframes

for file in landUse_files: # read files and save them a list of dataframes
    file_path = os.path.join(landUse_path, file)
    df = pd.read_csv(file_path) 
    landUse_dfs_list.append(df)

landUse_df = pd.concat(landUse_dfs_list, ignore_index=True) # merge dataframes in list 
landUse_df = landUse_df.drop(['system:index', 'status', '.geo'], axis=1) #drop system index variable

# replace all nan with 0 except for year and masterid cols
cols = ['Year', 'masterid']

landUse_df.loc[:, ~landUse_df.columns.isin(cols)] = landUse_df.loc[:, ~landUse_df.columns.isin(cols)].fillna(0)

# make year and masterid indexes
landUse_df = landUse_df.set_index(['Year', 'masterid'])
landUse_df['totalCover'] = landUse_df.sum(axis=1, numeric_only=True) #calculate (tatal land used) rowwise sum for all crops 

landUse_df = landUse_df[['1', '4', '5', '24', 'totalCover']] # select only the crops of interest "Corn", "Sorghum", "Soybeans", "Winter Wheat"]

# calculate % land cover
landUse_df['1'] = landUse_df['1']/landUse_df['totalCover']
landUse_df['4'] = landUse_df['4']/landUse_df['totalCover']
landUse_df['5'] = landUse_df['5']/landUse_df['totalCover']
landUse_df['24'] = landUse_df['24']/landUse_df['totalCover']

landUse_df = landUse_df.drop(['totalCover'], axis=1) # drop totalCover 
landUse_df = landUse_df.melt(ignore_index=False) #pivot longer on the pct covers
landUse_df = landUse_df.reset_index()
landUse_df = landUse_df.rename(columns={'masterid': 'UID', 'variable': 'CropCode', 'value': 'pctcov'}) # rename columns
landUse_df = landUse_df[landUse_df['UID'] != 0] # remove place holder row with uid of 0
landUse_df['CropCode'] = landUse_df['CropCode'].astype(int)
#x = landUse_df.head(5)

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

# irrigation status
irrig_status = pd.read_csv(wd + '/data/agricLand/irrigationStatus/Kansas/AIM_HPA_GMD4_2000_2020_Deines_etal_RSE_v01_samsPolygons.csv')
irrig_status['0'] = irrig_status['0'].fillna(0) # replace nan with 0
irrig_status['1'] = irrig_status['1'].fillna(0) # replace nan with 0
irrig_status['IrrigatedPrc'] = 1 - (irrig_status['0'] / (irrig_status['0'] + irrig_status['1']))
irrig_status = irrig_status.assign(irrig_management = np.where(irrig_status['IrrigatedPrc'] <= 0.5, 'rainfed', 'irrigated')) # variable with irrigation status
irrig_status = irrig_status[['UID', 'year', 'IrrigatedPrc', 'irrig_management']]
irrig_status = irrig_status.rename(columns={'year': 'Year'})
#y = irrig_status.tail(5)


# Create lists of crop codes, crop names, and crop groups to match them up in a single dataframe
CropCode = [1, 4, 5, 24]

crop_name = ["Corn", 
             "Sorghum", 
             "Soybeans",  
             "Winter Wheat"]


crops = pd.DataFrame(zip(CropCode, crop_name), columns=['CropCode', 'cropName'])

# new column with crop name int he crops_fields (full df)
crops_field = landUse_df.merge(crops, on='CropCode', how='left')

# one df with field and crop data
crops_irrig = crops_field.merge(irrig_status, on=['UID', 'Year'], how='inner') # create a dataframe which shows the field UID, irrigation status and the crop grown 
crops_irrig = crops_irrig.merge(fields_county, on=['UID'], how='inner') # merge to get df with the county


# make dataframe with county+crop+irrig_mngt polygon
from shapely.ops import unary_union

#mergedPolys = unary_union(polys)

crops_irrig_gdf = gpd.GeoDataFrame(crops_irrig) # transform to geodataframe
crops_irrig_gdf2 = crops_irrig_gdf.groupby(['name', 'cropName', 'irrig_management']).agg({'geometry': unary_union}) # combine polygons



x = x.head(10)
crops_irrig_gdf2 = crops_irrig_gdf.dissolve(['name', 'cropName', 'irrig_management'])  #.agg({'geometry': unary_union})

dfds = x.groupby(['name', 'cropName', 'irrig_management']).agg({'geometry': unary_union})



from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

polygon1 = Polygon([(0,5), (1,1), (3,0),])

fig, ax = plt.subplots(1,1)

ax.add_patch(x[['geometry']])

dfds.plot()



import geopandas as gpd
from shapely.geometry import box

gdf1 = gpd.GeoDataFrame(
    [
        {"col1": "a", "col2": 1, "geometry": box(0, 0, 1, 1)},
        {"col1": "a", "col2": 2, "geometry": box(0, 1, 1, 2)},
        {"col1": "b", "col2": 3, "geometry": box(2, 1, 3, 2)},
    ],
    geometry="geometry",
)


print(f"{gdf1._geometry_column_name=}")

unary_union = lambda x: x.unary_union  # workaround https://github.com/geopandas/geopandas/issues/2171
gdf2 = gdf1.groupby("col1").agg({"geometry": unary_union, "col2": sum}).reset_index(drop=False)

gdf2 = gdf1.dissolve("col1")

print(f"{gdf2._geometry_column_name=}")



crops_irrig = crops_irrig[['UID', 'Year', 'cropName', 'irrig_management', 'name']]              
#z = crops_irrig.head(5)

# gridMET data
gridMET = pd.read_csv(wd + '/data/hydrometeorology/gridMET/gridMET_4000m_GMD4_WNdlovu_v1_20230226.csv')



gridMET = gridMET.assign(Tmin = gridMET.tmmn-273.15,
                    Tmax = gridMET.tmmx-273.15,
                    date = pd.to_datetime(gridMET['date_ymd'], format='%Y%m%d'))

gridMET = gridMET.merge(fields_county, on=['UID'], how='inner') # get county name
#county_gridMET = county_gridMET[[]]
# separiting date
#gridMET = gridMET.assign(day =  gridMET['date'].dt.day,
                        # month = gridMET['date'].dt.month,
                         #year = gridMET['date'].dt.year)

# Canopy cover   sd6 for now
lai = pd.read_csv(wd + '/data/lai/LAI_4000m_SD6_WNdlovu_2000.csv')

lai = lai.assign(cc = (100.5*(1-np.exp(-0.6*lai['Lai']))**1.2)) # calc cc Hsiao et al. (2009)



lai = lai.assign(date = pd.to_datetime(lai['date_ymd'], format='%Y%m%d'))

county_lai = lai.merge(fields_county, on=['UID'], how='inner') # get county name
county_lai = county_lai[['UID', 'date_ymd', 'cc', 'name']]


# average county soils
sp_path = wd+ '/data/agricLand/soils/gmd4_soils/CountySoils' # soil profile folder path
sp_files = glob.glob(sp_path + '/*.csv') # read files from folder
sp_dfs_list = []  # List to store the dataframes

for file in sp_files: # read files and save them a list of dataframes
    df = pd.read_csv(file) 
    df = df.assign(om = (10**(df['log_om'])), # unit conversion
                         ksat = (10**(df['log_ksat'])))
    sp_dfs_list.append(df)

soils = pd.concat(sp_dfs_list, ignore_index=True)
soils = soils.rename(columns={'uid': 'UID'})
#f = df.head(5)



os.path.expanduser("~/EGGS-Aquacrop") 

grouped_info = [] # crop combination info (county, crop name and irrigation management (irrig or rainfed))
grouped_crop = [] # dataframe for each combination
gridMET_county = [] # daily average gridment for all fields in a county 
soil_county = []
custom_soil = []
lai_county = []

for name in crops_irrig.groupby(['name', 'cropName', 'irrig_management']):
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
    
        ids2 = lai.UID.isin(group_df.UID)
        county_lai = lai[ids2] 
        county_lai = county_lai.groupby(['date']).agg({'cc':'mean'})
        county_lai.reset_index(inplace=True)
        lai_county.append(county_lai) # reset depth to variable                                                          
        




custom_soil = []
for i in soil_list:
    for name in crops_irrig.groupby(['name', 'cropName', 'irrig_management']):

        #print(i)
        group_info = name[0] # collect the group combination information
        grouped_info.append(group_info)  
        group_df = name[1] # extract dataframe with the fields 
        group_df = group_df.drop_duplicates(subset=['UID']) # drop duplicated field IDS
        
        print(i)
        
        
        ids = i.uid.isin(group_df.UID) # filter gridMET df for fields in the group
        county_soil = i[ids] 
        #grouped_crop.append(name)
        group_info = name[0] # collect the group combination information   
        # dataframe with only the mean
        soil_df = county_soil.groupby(['depth','county' ]).agg({'silt_prc':'mean',
                                             'sand_prc':'mean',
                                             'clay_prc':'mean',
                                             'theta_s':'mean',
                                             'theta_r':'mean',
                                             'Ksat_cmHr':'mean',
                                             'lambda':'mean',
                                             'log_hb':'mean',
                                             'n':'mean',
                                             'log_alpha':'mean',
                                             'om':'mean'})
    
        soil_df.reset_index(inplace=True) # reset depth to variable
        soil_df = soil_df.assign(UID = 'X') # place holder UID
        soil_df[['depth_min', 'depth_min']] = soil_df.depth.str.split("-", expand = True) # split the depth variable
    
        soil_df['depth_min'] = soil_df['depth_min'].astype(str).astype(int) ## create numeric depth variable
    
        soil_df = soil_df.sort_values(by=['UID', 'depth_min'], ascending=True) # organise by UID and depth to enable correct looping in the next function
        custom_soil.append(soil_df)







grouped_info = [] # crop combination info (county, crop name and irrigation management (irrig or rainfed))
grouped_crop = [] # dataframe for each combination
gridMET_county = [] # daily average gridment for all fields in a county 
soil_county = []
custom_soil = []
for name in crops_irrig.groupby(['name', 'cropName', 'irrig_management']): #groupby county, crop name and irrigation management (irrig or rainfed)

    #print(name)
    #grouped_crop.append(name)
    group_info = name[0] # collect the group combination information
    grouped_info.append(group_info)  
    group_df = name[1] # extract dataframe with the fields 
    group_df = group_df.drop_duplicates(subset=['UID']) # drop duplicated field IDS
    
    #ids = gridMET.UID.isin(group_df.UID) # filter gridMET df for fields in the group
    county_gridMET = gridMET[gridMET['UID'].isin(group_df['UID'])]# dataframe with gridMET data for fields of interest
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

    
    soil_df = soils[soils['UID'].isin(group_df['UID'])]    #.tolist())]
    
    # calculate average soil characteristics
    
    # dataframe with only the mean
    soil_df = soil_df.groupby(['depth']).agg({'silt_prc':'mean',
                                         'sand_prc':'mean',
                                         'clay_prc':'mean',
                                         'theta_s':'mean',
                                         'theta_r':'mean',
                                         'ksat':'mean',
                                         'lambda':'mean',
                                         'log_hb':'mean',
                                         'n':'mean',
                                         'log_alpha':'mean',
                                         'om':'mean'})
    
    

    soil_df.reset_index(inplace=True) # reset depth to variable
    soil_df = soil_df.assign(UID = 'X') # place holder UID
    soil_df[['depth_min', 'depth_min']] = soil_df.depth.str.split("-", expand = True) # split the depth variable

    soil_df['depth_min'] = soil_df['depth_min'].astype(str).astype(int) ## create numeric depth variable

    soil_df = soil_df.sort_values(by=['UID', 'depth_min'], ascending=True) # organise by UID and depth to enable correct looping in the next function
    custom_soil.append(soil_df)
    


#creating dic of datafames
soil_gridMET_gmd4 = []
for name in crops_irrig.groupby(['name', 'cropName', 'irrig_management']): #groupby county, crop name and irrigation management (irrig or rainfed)

    #print(name)
    #grouped_crop.append(name)
    group_info = name[0] # collect the group combination information
    grouped_info.append(group_info)  
    group_df = name[1] # extract dataframe with the fields 
    group_df = group_df.drop_duplicates(subset=['UID']) # drop duplicated field IDS
    
    #ids = gridMET.UID.isin(group_df.UID) # filter gridMET df for fields in the group
    county_gridMET = gridMET[gridMET['UID'].isin(group_df['UID'])]# dataframe with gridMET data for fields of interest
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
    #gridMET_county.append(county_gridMET)

    
    soil_df = soils[soils['UID'].isin(group_df['UID'])]    #.tolist())]
    
    # calculate average soil characteristics
    
    # dataframe with only the mean
    soil_df = soil_df.groupby(['depth']).agg({'silt_prc':'mean',
                                         'sand_prc':'mean',
                                         'clay_prc':'mean',
                                         'theta_s':'mean',
                                         'theta_r':'mean',
                                         'ksat':'mean',
                                         'lambda':'mean',
                                         'log_hb':'mean',
                                         'n':'mean',
                                         'log_alpha':'mean',
                                         'om':'mean'})
    
    

    soil_df.reset_index(inplace=True) # reset depth to variable
    soil_df = soil_df.assign(UID = 'X') # place holder UID
    soil_df[['depth_min', 'depth_min']] = soil_df.depth.str.split("-", expand = True) # split the depth variable

    soil_df['depth_min'] = soil_df['depth_min'].astype(str).astype(int) ## create numeric depth variable

    soil_df = soil_df.sort_values(by=['UID', 'depth_min'], ascending=True) # organise by UID and depth to enable correct looping in the next function
    #custom_soil.append(soil_df)

    # create 

    # list of data frames
    soil_gridmet = [group_info, county_gridMET, soil_df] 

    # dictionary to save data frames
    full_dict={} 

    for key, value in enumerate(dataframes):    
      full_dict[key] = value # assigning data frame from list to key in dictionary


    # save dics to list
    soil_gridMET_gmd4.append(full_dict)

/EGGS/Data/Agriculture-Land/Soils/Kansas/GMD4/POLARISSoils_Processed_GMD4/POLARISMergedSoils_GMD4/POLARISSoilsClay_GMD4_WNdlovu_v1_20230228.csv



group_df2 = crops_irrig[crops_irrig['cropName'] == 'Corn']
group_df2 = group_df2[group_df2['name'] == 'Sherman']
group_df2 = group_df2[group_df2['irrig_management'] == 'rainfed']



soil_df = soils[soils['UID'].isin(group_df2['UID'])]    #.tolist())]

# calculate average soil characteristics

# dataframe with only the mean
soil_df = soil_df.groupby(['depth']).agg({'silt_prc':'mean',
                                     'sand_prc':'mean',
                                     'clay_prc':'mean',
                                     'theta_s':'mean',
                                     'theta_r':'mean',
                                     'ksat':'mean',
                                     'lambda':'mean',
                                     'log_hb':'mean',
                                     'n':'mean',
                                     'log_alpha':'mean',
                                     'om':'mean'})



soil_df.reset_index(inplace=True) # reset depth to variable
soil_df = soil_df.assign(UID = 'X') # place holder UID
soil_df[['depth_min', 'depth_min']] = soil_df.depth.str.split("-", expand = True) # split the depth variable

soil_df['depth_min'] = soil_df['depth_min'].astype(str).astype(int) 
soil_df = soil_df.sort_values(by=['UID', 'depth_min'], ascending=True)


    
 
#creating dic of datafames
# import the pandas library to make the data frame
import pandas as pd 

df1 = pd.DataFrame({
"Name": ["Shahroz", "Samad", "Usama"],
"Age": [22, 35, 58]    
})

df2 = pd.DataFrame({
"Class": ["Chemistry","Physics","Biology"],
"Students": [30, 35, 40]    
})

# list of data frames
dataframes = [df1, df2] 

# dictionary to save data frames
frames={} 

for key, value in enumerate(dataframes):    
  frames[key] = value # assigning data frame from list to key in dictionary
  print("key: ", key)
  print(frames[key], "\n")

# access to one data frame by key
print("Accessing the dataframe against key 0 \n", end ="")
print(frames[0])

# access to only column of specific data frame through dictionary
print("\nAccessing the first column of dataframe against key 1\n", end ="")
print(frames[1]["Class"])    
 
    
 
# function that takes dict as argument
def test(x):
    prod = x[0]['Age']* x[1]['Students']
    
    return(prod)
 
    
test(frames)
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
    

with open(r'./data/hydrometeorology/lai.pickle', 'wb') as laidf: # county crop managemnt
    pickle.dump(lai_county, laidf) 
      


with open(r'./data/crops_irrig.pickle', 'wb') as ci: # county crop managemnt
    pickle.dump(crops_irrig, ci) 
    
crops_irrig.to_csv(r'./data/crops_irrig.csv', sep=',', encoding='utf-8', header='true') 
    

# re-save yield data - wrangled files will be used to validate model
corn_gmd4.to_csv(r'./data/agricLand/yield/Kansas/CornYield_GMD4_WNdlovu_v1_20230117.csv', sep=',', encoding='utf-8', header='true') 
wheat_gmd4.to_csv(r'./data/agricLand/yield/Kansas/WheatYield_GMD4_WNdlovu_v1_20230117.csv', sep=',', encoding='utf-8', header='true') 
soybeans_gmd4.to_csv(r'./data/agricLand/yield/Kansas/SoybeansYield_GMD4_WNdlovu_v1_20230117.csv', sep=',', encoding='utf-8', header='true') 
sorghum_gmd4.to_csv(r'./data/agricLand/yield/Kansas/SorghumYield_GMD4_WNdlovu_v1_20230117.csv', sep=',', encoding='utf-8', header='true') 



