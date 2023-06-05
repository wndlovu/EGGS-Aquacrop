# -*- coding: utf-8 -*-
"""

"""
##Python 3.9.13
!pip install pyshp
!pip install geopandas

import os 
import numpy as np
import pandas as pd
from os import chdir, getcwd
import shapefile as shp
import geopandas as gpd
from shapely import geometry


# Getting current work directory
wd=getcwd() # set working directory
chdir(wd)


# KS counties
# https://public.opendatasoft.com/explore/dataset/us-county-boundaries/export/?disjunctive.statefp&disjunctive.countyfp&disjunctive.name&disjunctive.namelsad&disjunctive.stusab&disjunctive.state_name&refine.state_name=Kansas
county_boundary = gpd.read_file(wd + '/data/agricLand/countyBoundaries/Kansas/USCountyBoundaries_v1_WNdlovu_20221220.shp')

# gmd4 fields
gmd4_fields = gpd.read_file(wd + '/data/agricLand/Property Lines/Kansas/gmd4/ModelExtent_GMD4model_BWilson_v1_20230111.shp')


# soil paths

sp = wd+'/data/agricLand/soils/gmd4_soils'

# Create filepaths for all soil data
log_alpha_fp = sp+"/logAlpha"
clay_fp = sp+"/Clay"
log_hb_fp = sp+"/logHB"
log_om_fp = sp+"/logOm"
log_ksat_fp = sp+"/logKsat"
lambda_fp = sp+"/Lambda"
n_fp = sp+"/N"
sand_fp = sp+"/Sand"
silt_fp = sp+"/Silt"
theta_r_fp = sp+"/Theta_R"
theta_s_fp = sp+"/Theta_S"

## Import each soil type and depth

### Log Alpha


# Import alpha
os.chdir(log_alpha_fp)
log_alpha_0_5 = pd.read_csv('logAlpha_0_5_POLARIS_FieldsNoDups.csv')
log_alpha_5_15 = pd.read_csv('logAlpha_5_15_POLARIS_FieldsNoDups.csv')
log_alpha_15_30 = pd.read_csv('logAlpha_15_30_POLARIS_FieldsNoDups.csv')
log_alpha_30_60 = pd.read_csv('logAlpha_30_60_POLARIS_FieldsNoDups.csv')
log_alpha_60_100 = pd.read_csv('logAlpha_60_100_POLARIS_FieldsNoDups.csv')
log_alpha_100_200 = pd.read_csv('logAlpha_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
log_alpha_0_5['depth_cm'] = '0-5'
log_alpha_0_5 = log_alpha_0_5[['UID', 'depth_cm', 'mean']]

log_alpha_5_15['depth_cm'] = '5-15'
log_alpha_5_15 = log_alpha_5_15[['UID', 'depth_cm', 'mean']]

log_alpha_15_30['depth_cm'] = '15-30'
log_alpha_15_30 = log_alpha_15_30[['UID', 'depth_cm', 'mean']]

log_alpha_30_60['depth_cm'] = '30-60'
log_alpha_30_60 = log_alpha_30_60[['UID', 'depth_cm', 'mean']]

log_alpha_60_100['depth_cm'] = '60-100'
log_alpha_60_100 = log_alpha_60_100[['UID', 'depth_cm', 'mean']]

log_alpha_100_200['depth_cm'] = '100-200'
log_alpha_100_200 = log_alpha_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
log_alpha = pd.concat([log_alpha_0_5, log_alpha_5_15, log_alpha_15_30, log_alpha_30_60, log_alpha_60_100, log_alpha_100_200])
log_alpha = log_alpha.rename(columns = {'mean':'logAlpha_kPa1'})

### Clay"""

# Import clay
os.chdir(clay_fp)
clay_0_5 = pd.read_csv('Clay_0_5_POLARIS_FieldsNoDups.csv')
clay_5_15 = pd.read_csv('Clay_5_15_POLARIS_FieldsNoDups.csv')
clay_15_30 = pd.read_csv('Clay_15_30_POLARIS_FieldsNoDups.csv')
clay_30_60 = pd.read_csv('Clay_30_60_POLARIS_FieldsNoDups.csv')
clay_60_100 = pd.read_csv('Clay_60_100_POLARIS_FieldsNoDups.csv')
clay_100_200 = pd.read_csv('Clay_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
clay_0_5['depth_cm'] = '0-5'
clay_0_5 = clay_0_5[['UID', 'depth_cm', 'mean']]

clay_5_15['depth_cm'] = '5-15'
clay_5_15 = clay_5_15[['UID', 'depth_cm', 'mean']]

clay_15_30['depth_cm'] = '15-30'
clay_15_30 = clay_15_30[['UID', 'depth_cm', 'mean']]

clay_30_60['depth_cm'] = '30-60'
clay_30_60 = clay_30_60[['UID', 'depth_cm', 'mean']]

clay_60_100['depth_cm'] = '60-100'
clay_60_100 = clay_60_100[['UID', 'depth_cm', 'mean']]

clay_100_200['depth_cm'] = '100-200'
clay_100_200 = clay_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
clay = pd.concat([clay_0_5, clay_5_15, clay_15_30, clay_30_60, clay_60_100, clay_100_200])
clay = clay.rename(columns = {'mean':'clay_prc'})

### Log HB"""

# Import log_hb
os.chdir(log_hb_fp)
log_hb_0_5 = pd.read_csv('logHB_0_5_POLARIS_FieldsNoDups.csv')
log_hb_5_15 = pd.read_csv('logHB_5_15_POLARIS_FieldsNoDups.csv')
log_hb_15_30 = pd.read_csv('logHB_15_30_POLARIS_FieldsNoDups.csv')
log_hb_30_60 = pd.read_csv('logHB_30_60_POLARIS_FieldsNoDups.csv')
log_hb_60_100 = pd.read_csv('logHB_60_100_POLARIS_FieldsNoDups.csv')
log_hb_100_200 = pd.read_csv('logHB_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
log_hb_0_5['depth_cm'] = '0-5'
log_hb_0_5 = log_hb_0_5[['UID', 'depth_cm', 'mean']]

log_hb_5_15['depth_cm'] = '5-15'
log_hb_5_15 = log_hb_5_15[['UID', 'depth_cm', 'mean']]

log_hb_15_30['depth_cm'] = '15-30'
log_hb_15_30 = log_hb_15_30[['UID', 'depth_cm', 'mean']]

log_hb_30_60['depth_cm'] = '30-60'
log_hb_30_60 = log_hb_30_60[['UID', 'depth_cm', 'mean']]

log_hb_60_100['depth_cm'] = '60-100'
log_hb_60_100 = log_hb_60_100[['UID', 'depth_cm', 'mean']]

log_hb_100_200['depth_cm'] = '100-200'
log_hb_100_200 = log_hb_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
log_hb = pd.concat([log_hb_0_5, log_hb_5_15, log_hb_15_30, log_hb_30_60, log_hb_60_100, log_hb_100_200])
log_hb = log_hb.rename(columns = {'mean':'logHB_kPa'})


### Log OM"""

# Import log_hb
os.chdir(log_om_fp)
log_om_0_5 = pd.read_csv('logOm_0_5_POLARIS_FieldsNoDups.csv')
log_om_5_15 = pd.read_csv('logOm_5_15_POLARIS_FieldsNoDups.csv')
log_om_15_30 = pd.read_csv('logOm_15_30_POLARIS_FieldsNoDups.csv')
log_om_30_60 = pd.read_csv('logOm_30_60_POLARIS_FieldsNoDups.csv')
log_om_60_100 = pd.read_csv('logOm_60_100_POLARIS_FieldsNoDups.csv')
log_om_100_200 = pd.read_csv('logOm_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
log_om_0_5['depth_cm'] = '0-5'
log_om_0_5 = log_om_0_5[['UID', 'depth_cm', 'mean']]

log_om_5_15['depth_cm'] = '5-15'
log_om_5_15 = log_om_5_15[['UID', 'depth_cm', 'mean']]

log_om_15_30['depth_cm'] = '15-30'
log_om_15_30 = log_om_15_30[['UID', 'depth_cm', 'mean']]

log_om_30_60['depth_cm'] = '30-60'
log_om_30_60 = log_om_30_60[['UID', 'depth_cm', 'mean']]

log_om_60_100['depth_cm'] = '60-100'
log_om_60_100 = log_om_60_100[['UID', 'depth_cm', 'mean']]

log_om_100_200['depth_cm'] = '100-200'
log_om_100_200 = log_om_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
log_om = pd.concat([log_om_0_5, log_om_5_15, log_om_15_30, log_om_30_60, log_om_60_100, log_om_100_200])
log_om = log_om.rename(columns = {'mean':'logOm_kPa'})


### Log Ksat"""

# Import ksat
os.chdir(log_ksat_fp)
log_ksat_0_5 = pd.read_csv('logKsat_0_5_POLARIS_FieldsNoDups.csv')
log_ksat_5_15 = pd.read_csv('logKsat_5_15_POLARIS_FieldsNoDups.csv')
log_ksat_15_30 = pd.read_csv('logKsat_15_30_POLARIS_FieldsNoDups.csv')
log_ksat_30_60 = pd.read_csv('logKsat_30_60_POLARIS_FieldsNoDups.csv')
log_ksat_60_100 = pd.read_csv('logKsat_60_100_POLARIS_FieldsNoDups.csv')
log_ksat_100_200 = pd.read_csv('logKsat_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
log_ksat_0_5['depth_cm'] = '0-5'
log_ksat_0_5 = log_ksat_0_5[['UID', 'depth_cm', 'mean']]

log_ksat_5_15['depth_cm'] = '5-15'
log_ksat_5_15 = log_ksat_5_15[['UID', 'depth_cm', 'mean']]

log_ksat_15_30['depth_cm'] = '15-30'
log_ksat_15_30 = log_ksat_15_30[['UID', 'depth_cm', 'mean']]

log_ksat_30_60['depth_cm'] = '30-60'
log_ksat_30_60 = log_ksat_30_60[['UID', 'depth_cm', 'mean']]

log_ksat_60_100['depth_cm'] = '60-100'
log_ksat_60_100 = log_ksat_60_100[['UID', 'depth_cm', 'mean']]

log_ksat_100_200['depth_cm'] = '100-200'
log_ksat_100_200 = log_ksat_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
log_ksat = pd.concat([log_ksat_0_5, log_ksat_5_15, log_ksat_15_30, log_ksat_30_60, log_ksat_60_100, log_ksat_100_200])
log_ksat = log_ksat.rename(columns = {'mean':'logKsat_cmHr'})

### Lambda"""

# Import lambda
os.chdir(lambda_fp)
lambda_0_5 = pd.read_csv('Lambda_0_5_POLARIS_FieldsNoDups.csv')
lambda_5_15 = pd.read_csv('Lambda_5_15_POLARIS_FieldsNoDups.csv')
lambda_15_30 = pd.read_csv('Lambda_15_30_POLARIS_FieldsNoDups.csv')
lambda_30_60 = pd.read_csv('Lambda_30_60_POLARIS_FieldsNoDups.csv')
lambda_60_100 = pd.read_csv('Lambda_60_100_POLARIS_FieldsNoDups.csv')
lambda_100_200 = pd.read_csv('Lambda_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
lambda_0_5['depth_cm'] = '0-5'
lambda_0_5 = lambda_0_5[['UID', 'depth_cm', 'mean']]

lambda_5_15['depth_cm'] = '5-15'
lambda_5_15 = lambda_5_15[['UID', 'depth_cm', 'mean']]

lambda_15_30['depth_cm'] = '15-30'
lambda_15_30 = lambda_15_30[['UID', 'depth_cm', 'mean']]

lambda_30_60['depth_cm'] = '30-60'
lambda_30_60 = lambda_30_60[['UID', 'depth_cm', 'mean']]

lambda_60_100['depth_cm'] = '60-100'
lambda_60_100 = lambda_60_100[['UID', 'depth_cm', 'mean']]

lambda_100_200['depth_cm'] = '100-200'
lambda_100_200 = lambda_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
Lambda = pd.concat([lambda_0_5, lambda_5_15, lambda_15_30, lambda_30_60, lambda_60_100, lambda_100_200])
Lambda = Lambda.rename(columns = {'mean':'lambda'})

### N"""

# Import n
os.chdir(n_fp)
n_0_5 = pd.read_csv('N_0_5_POLARIS_FieldsNoDups.csv')
n_5_15 = pd.read_csv('N_5_15_POLARIS_FieldsNoDups.csv')
n_15_30 = pd.read_csv('N_15_30_POLARIS_FieldsNoDups.csv')
n_30_60 = pd.read_csv('N_30_60_POLARIS_FieldsNoDups.csv')
n_60_100 = pd.read_csv('N_60_100_POLARIS_FieldsNoDups.csv')
n_100_200 = pd.read_csv('N_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
n_0_5['depth_cm'] = '0-5'
n_0_5 = n_0_5[['UID', 'depth_cm', 'mean']]

n_5_15['depth_cm'] = '5-15'
n_5_15 = n_5_15[['UID', 'depth_cm', 'mean']]

n_15_30['depth_cm'] = '15-30'
n_15_30 = n_15_30[['UID', 'depth_cm', 'mean']]

n_30_60['depth_cm'] = '30-60'
n_30_60 = n_30_60[['UID', 'depth_cm', 'mean']]

n_60_100['depth_cm'] = '60-100'
n_60_100 = n_60_100[['UID', 'depth_cm', 'mean']]

n_100_200['depth_cm'] = '100-200'
n_100_200 = n_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
n = pd.concat([n_0_5, n_5_15, n_15_30, n_30_60, n_60_100, n_100_200])
n = n.rename(columns = {'mean':'n'})

### Sand"""

# Import sand
os.chdir(sand_fp)
sand_0_5 = pd.read_csv('Sand_0_5_POLARIS_FieldsNoDups.csv')
sand_5_15 = pd.read_csv('Sand_5_15_POLARIS_FieldsNoDups.csv')
sand_15_30 = pd.read_csv('Sand_15_30_POLARIS_FieldsNoDups.csv')
sand_30_60 = pd.read_csv('Sand_30_60_POLARIS_FieldsNoDups.csv')
sand_60_100 = pd.read_csv('Sand_60_100_POLARIS_FieldsNoDups.csv')
sand_100_200 = pd.read_csv('Sand_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
sand_0_5['depth_cm'] = '0-5'
sand_0_5 = sand_0_5[['UID', 'depth_cm', 'mean']]

sand_5_15['depth_cm'] = '5-15'
sand_5_15 = sand_5_15[['UID', 'depth_cm', 'mean']]

sand_15_30['depth_cm'] = '15-30'
sand_15_30 = sand_15_30[['UID', 'depth_cm', 'mean']]

sand_30_60['depth_cm'] = '30-60'
sand_30_60 = sand_30_60[['UID', 'depth_cm', 'mean']]

sand_60_100['depth_cm'] = '60-100'
sand_60_100 = sand_60_100[['UID', 'depth_cm', 'mean']]

sand_100_200['depth_cm'] = '100-200'
sand_100_200 = sand_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
sand = pd.concat([sand_0_5, sand_5_15, sand_15_30, sand_30_60, sand_60_100, sand_100_200])
sand = sand.rename(columns = {'mean':'sand_prc'})

### Silt"""

# Import silt
os.chdir(silt_fp)
silt_0_5 = pd.read_csv('Silt_0_5_POLARIS_FieldsNoDups.csv')
silt_5_15 = pd.read_csv('Silt_5_15_POLARIS_FieldsNoDups.csv')
silt_15_30 = pd.read_csv('Silt_15_30_POLARIS_FieldsNoDups.csv')
silt_30_60 = pd.read_csv('Silt_30_60_POLARIS_FieldsNoDups.csv')
silt_60_100 = pd.read_csv('Silt_60_100_POLARIS_FieldsNoDups.csv')
silt_100_200 = pd.read_csv('Silt_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
silt_0_5['depth_cm'] = '0-5'
silt_0_5 = silt_0_5[['UID', 'depth_cm', 'mean']]

silt_5_15['depth_cm'] = '5-15'
silt_5_15 = silt_5_15[['UID', 'depth_cm', 'mean']]

silt_15_30['depth_cm'] = '15-30'
silt_15_30 = silt_15_30[['UID', 'depth_cm', 'mean']]

silt_30_60['depth_cm'] = '30-60'
silt_30_60 = silt_30_60[['UID', 'depth_cm', 'mean']]

silt_60_100['depth_cm'] = '60-100'
silt_60_100 = silt_60_100[['UID', 'depth_cm', 'mean']]

silt_100_200['depth_cm'] = '100-200'
silt_100_200 = silt_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
silt = pd.concat([silt_0_5, silt_5_15, silt_15_30, silt_30_60, silt_60_100, silt_100_200])
silt = silt.rename(columns = {'mean':'silt_prc'})

### Theta R"""

# Import theta_r
os.chdir(theta_r_fp)
theta_r_0_5 = pd.read_csv('Theta_R_0_5_POLARIS_FieldsNoDups.csv')
theta_r_5_15 = pd.read_csv('Theta_R_5_15_POLARIS_FieldsNoDups.csv')
theta_r_15_30 = pd.read_csv('Theta_R_15_30_POLARIS_FieldsNoDups.csv')
theta_r_30_60 = pd.read_csv('Theta_R_30_60_POLARIS_FieldsNoDups.csv')
theta_r_60_100 = pd.read_csv('Theta_R_60_100_POLARIS_FieldsNoDups.csv')
theta_r_100_200 = pd.read_csv('Theta_R_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
theta_r_0_5['depth_cm'] = '0-5'
theta_r_0_5 = theta_r_0_5[['UID', 'depth_cm', 'mean']]

theta_r_5_15['depth_cm'] = '5-15'
theta_r_5_15 = theta_r_5_15[['UID', 'depth_cm', 'mean']]

theta_r_15_30['depth_cm'] = '15-30'
theta_r_15_30 = theta_r_15_30[['UID', 'depth_cm', 'mean']]

theta_r_30_60['depth_cm'] = '30-60'
theta_r_30_60 = theta_r_30_60[['UID', 'depth_cm', 'mean']]

theta_r_60_100['depth_cm'] = '60-100'
theta_r_60_100 = theta_r_60_100[['UID', 'depth_cm', 'mean']]

theta_r_100_200['depth_cm'] = '100-200'
theta_r_100_200 = theta_r_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
theta_r = pd.concat([theta_r_0_5, theta_r_5_15, theta_r_15_30, theta_r_30_60, theta_r_60_100, theta_r_100_200])
theta_r = theta_r.rename(columns = {'mean':'thetaR_m3m3'})

### Theta S"""

# Import theta_s
os.chdir(theta_s_fp)
theta_s_0_5 = pd.read_csv('Theta_S_0_5_POLARIS_FieldsNoDups.csv')
theta_s_5_15 = pd.read_csv('Theta_S_5_15_POLARIS_FieldsNoDups.csv')
theta_s_15_30 = pd.read_csv('Theta_S_15_30_POLARIS_FieldsNoDups.csv')
theta_s_30_60 = pd.read_csv('Theta_S_30_60_POLARIS_FieldsNoDups.csv')
theta_s_60_100 = pd.read_csv('Theta_S_60_100_POLARIS_FieldsNoDups.csv')
theta_s_100_200 = pd.read_csv('Theta_S_100_200_POLARIS_FieldsNoDups.csv')

# Adjust the data by adding a column for depth and selecting the columns of interest
theta_s_0_5['depth_cm'] = '0-5'
theta_s_0_5 = theta_s_0_5[['UID', 'depth_cm', 'mean']]

theta_s_5_15['depth_cm'] = '5-15'
theta_s_5_15 = theta_s_5_15[['UID', 'depth_cm', 'mean']]

theta_s_15_30['depth_cm'] = '15-30'
theta_s_15_30 = theta_s_15_30[['UID', 'depth_cm', 'mean']]

theta_s_30_60['depth_cm'] = '30-60'
theta_s_30_60 = theta_s_30_60[['UID', 'depth_cm', 'mean']]

theta_s_60_100['depth_cm'] = '60-100'
theta_s_60_100 = theta_s_60_100[['UID', 'depth_cm', 'mean']]

theta_s_100_200['depth_cm'] = '100-200'
theta_s_100_200 = theta_s_100_200[['UID', 'depth_cm', 'mean']]

# Concatenate into one dataframe and add soil type column
theta_s = pd.concat([theta_s_0_5, theta_s_5_15, theta_s_15_30, theta_s_30_60, theta_s_60_100, theta_s_100_200])
theta_s = theta_s.rename(columns = {'mean':'thetaS_m3m3'})

## Make a master file
#Includes all soil types and their depths


# Merge all of the files into one

os.path.expanduser("~/EGGS-Aquacrop")  # reset working directory


# filter for countys in GMD-4
gmd4 = ['Cheyenne', 'Rawlins', 'Decatur', 'Sherman', 'Gove', 'Thomas', 'Sheridan', 'Graham', 'Wallace', 'Logan'] # list of gmd4 coundties
county_boundary = county_boundary[county_boundary['name'].isin(gmd4)] # filter for counties inside gmd4 list

# county_boundary and fields layers have same projection
fields = gmd4_fields.to_crs({'init': 'epsg:4326'})
county_boundary = county_boundary.to_crs({'init': 'epsg:4326'})

# Execute spatial join
fields_county = county_boundary.sjoin(fields, how="inner", predicate='intersects') # has 11351 obs vs fields with 11314 obs
dup_fields = fields_county[fields_county['UID'].duplicated(keep= False)] # fields that appear in 2 counties

fields_county = fields_county.drop_duplicates(subset=['UID']) # dropped dupilcated observations based on UID
fields_county = fields_county[['UID', 'name', 'geometry']] # select useful variables


# soil variables to merge
soil_var = [log_alpha, clay, log_hb, log_om, log_ksat, Lambda, n, sand, silt, theta_r, theta_s]

for i in (soil_var):
    i.to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+i.columns[2]+'.csv')

# save fields county as a csv file
fields_county.to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/fields_county.csv')




!pip install sqldf
import sqldf

query = """
SELECT *
FROM fields_county AS f
CROSS JOIN clay AS c ON f.UID=c.UID;
"""

df1 = sqldf.run(query)










query = """
SELECT *
FROM logAlpha_kPa1 AS a
JOIN clay_prc AS c ON a.UID=c.UID
    AND a.depth_cm=c.depth_cm
JOIN logHB_kPa AS hb ON a.UID=hb.UID
    AND a.depth_cm=hb.depth_cm
JOIN logOm_kPa AS om ON a.UID=om.UID
    AND a.depth_cm=om.depth_cm
JOIN logKsat_cmHr AS ksat ON a.UID=ksat.UID
    AND a.depth_cm=ksat.depth_cm
JOIN lambda AS l ON a.UID=l.UID
    AND a.depth_cm=l.depth_cm
JOIN n ON a.UID=n.UID
    AND a.depth_cm=n.depth_cm
JOIN sand_prc AS sa ON a.UID=sa.UID
    AND a.depth_cm=sa.depth_cm
JOIN silt_prc AS sl ON a.UID=sl.UID
    AND a.depth_cm=sl.depth_cm
JOIN thetaR_m3m3 AS r ON a.UID=r.UID
    AND a.depth_cm=r.depth_cm
JOIN thetaS_m3m3 AS s ON a.UID=s.UID
    AND a.depth_cm=s.depth_cm;
"""

df_view = sqldf.run(query)









SELECT 
  origin, dest, 
  airports.name AS dest_name,
  flight, carrier
FROM flights
JOIN airports ON flights.dest = airports.faa





df = pd.DataFrame({'col1': ['A', 'B', np.NaN, 'C', 'D'], 'col2': ['F', np.NaN, 'G', 'H', 'I']})

# Define a SQL (SQLite3) query
query = """
SELECT *
FROM df as d
WHERE col1 IS NOT NULL;
"""

# Run the query
df_view = sqldf.run(query)


url = ('https://raw.github.com/pandas-dev/pandas/master/pandas/tests/data/tips.csv')
tips = pd.read_csv(url)

# Define a SQL (SQLite3) query
query = """
UPDATE tips
SET tip = tip*2
WHERE tip < 2;
"""

# Run the query
sqldf.run(query)


















# start sql querying
from sqlalchemy import create_engine
  
# Create the engine to connect to the inbuilt 
# sqllite database
engine = create_engine("sqlite+pysqlite:///:memory:")
  
# Read data from CSV which will be
# loaded as a dataframe object
data = pandas.read_csv('superstore.csv')
  
# print the sample of a dataframe
data.head()
  
# Write data into the table in sqllite database
p = log_alpha.to_sql('test', engine)





data = {'product_name': ['Computer','Tablet','Monitor','Printer'],
        'price': [900,300,450,150]
        }

df = pd.DataFrame(data, columns= ['product_name','price'])

import sqlite3

conn = sqlite3.connect('test_database')
c = conn.cursor()

c.execute('CREATE TABLE IF NOT EXISTS products (product_name text, price number)')
conn.commit()

data = {'product_name': ['Computer','Tablet','Monitor','Printer'],
        'price': [900,300,450,150]
        }

df = pd.DataFrame(data, columns= ['product_name','price'])
df.to_sql('products', conn, if_exists='replace', index = False)
 
c.execute('''  
SELECT * FROM products
          ''')

!pip install -U pandasql
!pip install pysqldf
import pysqldf
from pandasql import pysqldf
from sqlalchemy import text
#import pandas as pd
from sklearn import datasets

df_feature = datasets.load_iris(as_frame = True)['data']
df_target = datasets.load_iris(as_frame = True)['target']




!pip install pandasql
from pandasql import sqldf
sqldf("SELECT * FROM churn_data LIMIT 10").head(5)


!pip install duckdb
import duckdb
duckdb.query("SELECT * FROM churn_data LIMIT 10").df()


query = 'SELECT * FROM df_target LIMIT 3'
pysqldf(query)

from pandasql import sqldf

students= {
    'Students':["Sira","Ibrahim","Moussa","Mamadou","Nabintou"],
    'Gender':['Female','Male','Male', "Male", "Female"],
    'Age':[18, 27, 19, 22, 21],
    'Email': ["sira@info.com", "ib@info.com", "mouss@info.com", 
             "mam@info.com", "nab@info.com"]
          }
students_df = pd.DataFrame(students)

students_df


teaching_assistant= {
    'Teacher':["Ibrahim","Nabintou","Mamadou","Fatim","Aziz"],
    'Email':['ib@info.com','nab@info.com','mam@info.com', 
             "fat@info.com", "aziz@info.com"],
    'Degree':["M.S in Data Science", "B.S in Statistics", 
              "B. Comp Sc", "M.S. Architecture", "B.S in Accounting"],
    'Department': ["Business", "Statistics", "Comp Sc", 
             "Engineering", "Business"]
          }
teaching_assistant_df = pd.DataFrame(teaching_assistant)

teaching_assistant_df


all_students = sqldf("SELECT * FROM students_df as s")
all_students


query2 = """
SELECT * FROM students_df;
"""

# Run the query
df_view2 = sqldf.run(query2)


query = """ SELECT st.Students, st.Gender, st.Email, st.Age, tat.Department
            FROM students_df st INNER JOIN teaching_assistant_df tat 
            ON st.Email = tat.Email;
            """


query = 


result = sqldf(query)
result

















def soilCounty(x):
    soils_county = x.merge(fields_county, on=['UID'])
    soils_county.to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soils_county.columns[2]+'.csv')
    return(soils_county)


list(map(soilCounty, soil_var))

soil_mapdf = map(soilCounty, soil_var)
soil_county_lst = list(soil_mapdf)


soil_county_lst[0].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[0].columns[2]+'.csv'
soil_county_lst[1].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[1].columns[2]+'.csv'
soil_county_lst[2].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[2].columns[2]+'.csv'
soil_county_lst[3].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[3].columns[2]+'.csv'
soil_county_lst[4].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[4].columns[2]+'.csv'
soil_county_lst[5].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[5].columns[2]+'.csv'
soil_county_lst[6].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[6].columns[2]+'.csv'
soil_county_lst[7].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[7].columns[2]+'.csv'
soil_county_lst[8].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[8].columns[2]+'.csv'
soil_county_lst[9].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[9].columns[2]+'.csv'
soil_county_lst[10].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+soil_county_lst[10].columns[2]+'.csv'


soil_county_lst[4].to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/test.csv')
# for loop is super slow - dont run
for i in (soil_county_lst):
    i.to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+i.columns[2]+'.csv')

def export_csv(df):
    df.to_csv(r'./data/agricLand/soils/gmd4_soils/MergedSoil/'+i.columns[2]+'.csv')

t = map(export_csv, soil_county_lst)

list(map(export_csv, soil_county_lst))










folderpath = r'./data/agricLand/soils/gmd4_soils/MergedSoil/'
csv = 'csv'  # output file type
for i, df in enumerate(soil_county_lst, 1):
    filename = "df_{}.{}".format(i, csv)
    filepath = os.path.join(folderpath, filename)
    df.to_csv(filepath)




l = soil_county_lst[0]

l.columns[2]



county_name = []
soil_data = []
for df in soil_county_lst:
    #print(i)
    for col in df.groupby('name'):
        name = col[0]
        soil_ch = col[1]
        county_name.append(name)
        soil_data.append(soil_ch)




from collections import defaultdict
soil_dict = defaultdict(list)
for k, v in zip(county_name,soil_data):
      soil_dict[k].append(v)



# Merge the dataframes if the UIDs match and save the new dataframes in a list
merged_soil = []
for key in soil_dict:
    dfs = soil_dict[key]
    #df2 = soil_dict[key][1]
    merged_df = pd.merge(dfs[0], dfs[1], on=['UID', 'depth_cm', 'geometry', 'name'])
    #print(merged_df)
    for i in range(2, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], on=['UID', 'depth_cm', 'geometry', 'name'])
    merged_soil.append(merged_df)
    #my_dict[key] = merged_df





l = merged_soil[9]



for key in dictionary:
    df_list = dictionary[key]
    merged_df = pd.merge(df_list[0], df_list[1], how='outer')
    for i in range(2, len(df_list)):
        merged_df = pd.merge(merged_df, df_list[i], how='outer')
    merged_dataframes.append(merged_df)











ser = df['Fee'].map(lambda x:fun1(x))
print(ser)

soil = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(silt,sand, on=['UID', 'depth_cm']),
                                  clay,on=['UID', 'depth_cm']),
                         theta_s,on=['UID', 'depth_cm']),
                theta_r,on=['UID', 'depth_cm']),
                log_ksat,on=['UID', 'depth_cm']),
                Lambda,on=['UID', 'depth_cm']),
                log_hb,on=['UID', 'depth_cm']),
                log_om,on=['UID', 'depth_cm']),
                n,on=['UID', 'depth_cm']),
                log_alpha,on=['UID', 'depth_cm'])




# Make UID a float
soil['UID'] = soil['UID'].astype(np.int64)

# Make depth_cm a string
soil['depth_cm'] = soil['depth_cm'].astype(str)

# Round numbers to 2 decimal places
cols = ['silt_prc', 'sand_prc', 'clay_prc', 'thetaS_m3m3', 'thetaR_m3m3', 'logKsat_cmHr', 'lambda', 'logHB_kPa', 'logOm_%', 'n', 'logAlpha_kPa1']
soil[cols] = soil[cols].apply(pd.to_numeric)
soil[cols] = soil[cols].round(2)

"""## Save as CSV"""

soil.to_csv(r'./data/agricLand/soils/gmd4_soils/SoilData_GMD4_WNdlovu_v1_20230223.csv')












!pip install pandasql
import pandasql as ps


from pandasql import sqldf


url = (
    "https://raw.githubusercontent.com/pandas-dev"
    "/pandas/main/pandas/tests/io/data/csv/tips.csv"
)

import sqlite3
tips = pd.read_csv(url)
pysqldf = lambda q: sqldf(q, locals())

pt = """SELECT total_bill, tip, smoker, time
FROM tips;"""

joined = pyqldf(pt)
print(joined.head(joined))




















