#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:10:07 2023

@author: wayne
"""

!pip install aquacrop==2.2
!pip install numba==0.55
!pip install statsmodels==0.13.2
!pip install openpyxl
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
#from aquacrop.classes import    *
#from aquacrop.core import       *




wd=getcwd() # set working directory
chdir(wd)

water_use = pd.read_csv(wd + '/data/water/WaterUse_GMD4model_BWilson_v1_2023.csv')
water_use_crop =  pd.read_excel(wd + '/data/water/WaterUseCropCode_GMD4model_BWilson_v1_2023.xlsx')


# wimas crop codes
wimas_crop_code = [2, 3, 4, 5]

wimas_crop_name = ["Corn", 
             "Sorghum", 
             "Soybeans",
             "Winter Wheat"]

wimas_crops = pd.DataFrame(zip(wimas_crop_code, wimas_crop_name), columns=['crop_code', 'crop_name'])


# check to see which IDs have multiple crops reported per year
crops_perYear = water_use_crop.groupby(['WUA_YEAR', 'PDIV_ID', 'CONCATENATE_CROP_CODE'], as_index=False).count()

# ids with only one crop or the same crop reported multiple times
crops_perYear['multi_crops'] = crops_perYear['CONCATENATE_CROP_CODE'].str.split(' ,', 1, expand=True) # split crop column
crops_perYear['crop1'] = crops_perYear['multi_crops'].str.extract(r'(.*),') # crop 1
crops_perYear['crop2'] = crops_perYear['multi_crops'].str.extract(r'(\w+(?: \w+)*)$') # crops 2
# how to get 3rd crops?


crops_perYear = crops_perYear[(crops_perYear['crop1'] == crops_perYear['crop2']) | (crops_perYear['crop1'].isna())] # select rows where crop1 = crop2 or where 1 crops is reported
crops_perYear = crops_perYear.rename(columns={"CONCATENATE_CROP_CODE": 'crop_code'}) # rename crop code

# 12 observations with muliple crops
#multiple_crops = crops_perYear[(crops_perYear['crop1'] != crops_perYear['crop2'])]
#multiple_crops = multiple_crops.dropna()


# irrigation depth
irrigation_depth = water_use.assign(irrig_depth_1990 = (water_use['AF_USED_1990']/water_use['ACRES_1990'])*304.8,
                                    irrig_depth_1991 = (water_use['AF_USED_1991']/water_use['ACRES_1991'])*304.8,
                                    irrig_depth_1992 = (water_use['AF_USED_1992']/water_use['ACRES_1992'])*304.8,
                                    irrig_depth_1993 = (water_use['AF_USED_1993']/water_use['ACRES_1993'])*304.8,
                                    irrig_depth_1994 = (water_use['AF_USED_1994']/water_use['ACRES_1994'])*304.8,
                                    irrig_depth_1995 = (water_use['AF_USED_1995']/water_use['ACRES_1995'])*304.8,
                                    irrig_depth_1996 = (water_use['AF_USED_1996']/water_use['ACRES_1996'])*304.8,
                                    irrig_depth_1997 = (water_use['AF_USED_1997']/water_use['ACRES_1997'])*304.8,
                                    irrig_depth_1998 = (water_use['AF_USED_1998']/water_use['ACRES_1998'])*304.8,
                                    irrig_depth_1999 = (water_use['AF_USED_1999']/water_use['ACRES_1999'])*304.8,
                                    irrig_depth_2000 = (water_use['AF_USED_2000']/water_use['ACRES_2000'])*304.8,
                                    irrig_depth_2001 = (water_use['AF_USED_2001']/water_use['ACRES_2001'])*304.8,
                                    irrig_depth_2002 = (water_use['AF_USED_2002']/water_use['ACRES_2002'])*304.8,
                                    irrig_depth_2003 = (water_use['AF_USED_2003']/water_use['ACRES_2003'])*304.8,
                                    irrig_depth_2004 = (water_use['AF_USED_2004']/water_use['ACRES_2004'])*304.8,
                                    irrig_depth_2005 = (water_use['AF_USED_2005']/water_use['ACRES_2005'])*304.8,
                                    irrig_depth_2006 = (water_use['AF_USED_2006']/water_use['ACRES_2006'])*304.8,
                                    irrig_depth_2007 = (water_use['AF_USED_2007']/water_use['ACRES_2007'])*304.8,
                                    irrig_depth_2008 = (water_use['AF_USED_2008']/water_use['ACRES_2008'])*304.8,
                                    irrig_depth_2009 = (water_use['AF_USED_2009']/water_use['ACRES_2009'])*304.8,
                                    irrig_depth_2010 = (water_use['AF_USED_2010']/water_use['ACRES_2010'])*304.8,
                                    irrig_depth_2011 = (water_use['AF_USED_2011']/water_use['ACRES_2011'])*304.8,
                                    irrig_depth_2012 = (water_use['AF_USED_2012']/water_use['ACRES_2012'])*304.8,
                                    irrig_depth_2013 = (water_use['AF_USED_2013']/water_use['ACRES_2013'])*304.8,
                                    irrig_depth_2014 = (water_use['AF_USED_2014']/water_use['ACRES_2014'])*304.8,
                                    irrig_depth_2015 = (water_use['AF_USED_2015']/water_use['ACRES_2015'])*304.8,
                                    irrig_depth_2016 = (water_use['AF_USED_2016']/water_use['ACRES_2016'])*304.8,
                                    irrig_depth_2017 = (water_use['AF_USED_2017']/water_use['ACRES_2017'])*304.8,
                                    irrig_depth_2018 = (water_use['AF_USED_2018']/water_use['ACRES_2018'])*304.8,
                                    irrig_depth_2019 = (water_use['AF_USED_2019']/water_use['ACRES_2019'])*304.8,
                                    irrig_depth_2020 = (water_use['AF_USED_2020']/water_use['ACRES_2020'])*304.8,
                                    irrig_depth_2021 = (water_use['AF_USED_2021']/water_use['ACRES_2021'])*304.8)


# select irrigation depth and ids
# selecting columns where column name contains 'Average' string and identifiers
irrigation_depth = irrigation_depth.filter(regex='ID|nad|gmd|county|source|hpa|irrig_depth')

irrigation_depth["id"] = irrigation_depth.index # iadd index column

irrig_depth_long =pd.wide_to_long(irrigation_depth, ["irrig_depth_"], i='id', j="WUA_YEAR") # pivot longer and get the year
irrig_depth_long.reset_index(inplace=True)  # max index column
irrig_depth_long = irrig_depth_long.rename(columns={"irrig_depth_": "irrig_depth"}) # rename irrigation column to remove _
irrig_depth_long = irrig_depth_long[['PDIV_ID', 'long_nad83', 'lat_nad83', 'gmd',
                                     'county_abrev', 'hpa_region', 'WUA_YEAR', 'irrig_depth']] # arrange variables
 

# water use for each pdiv id
water_use_full = irrig_depth_long.merge(crops_perYear, on=['PDIV_ID', 'WUA_YEAR'], how='inner')

water_use_full = water_use_full.merge(wimas_crops, on= ['crop_code'], how = 'inner') # df with corn, sorghum, soybeans and winter wheat

water_use_full = water_use_full[['PDIV_ID', 'long_nad83', 'lat_nad83', 'gmd', 'county_abrev', 'hpa_region', 
                                 'WUA_YEAR', 'crop_code', 'crop_name', 'irrig_depth']]


# save file
water_use_full.to_csv(r'./data/water/Kansas/IrrigationDepth_GMD4_WNdlovu_v1_20230123.csv', sep=',', encoding='utf-8', header='true') 





