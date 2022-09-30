!pip install aquacrop==0.2
from os import chdir, getcwd
from aquacrop.classes import    *
from aquacrop.core import       *
import pandas as pd
import sys
import seaborn as sns
import os
import glob
_=[sys.path.append(i) for i in ['.', '..']]
wd=getcwd() # set working directory
chdir(wd)

#!pip install aquacrop







## Aquacrop Model
path = get_filepath(wd + '/data/hydrometeorology/gridMET/gridMET_1381151.txt') #replace folder name from folder name with file path
wdf = prepare_weather(path)
sim_start = '2000/01/01' #dates to match crop data
sim_end = '2015/12/31'
soil= SoilClass('SiltLoam')
crop = CropClass('Maize',PlantingDate='05/01')
initWC = InitWCClass(value=['FC'])


labels=[]
outputs=[]
for smt in range(0,110,20):
    crop.Name = str(smt) # add helpfull label
    labels.append(str(smt))
    irr_mngt = IrrMngtClass(IrrMethod=1,SMT=[smt]*4) # specify irrigation management [40,60,70,30]*4
    model = AquaCropModel(sim_start,sim_end,wdf,soil,crop,InitWC=initWC,IrrMngt=irr_mngt) # create model
    model.initialize() # initilize model
    model.step(till_termination=True) # run model till the end
    outputs.append(model.Outputs.Final) # save results
all_outputs = pd.concat(outputs)


## get irrigation values from Aquacrop
irrig_aqc = all_outputs.assign(Year =  all_outputs['Harvest Date (YYYY/MM/DD)'].dt.year)
irrig_aqc = irrig_aqc.pivot(index= 'Year', # show irrigation vals for each year horizontally
                             columns='Crop Type', 
                             values='Seasonal irrigation (mm)')
irrig_aqc.reset_index(inplace=True)       # make year a column 
                                                                 

irrig_aqc = irrig_aqc[irrig_aqc['Year'] >= 2006] # filter for yrs>2006 to match WIMAS


## Water Rights WIMAS
wr_groups = pd.read_csv(wd + "/data/water/WRgroups_FieldByYear.csv")
water_use = pd.read_csv(wd + "/data/water/WRgroups_UseByWRG.csv")

# merge water right groups and water use
irrig_wimas = pd.merge(wr_groups, water_use, on=["WR_GROUP", "Year"]) # 
irrig_wimas = irrig_wimas[irrig_wimas['UID'] == 	1381151] # filter for field
irrig_wimas = irrig_wimas[irrig_wimas['Year'] <= 	2015]
irrig_wimas = irrig_wimas.assign(irrig_wimas = (irrig_wimas['Irrigation_m3']/(irrig_wimas['TRGT_ACRES']*4046.86))*1000)

# WIMAS and Aquacrop irrigation df
irrig_df = pd.merge(irrig_wimas, irrig_aqc, on=["Year", "Year"])
irrig_df  = irrig_df[['UID', 'Year', 'Irrigation_m3', 'irrig_wimas', '0', '20', '40', '60', '80', '100']]

# boxplots showing WIMAS irrigation and irrigation levels at the different SMT 2006-20015
boxplot = irrig_df.boxplot(column=['irrig_wimas', '0', '20', '40', '60', '80', '100'])
boxplot.set_ylabel('Irrigation (mm)')
boxplot.set_xlabel('Soil-moisture threshold (%TAW)')


# Yield Data
# yield data from usda nass https://quickstats.nass.usda.gov/#D93A3218-8B77-31A6-B57C-5A5D97A157D8
yield_Irrig = pd.read_csv(wd + "/data/agricLand/yield/sheridanYield_Irrig.csv") #CORN, GRAIN, YIELD, MEASURED IN BU / ACRE
#yield_noIrrig = pd.read_csv(wd.replace('code',"data/agricLand/gridMET/sheridanYield_noIrrig.csv")) #CORN, GRAIN, IRRIGATED - YIELD, MEASURED IN BU / ACRE

# Select year and irrigation value
yield_Irrig = yield_Irrig[['Year', 'Value']]

# df with USDS NASS yield and Aquacrop yield
yield_df = all_outputs.assign(Year =  all_outputs['Harvest Date (YYYY/MM/DD)'].dt.year)
yield_df = yield_df.pivot(index= 'Year', # show irrigation vals for each year horizontally
                             columns='Crop Type', 
                             values='Yield (tonne/ha)')
yield_df = pd.merge(yield_df, yield_Irrig, on=["Year", "Year"])
yield_df = yield_df.assign(YieldUSDA = yield_df['Value']*0.0673) # convert yield from bushels/acre to tonne/ha

# yield 2000-2015
boxplot = yield_df.boxplot(column=['YieldUSDA', '0', '20', '40', '60', '80', '100'])
boxplot.set_ylabel('Yield (t/ha)')
boxplot.set_xlabel('Soil-moisture threshold (%TAW)')


yield_df.to_csv(r'./data/analysis_results/yield_df.csv', sep=',', encoding='utf-8', header='true')
                         
