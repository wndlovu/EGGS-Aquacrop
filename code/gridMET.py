from os import chdir, getcwd
wd=getcwd() # set working directory
chdir(wd)


# wrangle the gridMET data and filter for site 1381151
import pandas as pd
import os
import glob
from datetime import datetime


gridMET = pd.read_csv(wd + '\\data\\hydrometeorology\\gridMET/gridMET_400m_2000_21.csv')

# recalculate t0c in celcius and create ymd variables
gridMET = gridMET.assign(Tmin = gridMET.tmmn-273.15,
                    Tmax = gridMET.tmmx-273.15,
                    date = pd.to_datetime(gridMET['date_ymd'], format='%Y%m%d'))


# separiting date
gridMET = gridMET.assign(day =  gridMET['date'].dt.day,
                         month = gridMET['date'].dt.month,
                         year = gridMET['date'].dt.year)

# filter for UID 1381151 and specified columns
final_df = gridMET[gridMET['UID'] == 1381151]
final_df = final_df[["day", "month", "year", "Tmin", "Tmax", "pr", "eto"]]

               