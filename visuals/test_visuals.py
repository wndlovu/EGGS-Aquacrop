#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 08:11:18 2022

@author: wayne
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
wd=getcwd() # set working directory
chdir(wd)

# read in df
et_means_test = pd.read_csv(wd + '/data/analysis_results/et_df_test.csv', index_col=0)

et_means_test = et_means_test[['time', 'disalexi', 
                     'ensemble', 'geesebal', 
                     'ptjpl', 'sims', 'ssebop', 'Et']]

et_means_test = et_means_test.rename(columns={'Et': 'aquacrop'})

et_means_test['year'] = pd.to_datetime(et_means_test['time']).dt.strftime('%Y')

# make dfs pivoted longer
u = pd.melt(et_means_test, id_vars=['time', 'year'],
                  var_name='model_name', value_name='values')

v = pd.melt(et_means_test, id_vars=['time', 'year', 'aquacrop'],
                  var_name='model_name', value_name='values')

yield_df_test = pd.read_csv(wd + '/data/analysis_results/yield_df_test.csv', index_col=0)
irrig_df_test = pd.read_csv(wd + '/data/analysis_results/irrig_df_test.csv', index_col =0)

# irrig pivoted longer
irrig_df_test = irrig_df_test.rename(columns={'Seasonal irrigation (mm)': 'Aquacrop',
                                              'irrig_wimas': 'WIMAS'})

irrig_long = pd.melt(irrig_df_test, id_vars = ['UID', 'Year'],
                     var_name = 'model',
                     value_name = 'values')

## yield pivoted_longer
yield_df_test = yield_df_test.rename(columns={'Yield (tonne/ha)': 'Aquacrop',
                                              'YieldUSDA': 'USDA-NASS'})

yield_long = pd.melt(yield_df_test, id_vars = ['Year'],
                     var_name = 'model',
                     value_name = 'values')


fig, axs = plt.subplots(ncols=2, nrows = 3, figsize=(15,18))
sns.lineplot(x='time', y='values', data=u, ax=axs[0,0], hue = 'model_name', palette = 'colorblind') 
sns.scatterplot(x='aquacrop', y='values', data=v, ax=axs[0,1], hue = 'model_name', palette = 'colorblind') 
sns.scatterplot(x='Year',y='values', data=irrig_long, ax=axs[1,0], hue = 'model', palette = 'colorblind') 
sns.scatterplot(x='Aquacrop',y='WIMAS', data=irrig_df_test, ax=axs[1,1])
sns.scatterplot(x='Year',y='values', data=yield_long, ax=axs[2,0], hue = 'model', palette = 'colorblind') 
sns.scatterplot(x='Aquacrop',y='USDA-NASS', data=yield_df_test, ax=axs[2,1]) 
#ax=axs[0,0].tick_params(axis='time', labelsize=4)
ax=axs[0,0].set(xlabel = '', ylabel = 'ET (mm)')
ax=axs[0,1].set(xlabel = 'Aquacrop ET (mm)', ylabel = 'OpenET (mm)')
ax=axs[1,0].set(xlabel = '', ylabel = 'Irrigation (mm')
ax=axs[1,1].set(xlabel = 'Aquacrop Irrigation (mm)', ylabel = 'WIMAS Irrigation (mm)')
ax=axs[2,0].set(xlabel = 'Time', ylabel = "Yield (t/ha)")
ax=axs[2,1].set(xlabel = 'Aquacrop Yield (t/ha)', ylabel = "USDA Yield (t/ha)")
ax=axs[0,1].axline((0, 0), slope=1, color = "grey")
ax=axs[1,1].axline((250, 250), slope=1, color = "grey")
ax=axs[2,1].axline((9, 9), slope=1, color = "grey")
ax=axs[0,0].tick_params(axis='x', rotation=90, labelsize=8)
ax=axs[0,0].xaxis.set_major_locator(plt.MaxNLocator(12))
ax=axs[0,0].legend().set_title('') # remove legend titles
ax=axs[0,1].legend().set_title('')
ax=axs[1,0].legend().set_title('')
ax=axs[2,0].legend().set_title('')


