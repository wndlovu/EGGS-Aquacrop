#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:03:18 2023

@author: wayne
"""

def fun(crops, grid, soil):
    grouped_info = [] # crop combination info (county, crop name and irrigation management (irrig or rainfed))
    grouped_crop = [] # dataframe for each combination
    gridMET_county = [] # daily average gridment for all fields in a county 
    id_list = []
    soil_county = []
    custom_soil = []
    for name in crops.groupby(['name', 'cropName', 'irrig_management']): #groupby county, crop name and irrigation management (irrig or rainfed)
    #grouped_crop.append(name)
        group_info = name[0] # collect the group combination information
        grouped_info.append(group_info)  
        group_df = name[1] # extract dataframe with the fields 
        group_df = group_df.drop_duplicates(subset=['UID']) # drop duplicated field IDS
    
        ids = grid.UID.isin(group_df.UID) # filter gridMET df for fields in the group
        county_gridMET = grid[ids] # dataframe with gridMET data for fields of interest
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
    
        
        soil_df = soil[soil['UID'].isin(group_df['UID'].tolist())]
        
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
    
    
        
        for i, row in soil_df.iterrows():   #soil_df.itertuples():
            ids = soil_df['UID'][i] #create soil_df with UID from the soils file used - fix this
            #id_list.append(ids)
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
    return grouped_info, gridMET_county, soil_county, custom_soil
    
    
    
test1 = fun(crops_irrig, gridMET, soils)
