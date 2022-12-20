from netCDF4 import Dataset, num2date
import matplotlib as mpl
from matplotlib import gridspec,cm, rcParams, pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from eofs.standard import Eof
from os.path import exists
import sys  
from cartopy import config, crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.ndimage import uniform_filter1d

def subset_data(data, analysis_type, season, season_mean, month, running_mean):
    # -------------------------------------------------------------------
    # select subset of data:

    if (analysis_type == 'monthly'):
        #print(f' --- Monthly analysis: {month}  ({int(len(data)/12)} months) ---') 
        month_indices = {'January':0,'February':1,'March':2,'April':3,'May':4,'June':5,'July':6,'August':7,'September':8,'October':9,'November':10,'December':11}
        indices = np.arange(month_indices[month],len(data),12)
        data = data[indices,:,:]
        #print(f' Shape data:  {np.shape(data)}')


    if (analysis_type == 'seasonal'):
        years = int(len(data)/12)
        #print(f' --- Seasonal analysis: {season}  ({years} seasons) ---')
        month_indices = {'DJF':[0,1,11], 'MAM':[2,3,4], 'JJA':[5,6,7], 'SON':[8,9,10], 'NDJFM':[10,11,0,1,2]}
        month1 = np.arange(0,years)*12 + month_indices[season][0]   
        month2 = np.arange(0,years)*12 + month_indices[season][1] 
        month3 = np.arange(0,years)*12 + month_indices[season][2]
        indices = np.sort(np.concatenate((month1, month2, month3)))  

        if (season == 'NDJFM'): 
            month4 = np.arange(0,years)*12 + month_indices[season][3]   
            month5 = np.arange(0,years)*12 + month_indices[season][4]
            indices = np.sort(np.concatenate((month1, month2, month3,month4,month5)))  

        data = data[indices,:,:]
        print(f' Shape data (seasonal):  {np.shape(data)}')

        
        if season_mean:                      # calculate season mean
            #print(f' --- Seasonal mean ---')
            if (season == 'DJF'): 
                data = data[2:-1]   # disregard the first 2 (Jan, Feb) and very last (Dec) month, as they are incomplete seasons
            if (season == 'NDJFM'): 
                data = data[3:-2]   # disregard first JFM and last ND

            l = 0
            months_in_season = 3
            if (season == 'NDJFM'): 
                months_in_season = 5
            
            for i in range(int(len(data)/months_in_season)):
                data[i,:,:] = np.mean(data[l:l+months_in_season,:,:], axis=0)
                l += months_in_season
            data = data[:int(len(data)/months_in_season),:,:]    
            print(f' Seasonal mean: {np.shape(data)}')


            # select shorter identical timeframe for ERA and models:
            if (len(data)==61): # = ERA5: 1959-2020  => remember seasonal means are 1 less than years. winter season 2019 = winter 2019/20
                data = data[:55,:]  # 6 years shorter: 1959-2014
            elif (len(data)==164): # = models: 1850-2014
                data = data[109:,:] # 
            
            
            print(f' Seasonal mean: {np.shape(data)}')


            if (running_mean>1):         # calculate running mean:  
                def rolling_mean_along_axis(a=data, W=running_mean, axis=0):
                # a : Input ndarray, W : Window size, axis : Axis along which we will apply rolling/sliding mean
                    hW = W//2
                    L = a.shape[axis]-W+1   
                    indexer = [slice(None) for _ in range(a.ndim)]
                    indexer[axis] = slice(hW,hW+L)
                    return uniform_filter1d(a,W,axis=axis)[tuple(indexer)]
                data = rolling_mean_along_axis(data,running_mean)   # running_mean = number of years. if = 1, nothing happen     
                
        
    return data


# --------------------------    
# select extreme quartiles:
# ------------------------------------------------------------------------------------ 
def select_extreme_quartiles(regdata_type, extr_infile, analysis_type, season, season_mean, month, running_mean, startyear, endyear, 
    extreme_region_name, base_title, saveextremesplotas, saveextremesmapas, model,
    extreme_timescale='seasons', plot_extremes=True, plot_extremes_map=False):
    #(regdata_type, extreme_region, extreme_timescale='seasons', plot_extremes=True, plot_extremes_map=False):
    #(regdata_type, model, regdata_type, variant, grid, timeframe_long, timeframe, analysis_type, season, month, running_mean, extreme_region, extreme_region_name, running_mean, saveextremesplotas, saveextremesmapas, extreme_timescale, plot_extremes=True, plot_extremes_map=False):
    '''  
    This function selects warmest/coldest and wettest/driest years based on spatial mean of temp and prec data in a selected region
    Required input: nc file with the correct region (use CDO for lat-lon selection)
    '''
    if (model=='ERA5'):
        myvar = 'tp' if (regdata_type=='Precipitation') else 't2m'
    else:
        myvar = 'pr' if (regdata_type=='Precipitation') else 'tas'
    #ending = '.nc' if (regdata_type=='Precipitation') else '_detrended.nc'
    #loaddata = f'data/{model}/{extreme_region}_{myvar}_Amon_{model}_historical_{variant}_{grid}_{timeframe_long}{ending}'
    # assuming that all data should be detrended, also precip:
    #loaddata = f'data/{model}/{extreme_region}_{myvar}_Amon_{model}_historical_{variant}_{grid}_{timeframe_long}_detrended.nc'
    #dataset = Dataset(loaddata)
    dataset = Dataset(extr_infile)
    var = dataset.variables[myvar]
    if (extreme_timescale=='seasons'):
        NE_data = subset_data(var, analysis_type, season, season_mean=True, month=month, running_mean=running_mean)    
    elif (extreme_timescale=='months'):
        NE_data = subset_data(var, analysis_type, season, season_mean=False, month=month, running_mean=running_mean)  
    
    spatial_mean = np.mean(NE_data, axis=(1,2))
    
    q25 = spatial_mean < np.quantile(spatial_mean, 0.25)
    q75 = spatial_mean > np.quantile(spatial_mean, 0.75)
    extremes_mask = q25 + q75
    
    if plot_extremes:  
        fig, ax = plt.subplots(figsize=(6,4))
        if (running_mean>1):
            rmyears = np.floor(running_mean/2)
            ax.plot(np.linspace(startyear+rmyears,endyear-1-rmyears,len(spatial_mean)),spatial_mean,'.', color='black', 
                    label=f'mean winter {regdata_type} in {extreme_region_name}')
        else:
            ax.plot(np.linspace(startyear,endyear-1,len(spatial_mean)),spatial_mean,'.', color='black', 
                    label=f'mean winter {regdata_type} in {extreme_region_name}')
        ax.hlines(np.quantile(spatial_mean, 0.25), xmin=startyear, xmax=endyear-1, color='blue', 
                  label=f'25% quartile ({np.quantile(spatial_mean, 0.25):.3})')
        ax.hlines(np.quantile(spatial_mean, 0.75), xmin=startyear, xmax=endyear-1, color='red', 
                  label=f'75% quartile ({np.quantile(spatial_mean, 0.75):.3})')
        if (regdata_type=='Temperature'): 
            ax.set_ylabel('mean winter temperature [°C]')
        else:
            ax.set_ylabel('mean winter precipitation [kg/$m^2$/s]')
        ax.set_xlabel('Year')
        ax.legend()
        plt.suptitle(f'Mean {regdata_type} in {extreme_region_name} {startyear}-{endyear}', fontsize='x-large')
        # ToDo: implement a variant if (extreme_timescale=='months')!
        plt.title(f'{base_title}', fontsize='small')
        #fig.suptitle(f'Mean winter {regdata_type} in {extreme_region_name} {timeframe} ({model})')
        #if (running_mean>1):
            #fig.suptitle(f'Mean winter {regdata_type} in {extreme_region_name} {timeframe} ({model}) \n {running_mean} year running mean')
        #fig.savefig(f'output/{model}/{timeframe}_{model}_{variant}_{season}mean_{regdata_type}quartiles({extreme_timescale})_{extreme_region}_mean{running_mean}.pdf')
        fig.savefig(f'{saveextremesplotas}_{extreme_timescale}.pdf')
        
    if plot_extremes_map:
        if (model=='ERA5'):
            lats = dataset.variables['latitude'][:]
            lons = dataset.variables['longitude'][:]
        else:
            lats = dataset.variables['lat'][:]
            lons = dataset.variables['lon'][:]
        extrindex = 0
        if (extreme_timescale=='months'):
            when = find_month_from_index(extrindex, startyear, endyear)
        elif (extreme_timescale=='months'):
            when = f'{startyear}+{extrindex}'
        fig, ax = plt.subplots(figsize=(12,10), subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-40.0)})
        cmap = mpl.cm.coolwarm
        if (regdata_type=='Temperature'):
            colorlevels = np.linspace(-16,0,11) 
        if (regdata_type=='Precipitation'):
            colorlevels = np.linspace(2e-06,4e-05,11)    
        pcm = ax.contourf(lons, lats, NE_data[extrindex,:,:], levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend="both")          
        ax.coastlines()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.ylabel_style = {'size': 15, 'color': 'gray'}
        cb = fig.colorbar(pcm, ticks = colorlevels)
        if (regdata_type=='Temperature'):
            cb.set_label(f'Temperature [°C]')
        else: 
            cb.set_label(f'Precipitation [kg/$m^2$/s]')
        fig.suptitle(f'{extreme_region_name} {regdata_type} \n {when}', fontsize='xx-large')
        # ToDo: add model info
        #fig.savefig(f'output/{model}/{timeframe}_{model}_{variant}_{season}mean_{regdata_type}quartiles({extreme_timescale})_{extreme_region}_mean{running_mean}_map.pdf')
        fig.savefig(f'{saveextremesmapas}_{extreme_timescale}_{when}.pdf')
           
        
    # print years:
    if (extreme_timescale=='seasons'):
        years_text = np.arange(startyear,endyear)
        if (running_mean>1):
            rmyears = np.floor(running_mean/2)
            years_text = np.arange(startyear+rmyears,endyear-rmyears)
        if (regdata_type=='Temperature'):
            print(f' Cold years:  {years_text[q25]}')
            print(f' Warm years:  {years_text[q75]}')
        elif (regdata_type=='Precipitation'):
            print(f' Dry years:  {years_text[q25]}')
            print(f' Wet years:  {years_text[q75]}')
    # Remember: Year 1970 refers to the winter season 1970/71!

    
    return q25, q75, extremes_mask


    # Simon does it like this:
    #wet_yrs = ma.masked_less_equal(var, tp_pctl_high)
    #dry_yrs = ma.masked_greater_equal(var, tp_pctl_low)
    # where tp_pctl_low = np.percentile(var, 25) (which is equal to np.quantile(spatial_mean, 0.25))
    # => same same, doesn't make a difference


# --------------------------    
# select extreme months:
# ------------------------------------------------------------------------------------ 
    # need to decide whether I want to look only at the most extreme month, or at the most extreme 3/5/... 
    # if >1 month: do it individually or mean?
    # suggestion: plot pattern of most extreme month individually, as well as mean of most extreme 5

def find_month_from_index(monthindex,startyear,endyear):
    all_years = np.arange(startyear,endyear)
    myyear = all_years[math.floor(monthindex/5)]      # gives year that indicated the season (1970 = winter 1970/71)
    
    puremonthindex = monthindex - ((myyear - startyear) * 5)
    monthindexplus1 = puremonthindex + 1
    if (monthindexplus1%5==0):
        mymonth='March'
    elif (monthindexplus1%4==0):
        mymonth='February'
    elif (monthindexplus1%3==0):
        mymonth='January'
    elif (monthindexplus1%2==0):
        mymonth='December'
    elif (monthindexplus1%1==0):
        mymonth='November'
        
    if (mymonth=='January') or (mymonth=='February') or (mymonth=='March'):
        myyear += 1
        
    return f'{mymonth} {myyear}'  #mymonth, myyear
# -------------------------------    
    
    
    
def select_extreme_months(regdata_type, extreme_region, nmonths, plot_extremes=False):
    myvar = 'pr' if (regdata_type=='Precipitation') else 'tas'
    loaddata = f'data/{model}/{extreme_region}_{myvar}_Amon_{model}_historical_{variant}_{grid}_{timeframe_long}_detrended.nc'
    
    #if (regdata_type=='Precipitation'): 
        #loaddata = f'data/{model}/{extreme_region}_{myvar}_Amon_{model}_historical_{variant}_{grid}_{timeframe_long}.nc'
        #precip_data = Dataset(loaddata)
        #pr = precip_data.variables['pr'] 
        #NE_data = subset_data(pr, analysis_type, season, season_mean=False, month=month, running_mean=running_mean)
        
    #elif (regdata_type=='Temperature'):  # _detrended
        #loaddata = f'data/{model}/{extreme_region}_{myvar}_Amon_{model}_historical_{variant}_{grid}_{timeframe_long}_detrended.nc'
    
    ds_data = Dataset(loaddata)
    var = ds_data.variables[myvar]
    NE_data = subset_data(var, analysis_type, season, season_mean=False, month=month, running_mean=running_mean)

    spatial_mean = np.mean(NE_data, axis=(1,2))
    print(f'Data length = {len(NE_data)} months')
    
    
    posexmonth =  np.argmax(spatial_mean)
    negexmonth =  np.argmin(spatial_mean)
    
    # print months:
    if (regdata_type=='Temperature'):
        print(f' Coldest month: {find_month_from_index(negexmonth, startyear, endyear)}: {spatial_mean[negexmonth]:.2f} °C')
        print(f' Warmest month:  {find_month_from_index(posexmonth, startyear, endyear)}: {spatial_mean[posexmonth]:.2f} °C')
    elif (regdata_type=='Precipitation'):
        print(f' Driest month: {find_month_from_index(negexmonth, startyear, endyear)}')
        print(f' Wettest month:  {find_month_from_index(posexmonth, startyear, endyear)}')

        
        
    # select n most extreme months:
    n_posexmonths = np.argpartition(spatial_mean, -nmonths)[-nmonths:]
    n_negexmonths = np.argpartition(spatial_mean, nmonths)[:nmonths]
  

    if plot_extremes:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(np.linspace(startyear,endyear-1,len(spatial_mean)),spatial_mean,'.', color='black', 
                label=f'mean winter {regdata_type} in Northern Europe')
        ax.hlines(max(spatial_mean[n_negexmonths]), xmin=startyear, xmax=endyear-1, color='blue')
        ax.hlines(min(spatial_mean[n_posexmonths]), xmin=startyear, xmax=endyear-1, color='blue')
        if (regdata_type=='Temperature'): 
            ax.set_ylabel('mean winter temperature in NE region [°C]')
        else:
            ax.set_ylabel('mean winter precipitation in NE region [kg/$m^2$/s]')
        ax.set_xlabel('Year')

    return posexmonth, negexmonth, find_month_from_index(posexmonth, startyear, endyear), find_month_from_index(negexmonth, startyear, endyear), n_posexmonths, n_negexmonths
    
    
    
    

