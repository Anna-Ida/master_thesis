import matplotlib as mpl
from matplotlib import gridspec,cm, rcParams, pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
from cartopy import config, crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#import sys  
#sys.path.append('Functions')
from data_selection import subset_data


def select_extremes(q1_num, q2_num, extr_perc, regdata_type, extr_infile, analysis_type, season, season_mean, month, running_mean, startyear, endyear, 
    extreme_region_name, base_title, saveextremesplotas, model, 
    extreme_timescale, plot_extremes=True):
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


    dataset = Dataset(extr_infile)
    var = dataset.variables[myvar]
    if (extreme_timescale=='seasons'):
        NE_data = subset_data(var, analysis_type, season, season_mean=True, month=month, running_mean=running_mean)    
    elif (extreme_timescale=='months'):
        NE_data = subset_data(var, analysis_type, season, season_mean=False, month=month, running_mean=running_mean)  

    if (regdata_type=='Temperature'):  # transform from K to °C
        NE_data -= 273.15
        unit = '°C'
    elif (regdata_type=='Precipitation'):  # transform to mm/day
        unit = 'mm/day'
        if (model=='ERA5'):              # from m/day 
            NE_data *= 1000
        else:
            NE_data *= 86400

    print('ToDo! Include area weights in the spatial mean calculation!! + be mindful of the masked area!')
    spatial_mean = np.mean(NE_data, axis=(1,2))    # ToDo: need to include area weights here! Plus be mindful of the masked area!
    
    q25 = spatial_mean < np.quantile(spatial_mean, q1_num)
    q75 = spatial_mean > np.quantile(spatial_mean, q2_num)
    extremes_mask = q25 + q75
    
    if plot_extremes:  
        fig, ax = plt.subplots(figsize=(6.5,4))
        if (running_mean>1):
            rmyears = np.floor(running_mean/2)
            ax.plot(np.linspace(startyear+rmyears,endyear-1-rmyears,len(spatial_mean)),spatial_mean,'.', color='black') #, 
                    #label=f'mean winter {regdata_type} in {extreme_region_name}')
        else:
            ax.plot(np.linspace(startyear,endyear-1,len(spatial_mean)),spatial_mean,'.', color='black')#, 
                    #label=f'mean winter {regdata_type} in {extreme_region_name}')
        ax.hlines(np.quantile(spatial_mean, q2_num), xmin=startyear, xmax=endyear-1, color='red', 
                  label=f'highest {extr_perc}% (>{np.quantile(spatial_mean, q2_num):.3} {unit})')
        ax.hlines(np.quantile(spatial_mean, q1_num), xmin=startyear, xmax=endyear-1, color='blue', 
                  label=f'lowest {extr_perc}% (<{np.quantile(spatial_mean, q1_num):.3} {unit})')

        if (regdata_type=='Temperature'): 
            ax.set_ylabel('mean temperature [°C]')
        else:
            ax.set_ylabel('mean precipitation [mm/day]')
        ax.set_xlabel('Year')
        ax.legend(loc='center right')
        plt.suptitle(f'Spatial mean {regdata_type} in {extreme_region_name}  \n {model}, winter {extreme_timescale} {startyear}-{endyear}', fontsize='large')
        # ToDo: implement a variant if (extreme_timescale=='months')!
        #plt.title(f'{model}', fontsize='small')
        #fig.suptitle(f'Mean winter {regdata_type} in {extreme_region_name} {timeframe} ({model})')
        #if (running_mean>1):
            #fig.suptitle(f'Mean winter {regdata_type} in {extreme_region_name} {timeframe} ({model}) \n {running_mean} year running mean')
        #fig.savefig(f'output/{model}/{timeframe}_{model}_{variant}_{season}mean_{regdata_type}quartiles({extreme_timescale})_{extreme_region}_mean{running_mean}.pdf')
        fig.savefig(f'output/{model}/extremes/scatter_{startyear}-{endyear}_{model}_{regdata_type}_{extr_perc}%_{extreme_region_name}_{extreme_timescale}.pdf')
        #fig.savefig(f'{saveextremesplotas}_{extreme_timescale}.pdf')
         
        
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





def plot_extremes_in_PCs(q25, q75, extr_perc, extreme_timescale, regdata_type, pcs,running_mean,analysis_type,model, n_eofs, timeframe, variant, extreme_region,season,season_mean,startyear,endyear,extremes,standard_title,savePC,savePCas):

    # To DO:
    # change tickmark years from float (1992.0) to int
    # add title

    plt.style.use('default')
    fig_PC, axs = plt.subplots(4, 1, figsize=(6,10))
    tickmdist = 20  # one tick mark every 10 years

    clrs = []
    for imonth in range(len(pcs)):
        if q25.data[imonth]:
            clrs.append('red')
            #clrs[x] = 'red'
        elif q75.data[imonth]:
            clrs.append('blue')
            #clrs[x] = 'blue'
        else:
            clrs.append('black')

    for row in range(4):
        ax = axs[row]
        #clrs = ['red' if (x > 0) else 'blue' for x in pcs[:,row]]
        if (running_mean>1):
            rmyears=np.floor(running_mean/2)
          #  myxaxis = np.arange(len(pcs[:,row])-rmyears-rmyears)  # shorten axis by 2 years on each side if running mean = 5
        #else:
         #   myxaxis = np.arange(len(pcs[:,row]))
        myxaxis = np.arange(len(pcs[:,row]))
        ax.bar(myxaxis, pcs[:,row], color=clrs)

        if (extreme_timescale==months):
            myticks = np.arange(0,len(myxaxis),tickmdist*5); 
            mylabels = np.arange(startyear,endyear+1,tickmdist*5) 

        elif (extreme_timescale==seasons): 
            myticks = np.arange(0,len(myxaxis),tickmdist);
            mylabels = np.arange(startyear,endyear+1,tickmdist) 

        if not extremes:
            ax.set_xticks(ticks=myticks); 
            ax.set_xticklabels(labels=mylabels);
            ax.set_xlabel('Model Years');
        
        myylim=4
        ax.set_ylim(- myylim, myylim)
        if (max(np.abs(np.amin(pcs)), np.abs(np.amax(pcs))) > myylim):
            print('!! ATTENTION: PC loading exceeds the set y-axis range !!')   
        
        ax.margins(x=0.005, y=0)
        ax.set_title(f'PC {row+1}', loc='left', fontsize='xx-large');

        if (row==0):
            if (regdata_type=='Temperature'):
                colors = {'warmest winters':'red', 'coldest winters':'blue'}   
            elif (regdata_type=='Precipitation'):
                colors = {'wettest winters':'red', 'driest winters':'blue'}      
            labels = list(colors.keys())
            handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
            ax.legend(handles, labels, fontsize= 'x-small')


    
    fig_PC.suptitle(standard_title)

    fig_PC.tight_layout()
    if savePC:
        savePCas = f'output/{model}/extremes/{n_eofs}PCs_{timeframe}_{model}_{variant}_{season}_mean{running_mean}_{regdata_type}{extr_perc}%{extreme_timescale}_{extreme_region}_marked.pdf'
        fig_PC.savefig(savePCas)



def plot_mean_fields_of_extr_months(zg_data, precip_data, temp_data, lons, lats, extr_type, extreme_region, extreme_region_name, extreme_timescale, imon, imon_name, model, variant, timeframe):
    #(data,lons,lats,season,timeframe,model,variant,standard_title,savemean,savemeanas):
    
    #plt.style.use('ggplot')
    plt.style.use('default')
    fig_means, axs = plt.subplots(3, 1, figsize=(5,12),subplot_kw={'projection': ccrs.EquidistantConic(central_longitude=-20.0)})

    for row in range(3):
        ax = axs[row]
        if (row==0):
            cmap = mpl.cm.coolwarm 
            col_mean = zg_data #np.mean(zg_data, axis=0)
            colorlevels = np.linspace(5000,5900,10) 
            cbarlab = 'geopotential height at 500 hPa [m]'
        elif (row==1):
            cmap = mpl.cm.viridis 
            col_mean = precip_data #np.mean(precip_data, axis=0)
            colorlevels = np.linspace(0,9,10) 
            cbarlab = 'Precipitation [mm/day]'
        elif (row==2):
            cmap = mpl.cm.coolwarm 
            col_mean = temp_data #np.mean(temp_data, axis=0)
            colorlevels = np.linspace(-50,50,11) 
            cbarlab = 'Temperature [°C]'

        im = ax.contourf(lons, lats, col_mean, levels=colorlevels, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
        ax.coastlines()
        cbar = fig_means.colorbar(im, ax=axs[row])
        cbar.set_label(cbarlab)  
        
        ax.set(title=f'{extr_type} {extreme_timescale} {extreme_region_name} -  {imon_name} \n ({model} {variant})')
        #fig_mean.suptitle(f'Mean field of geopotential height at 500 hPa', fontsize='x-large')
        #plt.text(.5, 0.9, standard_title, transform=fig_mean.transFigure, fontsize='large', horizontalalignment='center')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.ylabel_style = {'size': 15, 'color': 'gray'}


    savemeanas = f'output/{model}/extremes/{extr_type}_{imon_name}_meanpatterns_{timeframe}_{model}_{variant}_{extreme_region}.pdf'
    fig_means.savefig(savemeanas)









