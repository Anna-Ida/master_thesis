import numpy as np
from netCDF4 import Dataset
from scipy import signal

def detrend_files(model,var,height_level,base_filename):
    '''
    var = how the required variable is called in the nc file, e.g. zg/z, tas/t2m, pr/tp
    '''
    infile = f'data/{model}/NA_{var}_{base_filename}.nc'
    dataset = Dataset(infile)

    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    tims = dataset.variables['time'][:] 

    myvar = dataset.variables[var]   
    if (len(np.shape(myvar))>3):             # select correct height level 
        myvar = myvar[:,height_level, :, :] 
        

    myvar_detrended = np.zeros((len(tims),len(lats),len(lons)))

    for i_lat in range(len(lats)):                              
        for i_lon in range(len(lons)):                          
            gridpoint_timeseries = dataset.variables[var][:,i_lat, i_lon]
            gridpoint_detrended = signal.detrend(gridpoint_timeseries)
            myvar_detrended[:,i_lat,i_lon] = gridpoint_detrended


    outfile = f'data/{model}/NA_{var}_{base_filename}_detrended_scipy'
    np.save(outfile, myvar_detrended)

    return myvar_detrended


