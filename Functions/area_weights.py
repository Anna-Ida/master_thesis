


# AREA WEIGHTS:
height_level = 5
analysis_type, season, season_mean, month, running_mean = 'seasonal', 'NDJFM', True, 'invalid', 1
sampledata ='data/ERA5/NA_zg_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412_detrended.nc'

ds = xr.open_dataset(sampledata)

zgs_detrended = ds.zg


if (len(np.shape(zgs_detrended))>3):             # if variable has several height levels
    print(f' Shape original data: {np.shape(zgs_detrended)}')

    zgs = zgs_detrended[:,height_level, :, :]    # select geopotential height at 500 hPa (= 5000 Pa)

subset_zg_data = subset_data(zgs, analysis_type, season, season_mean, month, running_mean)
         
zg = subset_zg_data

weights = np.cos(np.deg2rad(zg.lat))
weights.name = "weights"
weights

zg_weighted = zg.weighted(weights)
zg_weighted

weighted_mean = zg_weighted.mean(("lon", "lat"))
weighted_mean  