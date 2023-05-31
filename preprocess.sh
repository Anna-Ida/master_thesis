#!/bin/sh

#  preprocess.sh
#  
#
#  Created by Anna Kirchner on 10.09.21.
#  


# define variables:
# -----------------------------------------------------------

variable1="zg_Amon_"      # zg / z
variable2="tas_Amon_"    # tas /t2m
variable3="pr_Amon_"     # pr / tp
model="EC-Earth3"                      #"EC-Earth3" "IPSL-CM6A-LR"  "ERA5"   "CNRM-ESM2-1" "MPI-ESM1-2-HR" "UKESM1-0-LL" "MIROC6" "CESM2"
run="_historical"
variant="_r10i1p1f1"                        #"_r101i1p1f1" "_r1i1p1f1"    "_1"     "_r1i1p1f2"    "_r1i1p1f1"  "_r1i1p1f2"  "_r1i1p1f1" "_r1i1p1f1"
grid="_gr"
timeframe="_185001-201412"                #"_197001-201412" "_185001-201412" "_195901-202012" 
detrend="_detrended"
detrendedyn="_detrended"  # or ""       # to choose whether to use detrended version or not for region calculation
regridmodel="IPSL" #-CM6A-LR"
regrid="_regriddedto$regridmodel"
ending=".nc" 
                           


# name files:
# -----------------------------------------------------------
infile1="$variable1$model$run$variant$grid$timeframe$ending"    
infile2="$variable2$model$run$variant$grid$timeframe$ending"
infile3="$variable3$model$run$variant$grid$timeframe$ending"

outfile1="NA_$variable1$model$run$variant$grid$timeframe$ending"
outfile2="NA_$variable2$model$run$variant$grid$timeframe$ending"
outfile3="NA_$variable3$model$run$variant$grid$timeframe$ending"

outfile_detrended1="NA_$variable1$model$run$variant$grid$timeframe$detrend$ending"
#outfile_detrended2K="NA_tempinK$model$run$variant$grid$timeframe$detrend$ending"
outfile_detrended2="NA_$variable2$model$run$variant$grid$timeframe$detrend$ending"
outfile_detrended3="NA_$variable3$model$run$variant$grid$timeframe$detrend$ending"

newgrid="NA_zg_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412_detrended.nc"

regridoutfile1="NA_$variable1$model$run$variant$grid$timeframe$detrend$regrid$ending"
regridoutfile2="NA_$variable2$model$run$variant$grid$timeframe$detrend$regrid$ending"
regridoutfile3="NA_$variable3$model$run$variant$grid$timeframe$detrend$regrid$ending"


regionfile1="Gl_$variable1$model$run$variant$grid$timeframe$detrendedyn$ending"
regionfile2="Gl_$variable2$model$run$variant$grid$timeframe$detrendedyn$ending"
regionfile3="Gl_$variable3$model$run$variant$grid$timeframe$detrendedyn$ending"

regionoutfile1="Gl_masked_$variable1$model$run$variant$grid$timeframe$detrendedyn$ending"
regionoutfile2="Gl_masked_$variable2$model$run$variant$grid$timeframe$detrendedyn$ending"
regionoutfile3="Gl_masked_$variable3$model$run$variant$grid$timeframe$detrendedyn$ending"


Wregionfile1="GlW_$variable1$model$run$variant$grid$timeframe$detrendedyn$ending"
Wregionfile2="GlW_$variable2$model$run$variant$grid$timeframe$detrendedyn$ending"
Wregionfile3="GlW_$variable3$model$run$variant$grid$timeframe$detrendedyn$ending"

Wregionoutfile1="GlW_masked_$variable1$model$run$variant$grid$timeframe$detrendedyn$ending"
Wregionoutfile2="GlW_masked_$variable2$model$run$variant$grid$timeframe$detrendedyn$ending"
Wregionoutfile3="GlW_masked_$variable3$model$run$variant$grid$timeframe$detrendedyn$ending"

Eregionfile1="GlE_$variable1$model$run$variant$grid$timeframe$detrendedyn$ending"
Eregionfile2="GlE_$variable2$model$run$variant$grid$timeframe$detrendedyn$ending"
Eregionfile3="GlE_$variable3$model$run$variant$grid$timeframe$detrendedyn$ending"

Eregionoutfile1="GlE_masked_$variable1$model$run$variant$grid$timeframe$detrendedyn$ending"
Eregionoutfile2="GlE_masked_$variable2$model$run$variant$grid$timeframe$detrendedyn$ending"
Eregionoutfile3="GlE_masked_$variable3$model$run$variant$grid$timeframe$detrendedyn$ending"




# run commands:
# -----------------------------------------------------------
cd /Users/Anna/Documents/master_thesis/data/$model
#cd /Users/Anna/Documents/master_thesis/data/EC-Earth3/1970-2014_allzg/

# merge files:

cd /Users/Anna/Documents/master_thesis/wget_scripts

cdo mergetime zg*.nc $infile1
cdo mergetime tas*.nc $infile2
cdo mergetime pr*.nc $infile3  


cp -n $infile1 /Users/Anna/Documents/master_thesis/data/$model
cp -n $infile2 /Users/Anna/Documents/master_thesis/data/$model
cp -n $infile3 /Users/Anna/Documents/master_thesis/data/$model

rm zg*.nc
rm tas*.nc
rm pr*.nc

cd /Users/Anna/Documents/master_thesis/data/$model


# select region:

cdo sellonlatbox,-80,40,20,85 $infile1 $outfile1
cdo sellonlatbox,-80,40,20,85 $infile2 $outfile2
cdo sellonlatbox,-80,40,20,85 $infile3 $outfile3


#rm zg*.nc
#rm tas*.nc
#rm pr*.nc


# de-trend:
#<<comment
detrended1="trend_$variable1$model$run$variant$grid$timeframe$ending"  #"detrended_simple1.nc"
detrended2="trend_$variable2$model$run$variant$grid$timeframe$ending" #"detrended_simple2.nc"
detrended3="trend_$variable3$model$run$variant$grid$timeframe$ending" #"detrended_simple3.nc"

cdo -b F32 detrend $outfile1 $detrended1 
cdo -b F32 detrend $outfile2 $detrended2
cdo -b F32 detrend $outfile3 $detrended3

# and add mean back on top:
cdo -L -add $detrended1 -timmean $outfile1 $outfile_detrended1 
cdo -L -add $detrended2 -timmean $outfile2 $outfile_detrended2 
cdo -L -add $detrended3 -timmean $outfile3 $outfile_detrended3 

rm $detrended1 $detrended2 $detrended3
#rm $detrended2 $detrended3

# regrid:
cd /Users/Anna/Documents/master_thesis/data/IPSL-CM6A-LR
cp -n $newgrid /Users/Anna/Documents/master_thesis/data/$model
cd /Users/Anna/Documents/master_thesis/data/$model

cdo remapcon,$newgrid $outfile_detrended1 $regridoutfile1
cdo remapcon,$newgrid $outfile_detrended2 $regridoutfile2
cdo remapcon,$newgrid $outfile_detrended3 $regridoutfile3

#comment







# transform temperature from K to °C: (doesn't matter, I'll do it in python)
#cdo -setattribute,t2m@units="degC" -addc,-273.15 $outfile_detrended2K $outfile_detrended2
#cdo -addc,-273.15 $outfile_detrended2 $outfile_detrended2

# select region for extremes:
# Greenland: 60-15°W / 59-84°N 
# Greenland: 63-13°W / 59-84°N           
# -----------------------------------------------------------
<<comment
cdo sellonlatbox,-63,-13,59,84 $outfile_detrended1 $regionfile1
cdo sellonlatbox,-63,-13,59,84 $outfile_detrended2 $regionfile2
cdo sellonlatbox,-63,-13,59,84 $outfile_detrended3 $regionfile3


cdo -f nc setctomiss,0 -gtc,0 -remapcon,$regionfile1 -topo seamask_Gl.nc
cdo mul $regionfile1 seamask_Gl.nc $regionoutfile1
cdo -f nc setctomiss,0 -gtc,0 -remapcon,$regionfile2 -topo seamask_Gl.nc
cdo mul $regionfile2 seamask_Gl.nc $regionoutfile2
cdo -f nc setctomiss,0 -gtc,0 -remapcon,$regionfile3 -topo seamask_Gl.nc
cdo mul $regionfile3 seamask_Gl.nc $regionoutfile3



# Greenland West and East:
#cdo sellonlatbox,-63,-45,59,84 $infile $regionfileW
#cdo sellonlatbox,-45,-13,59,84 $infile $regionfileE
# -----------------------------------------------------------
cdo sellonlatbox,-63,-45,59,84 $outfile_detrended1 $Wregionfile1
cdo sellonlatbox,-63,-45,59,84 $outfile_detrended2 $Wregionfile2
cdo sellonlatbox,-63,-45,59,84 $outfile_detrended3 $Wregionfile3

cdo -f nc setctomiss,0 -gtc,0 -remapcon,$Wregionfile1 -topo seamask_Gl.nc
cdo mul $Wregionfile1 seamask_Gl.nc $Wregionoutfile1
cdo -f nc setctomiss,0 -gtc,0 -remapcon,$Wregionfile2 -topo seamask_Gl.nc
cdo mul $Wregionfile2 seamask_Gl.nc $Wregionoutfile2
cdo -f nc setctomiss,0 -gtc,0 -remapcon,$Wregionfile3 -topo seamask_Gl.nc
cdo mul $Wregionfile3 seamask_Gl.nc $Wregionoutfile3


cdo sellonlatbox,-45,-13,59,84 $outfile_detrended1 $Eregionfile1
cdo sellonlatbox,-45,-13,59,84 $outfile_detrended2 $Eregionfile2
cdo sellonlatbox,-45,-13,59,84 $outfile_detrended3 $Eregionfile3

cdo -f nc setctomiss,0 -gtc,0 -remapcon,$Eregionfile1 -topo seamask_Gl.nc
cdo mul $Eregionfile1 seamask_Gl.nc $Eregionoutfile1
cdo -f nc setctomiss,0 -gtc,0 -remapcon,$Eregionfile2 -topo seamask_Gl.nc
cdo mul $Eregionfile2 seamask_Gl.nc $Eregionoutfile2
cdo -f nc setctomiss,0 -gtc,0 -remapcon,$Eregionfile3 -topo seamask_Gl.nc
cdo mul $Eregionfile3 seamask_Gl.nc $Eregionoutfile3
comment








# OLD CODE:
# -----------------------------------------------------------


#outfile='merged_zg_Amon_EC-Earth3_historical_variant_gr_197001-201412'
#outfile=${outfile/variant/$r111i1p1f1}                                  # insert the correct variant


# merge files:

#cdo mergetime zg*.nc 'merged_zg_Amon_EC-Earth3_historical_r111i1p1f1_gr_197001-201412.nc'
#cdo mergetime pr*.nc 'merged_pr_Amon_EC-Earth3_historical_r102i1p1f1_gr_197001-201412.nc'
#cdo mergetime tas*.nc 'merged_tas_Amon_EC-Earth3_historical_r102i1p1f1_gr_197001-201412.nc'  
#rm tas*.nc



# select region:

#cdo sellonlatbox,-80,40,20,85 '' 'NA_'

#cdo sellonlatbox,-80,40,20,85 'merged_zg_Amon_EC-Earth3_historical_r111i1p1f1_gr_197001-201412.nc' 'NA_zg_Amon_EC-Earth3_historical_r111i1p1f1_gr_197001-201412.nc'
#cdo sellonlatbox,-80,40,20,85 'merged_pr_Amon_EC-Earth3_historical_r102i1p1f1_gr_197001-201412.nc' 'NA_pr_Amon_EC-Earth3_historical_r102i1p1f1_gr_197001-201412.nc'
#cdo sellonlatbox,-80,40,20,85 'merged_tas_Amon_EC-Earth3_historical_r102i1p1f1_gr_197001-201412.nc' 'NA_tas_Amon_EC-Earth3_historical_r102i1p1f1_gr_197001-201412.nc'




# de-trend:

#infile="NA_tas_Amon_EC-Earth3_historical_r102i1p1f1_gr_197001-201412.nc"
#outfile="NA_tas_Amon_EC-Earth3_historical_r102i1p1f1_gr_197001-201412_detrended.nc"

#cdo detrend ${infile} ${outfile}







