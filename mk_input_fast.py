#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 21:38:55 2022

@author: jlee
"""

import time
start_time = time.time()

import numpy as np
import glob, os, copy
import pandas as pd


# Copying the default parameter file to the current diretory
dir_eazy = "/home/jlee/Downloads/eazy-photoz/"
dir_fast = "/home/jlee/Downloads/FAST_v1.0/"
dir_src = dir_eazy+"src/"
dir_flt = dir_eazy+"filters/"
dir_temp = dir_eazy+"templates/"
dir_fsps = dir_temp+"fsps_full/"

dir_eazy_input = "EAZY_INPUT/"
dir_fast_input = "FAST_INPUT/"
dir_fast_output = "FAST_OUTPUT/"

catalogs = ['flux_EAZY_hst.cat', 'flux_EAZY_jwst.cat', 'flux_EAZY_total.cat']
cat_fast = ['photz_hst.cat', 'photz_jwst.cat', 'photz_total.cat']
for i in range(len(catalogs)):
    os.system("cp -rpv "+dir_eazy_input+catalogs[i]+" "+dir_fast_input+cat_fast[i])


'''
#... FAST V1.0: parameter file .........................................


#--- GENERAL INFORMATION -----------------------------------------------
#
# Please read this parameter file in detail, you can find all relevant 
# information here. Note that you have to adjust your input catalogs
# accordingly, otherwise FAST will not work properly!
#
# o Requirements:
#   - ~2.0 Gb memory (or more depending on the grid size)
#   - idl (make sure IDL is properly installed, with IDL_DIR defined)
#
# o The main (example_phot or example_spec) directory should contain 
#   the following files:
#   - Parameter file 
#   - [CATALOG].cat    If you fit broadband photometry
#   - [CATALOG].zout       If you input photometric redshifts
#   - [CATALOG].translate  If you input a translate file
#   - [FILTERS_RES]    If you fit broadband photometry 
#   - [SPECTRUM].spec      If you fit spectra
#
# o FAST can be run from the command line in the example directory. 
#   The first argument is the parameter file. Default is 'fast.param' 
#   $ ../fast       
#   $ ../fast my_fast.param
#
#-----------------------------------------------------------------------
 
#--- BROADBAND PHOTOMETRIC INFORMATION ---------------------------------
CATALOG        = 'hdfn_fs99'
AB_ZEROPOINT   = 25.            
FILTERS_RES    = '../Filters/FILTER.RES.v6.R300'
FILTER_FORMAT  = 1
TEMP_ERR_FILE  = '../Template_error/TEMPLATE_ERROR.fast.v0.2'
NAME_ZPHOT     = 'z_m2'

#--- SPECTROSCOPIC INFORMATION -----------------------------------------
SPECTRUM       = ''
AUTO_SCALE     = 0          # 0 / 1

#--- OUTPUT INFORMATION  -----------------------------------------------
OUTPUT_DIR     = ''
OUTPUT_FILE    = ''
N_SIM          = 0
C_INTERVAL     = 68         # 68 / 95 / 99 or [68,95] etc
BEST_FIT       = 1          # 0 / 1

#--- CHOOSE STELLAR POPULATIONS LIBRARY --------------------------------
LIBRARY_DIR    = '../Libraries/'
LIBRARY        = 'co09'         # 'bc03' / 'ma05' / 'co11'
RESOLUTION     = 'hr'           # 'pr' / 'lr' / 'hr'
IMF            = 'ch'           # 'ch' / 'sa' / 'kr'
SFH        = 'del'          # 'exp' / 'del' / 'tru'
DUST_LAW       = 'kc'           # 'calzetti' / 'mw' / 'kc' / 'noll'
# E_B          = 1          # only define for 'noll' dust law
# delta        = -0.2           # only define for 'noll' dust law
MY_SFH         = '' 

#--- DEFINE GRID -------------------------------------------------------
LOG_TAU_MIN    = 8.5            # log [yr]
LOG_TAU_MAX    = 10.            # log [yr]
LOG_TAU_STEP   = 0.5            # log [yr], min 0.1
LOG_AGE_MIN    = 8.0            # log [yr]
LOG_AGE_MAX    = 10.0           # log [yr]
LOG_AGE_STEP   = 0.2            # log [yr]
NO_MAX_AGE     = 0          # 0 / 1
Z_MIN          = 0.01           # Cannot be 0.  
Z_MAX          = 6.00   
Z_STEP         = 0.05
Z_STEP_TYPE    = 0          # 0: Z_STEP, 1: Z_STEP*(1+z)
A_V_MIN        = 0.         # [mag]
A_V_MAX        = 3.                 # [mag]
A_V_STEP       = 0.1            # [mag]
METAL          = [0.019]            # [0.0096,0.019,0.03]

#--- COSMOLOGY ---------------------------------------------------------
H0             = 70.0               # Hubble constant
OMEGA_M        = 0.3                # Omega matter
OMEGA_L        = 0.7                # Omega lambda 

#--- SAVE INTERMEDIATE PRODUCTS ----------------------------------------
SAVE_CHI_GRID  = 0          # 0 / 1
'''


# ----- Writing new parameter file ----- #
def fast_input(param_file, catalog, output_prefix, dir_output, magzero=23.93,
               filters=dir_flt+"FILTER.RES.latest", temperr_file=dir_temp+"uvista_nmf/template_error_10.def",
               dir_lib=dir_fast+"Libraries/", library='bc03', resolution='lr', imf='ch', sfh='del', dust='calzetti',
               tau_min=7.0, tau_max=10.0, tau_step=0.1, age_min=8.0, age_max=10.1, age_step=0.1,
               zmin=0.01, zmax=12.0, zstep=0.01, Av_min=0.0, Av_max=6.0, Av_step=0.1, metal=[0.019],
               H0=70, Omega_M=0.3, Omega_L=0.7):

    f = open(param_file, "w")
    f.write("#### FAST V1.0: parameter file\n\n")

    #--- BROADBAND PHOTOMETRIC INFORMATION
    f.write("## Broadband Photometric Information\n")
    f.write("CATALOG        = '"+catalog.split('.cat')[0]+"'\n")
    f.write(f"AB_ZEROPOINT   = {magzero:.2f}\n")
    f.write("FILTERS_RES    = '"+filters+"'\n")
    f.write("FILTER_FORMAT  = 1\n")
    f.write("TEMP_ERR_FILE  = '"+temperr_file+"'\n")
    f.write("NAME_ZPHOT     = 'z_peak'\n\n")

    #--- SPECTROSCOPIC INFORMATION
    f.write("## Spectroscopic Information\n")
    f.write("SPECTRUM       = ''\n")
    f.write("AUTO_SCALE     = 0\n\n")

    #--- OUTPUT INFORMATION
    f.write("## Output Information\n")
    f.write("OUTPUT_DIR     = '"+dir_output+"'\n")
    f.write("OUTPUT_FILE    = '"+output_prefix+"'\n")
    f.write("N_SIM          = 0\n")
    f.write("C_INTERVAL     = [68]\n")
    f.write("BEST_FIT       = 1\n\n")

    #--- CHOOSE STELLAR POPULATIONS LIBRARY
    f.write("## Stellar Populations Library\n")
    if (dir_lib[-1] == "/"):
        dir_lib = dir_lib[:-1]
    f.write("LIBRARY_DIR    = '"+dir_lib+"'\n")
    f.write("LIBRARY        = '"+library+"'\n")
    f.write("RESOLUTION     = '"+resolution+"'\n")
    f.write("IMF            = '"+imf+"'\n")
    f.write("SFH            = '"+sfh+"'\n")
    f.write("DUST_LAW       = '"+dust+"'\n")
    f.write("# E_B          = 1\n")
    f.write("# delta        = -0.2\n")
    f.write("MY_SFH         = ''\n\n")

    #--- DEFINE GRID
    f.write("## Define Grid\n")
    f.write(f"LOG_TAU_MIN    = {tau_min:.1f}\n")
    f.write(f"LOG_TAU_MAX    = {tau_max:.1f}\n")
    f.write(f"LOG_TAU_STEP   = {tau_step:.2f}\n")
    f.write(f"LOG_AGE_MIN    = {age_min:.1f}\n")
    f.write(f"LOG_AGE_MAX    = {age_max:.1f}\n")
    f.write(f"LOG_AGE_STEP   = {age_step:.1f}\n")
    f.write("NO_MAX_AGE     = 0\n")
    f.write(f"Z_MIN          = {zmin:.2f}\n")
    f.write(f"Z_MAX          = {zmax:.2f}\n")
    f.write(f"Z_STEP         = {zstep:.2f}\n")
    f.write("Z_STEP_TYPE    = 1\n")
    f.write(f"A_V_MIN        = {Av_min:.1f}\n")
    f.write(f"A_V_MAX        = {Av_max:.1f}\n")
    f.write(f"A_V_STEP       = {Av_step:.2f}\n")
    if (len(metal) == 1):
        metallicity = f"[{metal[0]:.4f}]"
    elif (len(metal) > 1):
        metallicity = "["
        for i in range(len(metal)):
            metallicity += f"{metal[i]:.4f},"
        metallicity = metallicity[:-1]+"]"
    f.write("METAL          = "+metallicity+"\n\n")

    #--- COSMOLOGY
    f.write("## Cosmology\n")
    f.write(f"H0             = {H0:.1f}\n")
    f.write(f"OMEGA_M        = {Omega_M:.1f}\n")
    f.write(f"OMEGA_L        = {Omega_L:.1f}\n\n")

    #--- SAVE INTERMEDIATE PRODUCTS 
    f.write("## Save Intermediate Products\n")
    f.write("SAVE_CHI_GRID  = 1\n")

    f.close()

fsParam = {'magzero':23.90,
           'filters':dir_flt+"FILTER.RES.latest", 'temperr_file':dir_temp+"uvista_nmf/template_error_10.def",
           'dir_lib':dir_fast+"Libraries/", 'library':'bc03', 'resolution':'lr', 'imf':'ch', 'sfh':'del', 'dust':'calzetti',
           'tau_min':7.0, 'tau_max':10.0, 'tau_step':0.1, 'age_min':8.0, 'age_max':10.1, 'age_step':0.1,
           'zmin':0.01, 'zmax':12.0, 'zstep':0.01, 'Av_min':0.0, 'Av_max':6.0, 'Av_step':0.1, 'metal':[0.02]}

param_file = ["fast_hst.param", "fast_jwst.param", "fast_total.param"]
output_prefix = ["FAST_RESULT_hst", "FAST_RESULT_jwst", "FAST_RESULT_total"]

os.system("rm -rfv "+dir_fast_output+"*")
os.system("rm -rfv bc03*")
g = open("./fast.log", "w")
for i in range(3):
    fast_input(param_file[i], dir_fast_input+cat_fast[i], output_prefix[i], dir_fast_output, **fsParam)
    os.system(dir_fast+"fast "+param_file[i])
    g.write("fast "+param_file[i]+"\n")
g.close()

