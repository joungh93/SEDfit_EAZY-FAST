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
dir_fast = "/home/jlee/Downloads/fastpp/"
dir_src = dir_eazy+"src/"
dir_flt = dir_eazy+"filters/"
dir_temp = dir_eazy+"templates/"
dir_fsps = dir_temp+"fsps_full/"

dir_lib = "/home/jlee/Downloads/FAST_v1.0/Libraries"

dir_eazy_input = "EAZY_INPUT/"
dir_fast_input = "FAST_INPUT/"
dir_fast_output = "FAST_OUTPUT/"
if (glob.glob(dir_fast_output) == []):
    os.system("mkdir "+dir_fast_output)
else:
    os.system("rm -rfv "+dir_fast_output+"*")

catalogs = ['flux_EAZY_hst.cat', 'flux_EAZY_jwst.cat', 'flux_EAZY_total.cat']
cat_fast = ['photz_hst.cat', 'photz_jwst.cat', 'photz_total.cat']
cat_zout = ['photz_hst.zout', 'photz_jwst.zout', 'photz_total.zout']
for i in range(len(catalogs)):
    os.system("cp -rpv "+dir_eazy_input+catalogs[i]+" "+dir_fast_input+cat_fast[i])
    os.system("cp -rpv "+dir_eazy_input+catalogs[i]+" ./"+cat_fast[i])
    os.system("cp -rpv "+dir_fast_input+cat_zout[i]+" ./"+cat_zout[i])


'''
#... FAST++ V1.2: parameter file .......................................

VERBOSE         = 1         # 0 / 1
PARALLEL        = 'none'    # 'none', 'generators', 'models', or 'sources'
N_THREAD        = 0
MAX_QUEUED_FITS = 1000

#--- BROADBAND PHOTOMETRIC INFORMATION ---------------------------------
CATALOG        = 'hdfn_fs99'
AB_ZEROPOINT   = 25.
FILTERS_RES    = '../../share/FILTER.RES.latest'
FILTER_FORMAT  = 1
TEMP_ERR_FILE  = '../../share/TEMPLATE_ERROR.fast.v0.2'
NAME_ZPHOT     = 'z_m2'
FORCE_ZPHOT    = 0          # 0 / 1
BEST_AT_ZPHOT  = 1          # 0 / 1
ZPHOT_CONF     = 68         # 68 / 95 / 99
USE_LIR        = 0          # 0 / 1


#--- SPECTROSCOPIC INFORMATION -----------------------------------------
SPECTRUM       = ''
AUTO_SCALE     = 0          # 0 / 1
APPLY_VDISP    = 0          # km/s


#--- OUTPUT INFORMATION  -----------------------------------------------
OUTPUT_DIR         = ''
OUTPUT_FILE        = ''
N_SIM              = 100
C_INTERVAL         = 68            # 68 / 95 / 99 or [68,95] etc
BEST_FIT           = 0             # 0 / 1
BEST_FROM_SIM      = 0             # 0 / 1
INTERVAL_FROM_CHI2 = 0             # 0 / 1
SAVE_SIM           = 0             # 0 / 1
SFR_AVG            = 0             # 0, 100 Myr, 300 Myr etc
INTRINSIC_BEST_FIT = 0             # 0 / 1
BEST_SFHS          = 0             # 0 / 1
SFH_OUTPUT_STEP    = 10            # 10 Myr, 100 Myr etc
SFH_OUTPUT         = 'sfr'         # 'sfr' or 'mass'
REST_MAG           = []            # [140,142,161] for UVJ colors
CONTINUUM_INDICES  = ''
SFH_QUANTITIES     = []            # ['tquench10','tform50','brate10', ...]
OUTPUT_COLUMNS     = []            # ['id','Av','lmass','lsfr', ...]


#--- CHOOSE STELLAR POPULATIONS LIBRARY --------------------------------
LIBRARY_DIR         = '../../share/libraries/'
LIBRARY             = 'bc03'         # 'bc03' / 'ma05' / 'co11'
RESOLUTION          = 'hr'           # 'pr' / 'lr' / 'hr'
IMF                 = 'ch'           # 'ch' / 'sa' / 'kr'
SFH                 = 'del'          # 'exp' / 'del' / 'tru'
DUST_LAW            = 'calzetti'     # 'calzetti' / 'mw' / 'kc' / 'noll'
# E_B               = 1              # only define for 'noll' dust law
# delta             = -0.2           # only define for 'noll' dust law
MY_SFH              = ''
CUSTOM_SFH          = ''             # '(1 + t)^alpha'
CUSTOM_PARAMS       = []             # ['alpha', ...]
CUSTOM_SFH_LOOKBACK = 0              # 0 / 1
# ALPHA_MIN         = -2             # define these for each parameter
# ALPHA_MAX         = 2              # define these for each parameter
# ALPHA_STEP        = 0.1            # define these for each parameter


#--- DEFINE GRID -------------------------------------------------------
LOG_TAU_MIN      = 8.5            # log [yr]
LOG_TAU_MAX      = 10.            # log [yr]
LOG_TAU_STEP     = 0.5            # log [yr], min 0.1
LOG_AGE_MIN      = 8.0            # log [yr]
LOG_AGE_MAX      = 10.0           # log [yr]
LOG_AGE_STEP     = 0.2            # log [yr]
NO_MAX_AGE       = 0              # 0 / 1
Z_MIN            = 0.01           # Cannot be 0.0
Z_MAX            = 6.00
Z_STEP           = 0.05
Z_STEP_TYPE      = 0              # 0: Z_STEP, 1: Z_STEP*(1+z)
A_V_MIN          = 0.             # [mag]
A_V_MAX          = 3.             # [mag]
A_V_STEP         = 0.1            # [mag]
A_V_BC_MIN       = 0.             # [mag]
A_V_BC_MAX       = 0.             # [mag]
A_V_BC_STEP      = 0.1            # [mag]
DIFFERENTIAL_A_V = 0              # 0 / 1
LOG_BC_AGE_MAX   = 7.0            # log [yr]
METAL            = [0.02]         # [0.0096,0.019,0.03]
NO_CACHE         = 0              # 0 / 1


#--- COSMOLOGY ---------------------------------------------------------
H0             = 70.0               # Hubble constant
OMEGA_M        = 0.3                # Omega matter
OMEGA_L        = 0.7                # Omega lambda
NO_IGM         = 0                  # 0 / 1


#--- SAVE INTERMEDIATE PRODUCTS ----------------------------------------
SAVE_CHI_GRID  = 0          # 0 / 1
SAVE_BESTCHI   = 0          # 1 (68%) / 2.71 (90%) / 6.63 (99%) / etc
'''


# ----- Writing new parameter file ----- #
def fast_input(param_file, catalog, output_prefix, dir_output, magzero=23.93,
               parallel='none', n_thread=0, max_queued_fits=1000,
               filters=dir_flt+"FILTER.RES.latest", temperr_file=dir_temp+"uvista_nmf/template_error_10.def",
               name_zphot='z_peak', force_zphot=1, best_at_zphot=1, zphot_conf=68,
               dir_lib=dir_fast+"Libraries/", library='bc03', resolution='lr', imf='ch', sfh='del', dust='calzetti',
               tau_min=7.0, tau_max=10.0, tau_step=0.1, age_min=8.0, age_max=10.1, age_step=0.1,
               zmin=0.01, zmax=12.0, zstep=0.01, Av_min=0.0, Av_max=6.0, Av_step=0.1, metal=[0.019],
               H0=70, Omega_M=0.3, Omega_L=0.7):

    f = open(param_file, "w")
    f.write("#### FAST++ V1.2: parameter file\n\n")
    f.write("VERBOSE             = 1\n")
    f.write("PARALLEL            = '"+parallel+"'\n")
    f.write(f"N_THREAD            = {n_thread:d}\n")
    f.write(f"MAX_QUEUED_FITS     = {max_queued_fits:d}\n\n")

    #--- BROADBAND PHOTOMETRIC INFORMATION
    f.write("## Broadband Photometric Information\n")
    f.write("CATALOG             = '"+catalog.split('.cat')[0]+"'\n")
    f.write(f"AB_ZEROPOINT        = {magzero:.2f}\n")
    f.write("FILTERS_RES         = '"+filters+"'\n")
    f.write("FILTER_FORMAT       = 1\n")
    f.write("TEMP_ERR_FILE       = '"+temperr_file+"'\n")
    f.write("NAME_ZPHOT          = '"+name_zphot+"'\n")
    f.write("FORCE_ZPHOT         = "+f"{force_zphot:d}"+"\n")
    f.write("BEST_AT_ZPHOT       = "+f"{best_at_zphot:d}"+"\n")
    f.write("ZPHOT_CONF          = "+f"{zphot_conf:d}"+"\n")
    f.write("USE_LIR             = 0\n\n")

    #--- SPECTROSCOPIC INFORMATION
    f.write("## Spectroscopic Information\n")
    f.write("SPECTRUM            = ''\n")
    f.write("AUTO_SCALE          = 0\n")
    f.write("APPLY_VDISP         = 0\n\n")

    #--- OUTPUT INFORMATION
    f.write("## Output Information\n")
    f.write("OUTPUT_DIR          = '"+dir_output+"'\n")
    f.write("OUTPUT_FILE         = '"+output_prefix+"'\n")
    f.write("N_SIM               = 100\n")
    f.write("C_INTERVAL          = 68\n")
    f.write("BEST_FIT            = 1\n")
    f.write("BEST_FROM_SIM       = 0\n")
    f.write("INTERVAL_FROM_CHI2  = 0\n")
    f.write("SAVE_SIM            = 0\n")
    f.write("SFR_AVG             = 0\n")
    f.write("INTRINSIC_BEST_FIT  = 0\n")
    f.write("BEST_SFHS           = 0\n")
    f.write("SFH_OUTPUT_STEP     = 10\n")
    f.write("SFH_OUTPUT          = 'sfr'\n")
    f.write("REST_MAG            = []\n")
    f.write("CONTINUUM_INDICES   = ''\n")
    f.write("SFH_QUANTITIES      = []\n")
    f.write("OUTPUT_COLUMNS      = []\n\n")

    #--- CHOOSE STELLAR POPULATIONS LIBRARY
    f.write("## Stellar Populations Library\n")
    if (dir_lib[-1] == "/"):
        dir_lib = dir_lib[:-1]
    f.write("LIBRARY_DIR         = '"+dir_lib+"'\n")
    f.write("LIBRARY             = '"+library+"'\n")
    f.write("RESOLUTION          = '"+resolution+"'\n")
    f.write("IMF                 = '"+imf+"'\n")
    f.write("SFH                 = '"+sfh+"'\n")
    f.write("DUST_LAW            = '"+dust+"'\n")
    f.write("# E_B               = 1\n")
    f.write("# delta             = -0.2\n")
    f.write("MY_SFH              = ''\n")
    f.write("CUSTOM_SFH          = ''\n")
    f.write("CUSTOM_PARAMS       = []\n")
    f.write("CUSTOM_SFH_LOOKBACK = 0\n")
    f.write("# ALPHA_MIN         = -2\n")
    f.write("# ALPHA_MAX         = 2\n")
    f.write("# ALPHA_STEP        = 0.1\n\n")

    #--- DEFINE GRID
    f.write("## Define Grid\n")
    f.write(f"LOG_TAU_MIN         = {tau_min:.1f}\n")
    f.write(f"LOG_TAU_MAX         = {tau_max:.1f}\n")
    f.write(f"LOG_TAU_STEP        = {tau_step:.2f}\n")
    f.write(f"LOG_AGE_MIN         = {age_min:.1f}\n")
    f.write(f"LOG_AGE_MAX         = {age_max:.1f}\n")
    f.write(f"LOG_AGE_STEP        = {age_step:.1f}\n")
    f.write("NO_MAX_AGE          = 0\n")
    f.write(f"Z_MIN               = {zmin:.2f}\n")
    f.write(f"Z_MAX               = {zmax:.2f}\n")
    f.write(f"Z_STEP              = {zstep:.2f}\n")
    f.write("Z_STEP_TYPE         = 1\n")
    f.write(f"A_V_MIN             = {Av_min:.1f}\n")
    f.write(f"A_V_MAX             = {Av_max:.1f}\n")
    f.write(f"A_V_STEP            = {Av_step:.2f}\n")
    f.write("A_V_BC_MIN          = 0.\n")
    f.write("A_V_BC_MAX          = 0.\n")
    f.write("A_V_BC_STEP         = 0.1\n")
    f.write("DIFFERENTIAL_A_V    = 0\n")
    f.write("LOG_BC_AGE_MAX      = 7.0\n")
    if (len(metal) == 1):
        metallicity = f"[{metal[0]:.4f}]"
    elif (len(metal) > 1):
        metallicity = "["
        for i in range(len(metal)):
            metallicity += f"{metal[i]:.4f},"
        metallicity = metallicity[:-1]+"]"
    f.write("METAL               = "+metallicity+"\n")
    f.write("NO_CACHE            = 0\n\n")

    #--- COSMOLOGY
    f.write("## Cosmology\n")
    f.write(f"H0                  = {H0:.1f}\n")
    f.write(f"OMEGA_M             = {Omega_M:.1f}\n")
    f.write(f"OMEGA_L             = {Omega_L:.1f}\n")
    f.write("NO_IGM              = 0\n\n")

    #--- SAVE INTERMEDIATE PRODUCTS 
    f.write("## Save Intermediate Products\n")
    f.write("SAVE_CHI_GRID       = 0\n")
    f.write("SAVE_BESTCHI        = 0\n\n")

    f.close()

fsParam = {'magzero':23.90, 'parallel':'none', 'n_thread':0, 'max_queued_fits':1000,
           'filters':dir_flt+"FILTER.RES.latest", 'temperr_file':dir_temp+"uvista_nmf/template_error_10.def",
           'name_zphot':'z_peak', 'force_zphot':1, 'best_at_zphot':1, 'zphot_conf':68,
           'dir_lib':dir_lib, 'library':'bc03', 'resolution':'lr', 'imf':'ch', 'sfh':'del', 'dust':'calzetti',
           'tau_min':8.5, 'tau_max':10.0, 'tau_step':0.5, 'age_min':8.0, 'age_max':10.0, 'age_step':0.2,
           'zmin':0.01, 'zmax':12.0, 'zstep':0.01, 'Av_min':0.0, 'Av_max':6.0, 'Av_step':0.1, 'metal':[0.02],
           'H0':68.4, 'Omega_M':0.3, 'Omega_L':0.7}

param_file = ["fast_hst.param", "fast_jwst.param", "fast_total.param"]
output_prefix = ["FAST_RESULT_hst", "FAST_RESULT_jwst", "FAST_RESULT_total"]

os.system("rm -rfv "+dir_fast_output+"*")
os.system("rm -rfv bc03*")
g = open("./fast.log", "w")
for i in range(3):
    fast_input(param_file[i], cat_fast[i], output_prefix[i], dir_fast_output, **fsParam)
    os.system(dir_fast+"bin/fast++ "+param_file[i])
    g.write("fastpp "+param_file[i]+"\n")
g.close()

os.system("rm -rfv *.cat *.zout")

# Printing the running time
print(f"--- {time.time()-start_time:.4f} sec ---")
