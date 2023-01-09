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


# Copying the default parameter file to the current diretory
dir_eazy = "/home/jlee/Downloads/eazy-photoz/"
dir_src = dir_eazy+"src/"
dir_flt = dir_eazy+"filters/"
dir_temp = dir_eazy+"templates/"
dir_fsps = dir_temp+"fsps_full/"

dir_eazy_input = "EAZY_INPUT/"
dir_fast_input = "FAST_INPUT/"
if (glob.glob(dir_fast_input) == []):
    os.system("mkdir "+dir_fast_input)
else:
    os.system("rm -rfv "+dir_fast_input+"*")

temp_file = "tweak_fsps_QSF_12_v3.param"
param_file = dir_src+"zphot.param.default"
os.system("cp -rpv "+param_file+" .")


'''
#### EAZY Default parameters

## Filters
FILTERS_RES          FILTER.RES.latest  # Filter transmission data
FILTER_FORMAT        1                  # Format of FILTERS_RES file -- 0: energy-  1: photon-counting detector
SMOOTH_FILTERS       n                  # Smooth filter curves with Gaussian
SMOOTH_SIGMA         100.               # Gaussian sigma (in Angstroms) to smooth filters

## Templates
TEMPLATES_FILE       templates/eazy_v1.2_dusty.spectra.param # Template definition file
TEMPLATE_COMBOS      a                  # Template combination options: 
                                        #         1 : one template at a time
                                        #         2 : two templates, read allowed combinations from TEMPLATES_FILE
                                        #        -2 : two templates, all permutations
                                        # a <or> 99 : all templates simultaneously
NMF_TOLERANCE        1.e-4              # Tolerance for non-negative combinations (TEMPLATE_COMBOS=a)
WAVELENGTH_FILE      templates/EAZY_v1.1_lines/lambda_v1.1.def # Wavelength grid definition file
TEMP_ERR_FILE        templates/TEMPLATE_ERROR.eazy_v1.0 # Template error definition file
TEMP_ERR_A2          0.50               # Template error amplitude
SYS_ERR              0.00               # Systematic flux error (% of flux)
APPLY_IGM            y                  # Apply IGM absorption (1/y=Inoue2014, 2/x=Madau1995)
LAF_FILE             templates/LAFcoeff.txt # File containing the Lyman alpha forest data from Inoue(2014)
DLA_FILE             templates/DLAcoeff.txt # File containing the damped Lyman absorber data from Inoue(2014)
SCALE_2175_BUMP      0.00               # Scaling of 2175A bump.  Values 0.13 (0.27) absorb ~10 (20) % at peak.

DUMP_TEMPLATE_CACHE  n                  # Write binary template cache
USE_TEMPLATE_CACHE   n                  # Load in template cache
CACHE_FILE           photz.tempfilt     # Template cache file (in OUTPUT_DIRECTORY)

## Input Files
CATALOG_FILE         hdfn_fs99_eazy.cat # Catalog data file
MAGNITUDES           n                  # Catalog photometry in magnitudes rather than f_nu fluxes
NOT_OBS_THRESHOLD    -90                # Ignore flux point if <NOT_OBS_THRESH
N_MIN_COLORS         5                  # Require N_MIN_COLORS to fit

## Output Files
OUTPUT_DIRECTORY     OUTPUT             # Directory to put output files in
MAIN_OUTPUT_FILE     photz              # Main output file, .zout
PRINT_ERRORS         y                  # Print 68, 95 and 99% confidence intervals
CHI2_SCALE           1.0                # Scale ML Chi-squared values to improve confidence intervals
VERBOSE_LOG          y                  # Dump information from the run into [MAIN_OUTPUT_FILE].param
OBS_SED_FILE         n                  # Write out observed SED/object, .obs_sed
TEMP_SED_FILE        n                  # Write out best template fit/object, .temp_sed
POFZ_FILE            n                  # Write out Pofz/object, .pz
BINARY_OUTPUT        y                  # Save OBS_SED, TEMP_SED, PZ in binary format to read with e.g IDL

## Redshift / Mag prior
APPLY_PRIOR          y                  # Apply apparent magnitude prior
PRIOR_FILE           templates/prior_K_extend.dat # File containing prior grid
PRIOR_FILTER         28                 # Filter from FILTER_RES corresponding to the columns in PRIOR_FILE
PRIOR_ABZP           25.0               # AB zeropoint of fluxes in catalog.  Needed for calculating apparent mags!

## Redshift Grid
FIX_ZSPEC            n                  # Fix redshift to catalog zspec
Z_MIN                0.01               # Minimum redshift
Z_MAX                6.0                # Maximum redshift
Z_STEP               0.01               # Redshift step size
Z_STEP_TYPE          1                  #  0 = ZSTEP, 1 = Z_STEP*(1+z)

## Zeropoint Offsets
GET_ZP_OFFSETS       n                  # Look for zphot.zeropoint file and compute zeropoint offsets
ZP_OFFSET_TOL        1.e-4              # Tolerance for iterative fit for zeropoint offsets [not implemented]

## Rest-frame colors
REST_FILTERS         ---                # Comma-separated list of rest frame filters to compute
RF_PADDING           1000.              # Padding (Ang) for choosing observed filters around specified rest-frame pair.
RF_ERRORS            n                  # Compute RF color errors from p(z)
Z_COLUMN             z_peak             # Redshift to use for rest-frame color calculation (z_a, z_p, z_m1, z_m2, z_peak)
USE_ZSPEC_FOR_REST   y                  # Use z_spec when available for rest-frame colors
READ_ZBIN            n                  # Get redshifts from OUTPUT_DIRECTORY/MAIN_OUTPUT_FILE.zbin rather than fitting them.

## Cosmology
H0                   70.0               # Hubble constant (km/s/Mpc)
OMEGA_M              0.3                # Omega_matter
OMEGA_L              0.7                # Omega_lambda
'''


# ----- Writing new parameter file ----- #
def eazy_input(param_file, catalog, dir_output, output_file,
               filters=dir_flt+"FILTER.RES.latest", temp_file=temp_file,
               wave_file=dir_temp+"uvista_nmf/lambda.def", temperr_file=dir_temp+"uvista_nmf/template_error_10.def",
               laf_file=dir_temp+"LAFcoeff.txt", dla_file=dir_temp+"DLAcoeff.txt",
               zmin=0.01, zmax=12.0, zstep=0.01, H0=68.4, Omega_M=0.3, Omega_L=0.7):

    f = open(param_file, "w")
    f.write("#### EAZY Default parameters\n\n")

    # Filters
    f.write("## Filters\n")
    f.write("FILTERS_RES         "+filters+"\n")
    f.write("FILTER_FORMAT       1\n")
    f.write("SMOOTH_FILTERS      n\n")
    f.write("SMOOTH_SIGMA        100.\n\n")

    # Templates
    f.write("## Templates\n")
    f.write("TEMPLATES_FILE      "+temp_file+"\n")
    f.write("TEMPLATE_COMBOS     a\n")
    f.write("NMF_TOLERANCE       1.e-4\n")
    f.write("WAVELENGTH_FILE     "+wave_file+"\n")
    f.write("TEMP_ERR_FILE       "+temperr_file+"\n")
    f.write("TEMP_ERR_A2         0.20\n")
    f.write("SYS_ERR             0.01\n")
    f.write("APPLY_IGM           y\n")
    f.write("LAF_FILE            "+laf_file+"\n")
    f.write("DLA_FILE            "+dla_file+"\n")
    f.write("SCALE_2175_BUMP     0.00\n")
    f.write("DUMP_TEMPLATE_CACHE n\n")
    f.write("USE_TEMPLATE_CACHE  n\n")
    f.write("CACHE_FILE          photz.tempfilt\n\n")

    # Input files
    f.write("## Input Files\n")
    f.write("CATALOG_FILE        "+catalog+"\n")
    f.write("MAGNITUDES          n\n")
    f.write("NOT_OBS_THRESHOLD   -90\n")
    f.write("N_MIN_COLORS        3\n\n")

    # Output files
    f.write("## Output Files\n")
    if (dir_output[-1] == "/"):
        dir_output = dir_output[:-1]
    f.write("OUTPUT_DIRECTORY    "+dir_output+"\n")
    f.write("MAIN_OUTPUT_FILE    "+output_file+"\n")
    f.write("PRINT_ERRORS        y\n")
    f.write("CHI2_SCALE          1.0\n")
    f.write("VERBOSE_LOG         n\n")
    f.write("OBS_SED_FILE        y\n")
    f.write("TEMP_SED_FILE       y\n")
    f.write("POFZ_FILE           y\n")
    f.write("BINARY_OUTPUT       n\n\n")

    # Redshift / Magnitude prior
    f.write("## Redshift / Mag prior\n")
    f.write("APPLY_PRIOR         n\n")
    f.write("PRIOR_FILE          "+dir_temp+"prior_K_extend.dat\n")
    f.write("PRIOR_FILTER        28\n")
    f.write("PRIOR_ABZP          23.90\n\n")

    # Redshift grid
    f.write("## Redshift Grid\n")
    f.write("FIX_ZSPEC           n\n")
    f.write(f"Z_MIN               {zmin:.2f}\n")
    f.write(f"Z_MAX               {zmax:.2f}\n")
    f.write(f"Z_STEP              {zstep:.3f}\n")
    f.write("Z_STEP_TYPE         1\n\n")

    # Zeropoint offsets
    f.write("## Zeropoint Offsets\n")
    f.write("GET_ZP_OFFSETS      n\n")
    f.write("ZP_OFFSET_TOL       1.e-4\n\n")

    # Rest-frame colors
    f.write("## Rest-frame Colors\n")
    f.write("REST_FILTERS        140,142,161\n")    # UVJ filters
    f.write("RF_PADDING          1000.\n")
    f.write("RF_ERRORS           n\n")
    f.write("Z_COLUMN            z_peak\n")
    f.write("USE_ZSPEC_FOR_REST  y\n")
    f.write("READ_ZBIN           n\n\n")

    # Cosmology
    f.write("## Cosmology\n")
    f.write(f"H0                  {H0:.1f}\n")
    f.write(f"OMEGA_M             {Omega_M:.1f}\n")
    f.write(f"OMEGA_L             {Omega_L:.1f}\n")

    f.close()



ezParam = {'filters':dir_flt+"FILTER.RES.latest", 'temp_file':temp_file,
           'wave_file':dir_temp+"uvista_nmf/lambda.def", 'temperr_file':dir_temp+"uvista_nmf/template_error_10.def",
           'laf_file':dir_temp+"LAFcoeff.txt", 'dla_file':dir_temp+"DLAcoeff.txt",
           'zmin':0.1, 'zmax':12.0, 'zstep':0.005, 'H0':68.4, 'Omega_M':0.3, 'Omega_L':0.7}

param_file = ["zphot_hst.param", "zphot_jwst.param", "zphot_total.param"]
catalog = [dir_eazy_input+"flux_EAZY_hst.cat", dir_eazy_input+"flux_EAZY_jwst.cat", dir_eazy_input+"flux_EAZY_total.cat"]
output_file = ["photz_hst", "photz_jwst", "photz_total"]

g = open("./eazy.log", "w")
for i in range(len(param_file)):
    eazy_input(param_file[i], catalog[i], dir_fast_input, output_file[i], **ezParam)
    os.system("cp -rpv "+param_file[i]+" zphot.param")
    os.system(dir_src+"eazy zphot.param")
    g.write("eazy "+param_file[i]+"\n")
g.close()

# os.system("source ./eazy.sh")

# Printing the running time
print(f"--- {time.time()-start_time:.4f} sec ---")
