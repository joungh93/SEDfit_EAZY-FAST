#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 23:37:43 2022

@author: jlee
"""

import numpy as np
import glob, os
import pandas as pd


# ----- Loading the photometry data ----- #
dir_phot = "/data/jlee/DATA/JWST/First/SMACS0723/"
dir_eazy_input = "EAZY_INPUT/"
if (glob.glob(dir_eazy_input) == []):
    os.system("mkdir "+dir_eazy_input)
else:
    os.system("rm -rfv "+dir_eazy_input+"*")

# load data
phot_data = pd.read_csv(dir_phot+"Noirot+22_redshift_catalog.txt", skiprows=2, sep=' ')


# ----- HST/JWST Bandpass ----- #
bands_hst = ['f435w', 'f606w', 'f814w']
bands_jwst = ['f090w', 'f150w', 'f200w',
              'f277w', 'f356w', 'f444w']
n_hst = len(bands_hst)
n_jwst = len(bands_jwst)
n_obj = len(phot_data)

id_flt_hst = [233, 236, 239]    # from 'FILTER.RES.latest.info' file
id_flt_jwst = [363, 365, 366, 375, 376, 377]    # from 'FILTER.RES.latest.info' file

# Amags_hst = [0.787, 0.539, 0.333,
#              0.211, 0.158, 0.134, 0.112]
# Amags_jwst = [0.296, 0.134, 0.083,
#               0.053, 0.039, 0.031]
# idx_ref = 2
# mag_offset = 0.0
# # mag_offset = np.maximum(0.0, phot_data[bands_jwst[idx_ref]].iloc[gids-1]['mag_auto_1/2'] - \
# #                              phot_data[bands_jwst[idx_ref]].iloc[gids-1]['mag_auto'])
# e_mag_offset = 0.0
# # e_mag_offset = np.sqrt(phot_data[bands_jwst[idx_ref]].iloc[gids-1]['e_mag_auto_1/2']**2 + \
# #                        phot_data[bands_jwst[idx_ref]].iloc[gids-1]['e_mag_auto']**2)


# # ----- Flux calculations ----- #
# mag_AB_hst, e_mag_AB_hst = np.zeros((n_obj, n_hst)), np.zeros((n_obj, n_hst))
# for i in range(n_hst):
#     mag_AB_hst[:, i] = phot_data[bands_hst[i]].iloc[gids-1]['mag_iso']  - Amags_hst[i]
#     # mag_AB_hst[:, i] = phot_data[bands_hst[i]].iloc[gids-1]['mag_auto']  - Amags_hst[i]
#     # mag_AB_hst[:, i] = phot_data[bands_hst[i]].iloc[gids-1]['mag_auto_1/2'] - mag_offset - Amags_hst[i]
#     e_mag_AB_hst[:, i] = np.maximum(0.1, phot_data[bands_hst[i]].iloc[gids-1]['e_mag_iso'])
#     # e_mag_AB_hst[:, i] = np.maximum(0.1, phot_data[bands_hst[i]].iloc[gids-1]['e_mag_auto'])
#     # e_mag_AB_hst[:, i] = np.maximum(0.1, np.sqrt(phot_data[bands_hst[i]].iloc[gids-1]['e_mag_auto_1/2']**2 + \
#     #                                      e_mag_offset**2))

# mag_AB_jwst, e_mag_AB_jwst = np.zeros((n_obj, n_jwst)), np.zeros((n_obj, n_jwst))
# for i in range(n_jwst):
#     mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].iloc[gids-1]['mag_iso']  - Amags_jwst[i]
#     # mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].iloc[gids-1]['mag_auto']  - Amags_jwst[i]
#     # mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].iloc[gids-1]['mag_auto_1/2'] - mag_offset - Amags_jwst[i]
#     e_mag_AB_jwst[:, i] = np.maximum(0.1, phot_data[bands_jwst[i]].iloc[gids-1]['e_mag_iso'])
#     # e_mag_AB_jwst[:, i] = np.maximum(0.1, phot_data[bands_jwst[i]].iloc[gids-1]['e_mag_auto'])
#     # if (i == idx_ref):
#     #     e_mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].iloc[gids-1]['e_mag_auto_1/2']
#     # else:
#     #     e_mag_AB_jwst[:, i] = np.maximum(0.1, np.sqrt(phot_data[bands_jwst[i]].iloc[gids-1]['e_mag_auto_1/2']**2 + \
#     #                                           e_mag_offset**2))

# magzero = 23.90

# Fv_hst = 10.0 ** ((magzero-mag_AB_hst)/2.5)   # not micro Jansky
# e_Fv_hst = Fv_hst * (np.log(10.0)/2.5) * e_mag_AB_hst
# e_Fv_hst[np.isnan(e_Fv_hst)] = Fv_hst[np.isnan(e_Fv_hst)] / 10.
# Fv_hst[(Fv_hst < 1.0e-10) | (e_Fv_hst < 1.0e-10)] = np.nan
# e_Fv_hst[(Fv_hst < 1.0e-10) | (e_Fv_hst < 1.0e-10)] = np.nan

Fv_hst, e_Fv_hst = np.zeros((n_obj, n_hst)), np.zeros((n_obj, n_hst))
for i in range(n_hst):
    Fv_hst[:, i] = phot_data['FLUX_'+bands_hst[i].upper()] * 1.0e-3
    e_Fv_hst[:, i] = phot_data['FLUXERR_'+bands_hst[i].upper()] * 1.0e-3
Fv_hst[(np.isnan(Fv_hst)) | (np.isnan(e_Fv_hst))] = -99.
e_Fv_hst[(np.isnan(Fv_hst)) | (np.isnan(e_Fv_hst))] = -99.

# Fv_jwst = 10.0 ** ((magzero-mag_AB_jwst)/2.5)   # not micro Jansky
# e_Fv_jwst = Fv_jwst * (np.log(10.0)/2.5) * e_mag_AB_jwst
# e_Fv_jwst[np.isnan(e_Fv_jwst)] = Fv_jwst[np.isnan(e_Fv_jwst)] / 10.
# Fv_jwst[(Fv_jwst < 1.0e-10) | (e_Fv_jwst < 1.0e-10)] = np.nan
# e_Fv_jwst[(Fv_jwst < 1.0e-10) | (e_Fv_jwst < 1.0e-10)] = np.nan

Fv_jwst, e_Fv_jwst = np.zeros((n_obj, n_jwst)), np.zeros((n_obj, n_jwst))
for i in range(n_jwst):
    Fv_jwst[:, i] = phot_data['FLUX_'+bands_jwst[i].upper()] * 1.0e-3
    e_Fv_jwst[:, i] = phot_data['FLUXERR_'+bands_jwst[i].upper()] * 1.0e-3
Fv_jwst[(np.isnan(Fv_jwst)) | (np.isnan(e_Fv_jwst))] = -99.
e_Fv_jwst[(np.isnan(Fv_jwst)) | (np.isnan(e_Fv_jwst))] = -99.


# ----- Writing input files ----- #
def write_input_file(filename, objid, z_spec, flux, err_flux, id_filter):
    f = open(filename, "w")
    columns = "# id z_spec "
    for i in range(len(id_filter)):
        columns += f"F{id_filter[i]:d} E{id_filter[i]:d} "
    f.write(columns+"\n")
    for j in range(len(objid)):
        txt = f"{objid[j]:d} {z_spec[j]:.4f} "
        for i in range(len(id_filter)):
            txt += f"{flux[j, i]:5.3e} {err_flux[j, i]:5.3e} "
        f.write(txt+"\n")
    f.close()

z_spec = np.where(np.isnan(phot_data['Z_SPEC']), phot_data['Z_GRISM'], phot_data['Z_SPEC'])

# HST only
write_input_file(dir_eazy_input+"flux_EAZY_hst.cat", phot_data['ID'],
                 z_spec, Fv_hst, e_Fv_hst, id_flt_hst)

# JWST only
write_input_file(dir_eazy_input+"flux_EAZY_jwst.cat", phot_data['ID'],
                 z_spec, Fv_jwst, e_Fv_jwst, id_flt_jwst)

# HST+JWST
write_input_file(dir_eazy_input+"flux_EAZY_total.cat", phot_data['ID'], z_spec,
                 np.column_stack([Fv_hst, Fv_jwst]), np.column_stack([e_Fv_hst, e_Fv_jwst]),
                 id_flt_hst + id_flt_jwst)

# RELICS-matched sources with RELICS fluxes
from pystilts import wcs_match1

dir_clu = "/data/jlee/DATA/JWST/First/SMACS0723/Ferreira+22/RELICS/cat/"
catname = "hlsp_relics_hst_acs-wfc3ir_smacs0723-73_multi_v1_cat.txt"

with open(dir_clu+catname, "r") as f:
    ll = f.readlines()

colnames = ll[128].split()[1:]
df_relics = np.genfromtxt(dir_clu+catname, dtype=None, skip_header=129, names=colnames)
# df_relics = pd.read_fwf(dir_clu+catname, comment="#", skiprows=129, header=None, names=colnames,
#                         na_values='inf')

tol = 0.5   # arcsec
id_n22, id_relics, sepr = wcs_match1(phot_data['RA'].values, phot_data['DEC'].values,
                                     df_relics['RA'], df_relics['Dec'], tol, ".")
print(len(id_n22))

band_relics = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
id_flt_relics = [233, 236, 239, 202, 203, 204, 205]
Fv_relics, e_Fv_relics = np.zeros((len(df_relics), len(band_relics))), np.zeros((len(df_relics), len(band_relics)))
for i in range(len(band_relics)):
    Fv_relics[:, i] = np.where(np.isnan(df_relics[band_relics[i]+'_fluxnJy']),
                               np.nan, df_relics[band_relics[i]+'_fluxnJy'] * 1.0e-3)
    e_Fv_relics[:, i] = np.where(np.isnan(df_relics[band_relics[i]+'_fluxnJyerr']),
                                 np.nan, df_relics[band_relics[i]+'_fluxnJyerr'] * 1.0e-3)
fault = ((Fv_relics < 0.) | (np.isnan(Fv_relics)) | (np.isinf(Fv_relics)) | \
         (e_Fv_relics < 0.) | (np.isnan(e_Fv_relics)) | (np.isinf(e_Fv_relics)))
Fv_relics[fault], e_Fv_relics[fault] = -99., -99.

write_input_file(dir_eazy_input+"flux_EAZY_RELICS_hstacs.cat", phot_data['ID'].values[id_n22],
                 z_spec[id_n22], Fv_relics[id_relics, :3], e_Fv_relics[id_relics, :3], id_flt_relics[:3])

write_input_file(dir_eazy_input+"flux_EAZY_RELICS_hstnir.cat", phot_data['ID'].values[id_n22],
                 z_spec[id_n22], Fv_relics[id_relics, 3:], e_Fv_relics[id_relics, 3:], id_flt_relics[3:])

write_input_file(dir_eazy_input+"flux_EAZY_RELICS_total.cat", phot_data['ID'].values[id_n22],
                 z_spec[id_n22], Fv_relics[id_relics, :], e_Fv_relics[id_relics, :], id_flt_relics)
