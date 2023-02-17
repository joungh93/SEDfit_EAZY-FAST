#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 23:37:43 2022

@author: jlee
"""

import numpy as np
import glob, os
import pandas as pd
import pickle
from pystilts import wcs_match1


# ----- Loading the photometry data ----- #
dir_phot = "/data01/jhlee/DATA/JWST/A2744/Weaver+23/Phot/"
dir_eazy_input = "EAZY_INPUT/"
if (glob.glob(dir_eazy_input) == []):
    os.system("mkdir "+dir_eazy_input)
else:
    os.system("rm -rfv "+dir_eazy_input+"*")

### Load the photometric data
with open(dir_phot+"phot_data.pickle", 'rb') as fr:
    phot_data = pickle.load(fr)


# ----- HST/JWST Bandpass ----- #
bands_hst = ['f435w', 'f606w', 'f814w',
             'f105w', 'f125w', 'f140w', 'f160w']
bands_jwst = ['f115w', 'f150w', 'f200w',
              'f277w', 'f356w', 'f410m', 'f444w']
n_hst = len(bands_hst)
n_jwst = len(bands_jwst)

### Selecting sources (extended + point)
mag_cnd = np.ones_like(phot_data['f200w']['num'].values, dtype=bool)
merr_cnd = np.ones_like(phot_data['f200w']['num'].values, dtype=bool)
size_cnd = np.ones_like(phot_data['f200w']['num'].values, dtype=bool)
for i in range(len(bands_jwst)):
    mag_cnd = (mag_cnd & (phot_data[bands_jwst[i]]['mag_iso'] < 30.0))
    merr_cnd = (merr_cnd & (phot_data[bands_jwst[i]]['e_mag_iso'] < 1.0))
    size_cnd = (size_cnd & (phot_data[bands_jwst[i]]['flxrad'] > 0.0))
eff_cnd = (mag_cnd & merr_cnd & size_cnd)
print(np.sum(eff_cnd))

gal_cnd = (mag_cnd & merr_cnd & size_cnd & \
           (phot_data['f200w']['flag'] <= 4) & \
           (phot_data['f200w']['flxrad'] >= 2.0))  #(phot_data['f200w']['cl'] < 0.4))
print(np.sum(gal_cnd))

# poi_cnd = (mag_cnd & merr_cnd & size_cnd & \
#            (phot_data['f200w']['flag'] <= 4) & (phot_data['f200w']['cl'] > 0.8))
# print(np.sum(poi_cnd))

### Filter information & extinction
obj_cnd = gal_cnd  #(gal_cnd | poi_cnd)
n_obj = np.sum(obj_cnd)

id_flt_hst = [233, 236, 239, 202, 203, 204, 205]    # from 'FILTER.RES.latest.info' file
id_flt_jwst = [364, 365, 366, 375, 376, 383, 377]    # from 'FILTER.RES.latest.info' file

Amags_hst = [0.053, 0.035, 0.022,
             0.014, 0.010, 0.009, 0.007]
Amags_jwst = [0.012, 0.008, 0.005,
              0.003, 0.000, 0.000, 0.000]

# idx_ref = 2
mag_offset = 0.0
# mag_offset = np.maximum(0.0, phot_data[bands_jwst[idx_ref]].iloc[gids-1]['mag_auto_1/2'] - \
#                              phot_data[bands_jwst[idx_ref]].iloc[gids-1]['mag_auto'])
e_mag_offset = 0.0
# e_mag_offset = np.sqrt(phot_data[bands_jwst[idx_ref]].iloc[gids-1]['e_mag_auto_1/2']**2 + \
#                        phot_data[bands_jwst[idx_ref]].iloc[gids-1]['e_mag_auto']**2)


# ----- Flux calculations ----- #
mag_AB_hst, e_mag_AB_hst = np.zeros((n_obj, n_hst)), np.zeros((n_obj, n_hst))
for i in range(n_hst):
    mag_AB_hst[:, i] = phot_data[bands_hst[i]].loc[obj_cnd]['mag_auto']  - Amags_hst[i]
    # mag_AB_hst[:, i] = phot_data[bands_hst[i]].iloc[gids-1]['mag_auto']  - Amags_hst[i]
    # mag_AB_hst[:, i] = phot_data[bands_hst[i]].iloc[gids-1]['mag_auto_1/2'] - mag_offset - Amags_hst[i]
    e_mag_AB_hst[:, i] = np.maximum(0.1, phot_data[bands_hst[i]].loc[obj_cnd]['e_mag_auto'])
    # e_mag_AB_hst[:, i] = np.maximum(0.1, phot_data[bands_hst[i]].iloc[gids-1]['e_mag_auto'])
    # e_mag_AB_hst[:, i] = np.maximum(0.1, np.sqrt(phot_data[bands_hst[i]].iloc[gids-1]['e_mag_auto_1/2']**2 + \
    #                                      e_mag_offset**2))

mag_AB_jwst, e_mag_AB_jwst = np.zeros((n_obj, n_jwst)), np.zeros((n_obj, n_jwst))
for i in range(n_jwst):
    mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].loc[obj_cnd]['mag_auto']  - Amags_jwst[i]
    # mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].iloc[gids-1]['mag_auto']  - Amags_jwst[i]
    # mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].iloc[gids-1]['mag_auto_1/2'] - mag_offset - Amags_jwst[i]
    e_mag_AB_jwst[:, i] = np.maximum(0.1, phot_data[bands_jwst[i]].loc[obj_cnd]['e_mag_auto'])
    # e_mag_AB_jwst[:, i] = np.maximum(0.1, phot_data[bands_jwst[i]].iloc[gids-1]['e_mag_auto'])
    # if (i == idx_ref):
    #     e_mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].iloc[gids-1]['e_mag_auto_1/2']
    # else:
    #     e_mag_AB_jwst[:, i] = np.maximum(0.1, np.sqrt(phot_data[bands_jwst[i]].iloc[gids-1]['e_mag_auto_1/2']**2 + \
    #                                           e_mag_offset**2))

### Flux to micro-Jansky
magzero = 23.90

Fv_hst = 10.0 ** ((magzero-mag_AB_hst)/2.5)   # micro-Jansky
e_Fv_hst = Fv_hst * (np.log(10.0)/2.5) * e_mag_AB_hst
e_Fv_hst[np.isnan(e_Fv_hst)] = Fv_hst[np.isnan(e_Fv_hst)] / 10.
Fv_hst[(Fv_hst < 1.0e-10) | (e_Fv_hst < 1.0e-10)] = np.nan
e_Fv_hst[(Fv_hst < 1.0e-10) | (e_Fv_hst < 1.0e-10)] = np.nan
Fv_hst[(np.isnan(Fv_hst)) | (np.isnan(e_Fv_hst))] = -99.
e_Fv_hst[(np.isnan(Fv_hst)) | (np.isnan(e_Fv_hst))] = -99.

Fv_jwst = 10.0 ** ((magzero-mag_AB_jwst)/2.5)   # micro Jansky
e_Fv_jwst = Fv_jwst * (np.log(10.0)/2.5) * e_mag_AB_jwst
e_Fv_jwst[np.isnan(e_Fv_jwst)] = Fv_jwst[np.isnan(e_Fv_jwst)] / 10.
Fv_jwst[(Fv_jwst < 1.0e-10) | (e_Fv_jwst < 1.0e-10)] = np.nan
e_Fv_jwst[(Fv_jwst < 1.0e-10) | (e_Fv_jwst < 1.0e-10)] = np.nan
Fv_jwst[(np.isnan(Fv_jwst)) | (np.isnan(e_Fv_jwst))] = -99.
e_Fv_jwst[(np.isnan(Fv_jwst)) | (np.isnan(e_Fv_jwst))] = -99.


# ----- Reading the spectroscopic catalogs ----- #
from astropy.io import fits
dir_w23 = "/data01/jhlee/DATA/JWST/A2744/Weaver+23/"
df_w23 = fits.getdata(dir_w23+"UNCOVER_DR1_LW_D070_catalog.fits")
df_w23 = pd.DataFrame(df_w23)
# df_w23.head(6)

z_spec = df_w23['z_spec'].values
n_obj2 = len(df_w23)

mag2_AB_hst, e_mag2_AB_hst = np.zeros((n_obj2, n_hst)), np.zeros((n_obj2, n_hst))
for i in range(n_hst):
    mag2_AB_hst[:, i] = 28.9 - 2.5*np.log10(df_w23['f_'+bands_hst[i]].values)
    e_mag2_AB_hst[:, i] = np.maximum(0.1, (2.5/np.log(10.0)) * \
                                    (df_w23['e_'+bands_hst[i]].values/df_w23['f_'+bands_hst[i]].values))
  
mag2_AB_jwst, e_mag2_AB_jwst = np.zeros((n_obj2, n_jwst)), np.zeros((n_obj2, n_jwst))
for i in range(len(bands_jwst)):
    mag2_AB_jwst[:, i] = 28.9 - 2.5*np.log10(df_w23['f_'+bands_jwst[i]].values)
    e_mag2_AB_jwst[:, i] = np.maximum(0.1, (2.5/np.log(10.0)) * \
                                      (df_w23['e_'+bands_jwst[i]].values/df_w23['f_'+bands_jwst[i]].values))

Fv2_hst = 10.0 ** ((magzero-mag2_AB_hst)/2.5)   # micro-Jansky
e_Fv2_hst = Fv2_hst * (np.log(10.0)/2.5) * e_mag2_AB_hst
e_Fv2_hst[np.isnan(e_Fv2_hst)] = Fv2_hst[np.isnan(e_Fv2_hst)] / 10.
Fv2_hst[(Fv2_hst < 1.0e-10) | (e_Fv2_hst < 1.0e-10)] = np.nan
e_Fv2_hst[(Fv2_hst < 1.0e-10) | (e_Fv2_hst < 1.0e-10)] = np.nan
Fv2_hst[(np.isnan(Fv2_hst)) | (np.isnan(e_Fv2_hst))] = -99.
e_Fv2_hst[(np.isnan(Fv2_hst)) | (np.isnan(e_Fv2_hst))] = -99.

Fv2_jwst = 10.0 ** ((magzero-mag2_AB_jwst)/2.5)   # micro Jansky
e_Fv2_jwst = Fv2_jwst * (np.log(10.0)/2.5) * e_mag2_AB_jwst
e_Fv2_jwst[np.isnan(e_Fv2_jwst)] = Fv2_jwst[np.isnan(e_Fv2_jwst)] / 10.
Fv2_jwst[(Fv2_jwst < 1.0e-10) | (e_Fv2_jwst < 1.0e-10)] = np.nan
e_Fv2_jwst[(Fv2_jwst < 1.0e-10) | (e_Fv2_jwst < 1.0e-10)] = np.nan
Fv2_jwst[(np.isnan(Fv2_jwst)) | (np.isnan(e_Fv2_jwst))] = -99.
e_Fv2_jwst[(np.isnan(Fv2_jwst)) | (np.isnan(e_Fv2_jwst))] = -99.


### Matching
tol = 0.5   # arcsec
idx_matched, idx_spec, sepr = wcs_match1(phot_data['f200w'].loc[obj_cnd]['ra'].values,
                                         phot_data['f200w'].loc[obj_cnd]['dec'].values,
                                         df_w23['ra'].values, df_w23['dec'].values, tol, ".")
print(len(idx_matched))


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

z_spec2 = -1.0 * np.ones_like(phot_data['f200w'].loc[obj_cnd]['num'].values)
z_spec2[idx_matched] = z_spec[idx_spec]
print(np.sum(z_spec2 > 0.))
# np.where(np.isnan(phot_data['Z_SPEC']), phot_data['Z_GRISM'], phot_data['Z_SPEC'])

# # HST only
# write_input_file(dir_eazy_input+"flux_EAZY_hst.cat", phot_data['f200w'].loc[obj_cnd]['num'].values,
#                  z_spec2, Fv_hst, e_Fv_hst, id_flt_hst)

# # JWST only
# write_input_file(dir_eazy_input+"flux_EAZY_jwst.cat", phot_data['f200w'].loc[obj_cnd]['num'].values,
#                  z_spec2, Fv_jwst, e_Fv_jwst, id_flt_jwst)

# HST+JWST - RUN #1
write_input_file(dir_eazy_input+"flux_EAZY_total_run1.cat",
                 phot_data['f200w'].loc[obj_cnd]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 np.column_stack([Fv_hst[z_spec2 > 0.], Fv_jwst[z_spec2 > 0.]]),
                 np.column_stack([e_Fv_hst[z_spec2 > 0.], e_Fv_jwst[z_spec2 > 0.]]),
                 id_flt_hst + id_flt_jwst)

# HST+JWST - RUN #2
write_input_file(dir_eazy_input+"flux_EAZY_total_run2.cat",
                 phot_data['f200w'].loc[obj_cnd]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 np.column_stack([Fv_hst[z_spec2 > 0.], Fv_jwst[z_spec2 > 0.]]),
                 np.column_stack([e_Fv_hst[z_spec2 > 0.], e_Fv_jwst[z_spec2 > 0.]]),
                 id_flt_hst + id_flt_jwst)

# HST+JWST - RUN #3
write_input_file(dir_eazy_input+"flux_EAZY_total_run3.cat",
                 phot_data['f200w'].loc[obj_cnd]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 np.column_stack([Fv_hst[z_spec2 > 0.], Fv_jwst[z_spec2 > 0.]]),
                 np.column_stack([e_Fv_hst[z_spec2 > 0.], e_Fv_jwst[z_spec2 > 0.]]),
                 id_flt_hst + id_flt_jwst)

# HST+JWST - RUN #4
write_input_file(dir_eazy_input+"flux_EAZY_total_run4.cat",
                 phot_data['f200w'].loc[obj_cnd]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 np.column_stack([Fv_hst[z_spec2 > 0.], Fv_jwst[z_spec2 > 0.]]),
                 np.column_stack([e_Fv_hst[z_spec2 > 0.], e_Fv_jwst[z_spec2 > 0.]]),
                 id_flt_hst + id_flt_jwst)

# # HST+JWST - RUN #5
# write_input_file(dir_eazy_input+"flux_EAZY_total_run5.cat",
#                  phot_data['f200w'].loc[obj_cnd]['num'].values[z_spec2 > 0.],
#                  z_spec2[z_spec2 > 0.],
#                  np.column_stack([Fv_hst[z_spec2 > 0.], Fv_jwst[z_spec2 > 0.]]),
#                  np.column_stack([e_Fv_hst[z_spec2 > 0.], e_Fv_jwst[z_spec2 > 0.]]),
#                  id_flt_hst + id_flt_jwst)

# # JWST fluxes from Weaver+23
# write_input_file(dir_eazy_input+"flux_EAZY_jwst_w23.cat", phot_data['f200w'].loc[obj_cnd]['num'].values[idx_matched],
#                  z_spec2[idx_matched], Fv2_jwst[idx_spec], e_Fv2_jwst[idx_spec], id_flt_jwst)

# # # HST+JWST fluxes from Weaver+23
# write_input_file(dir_eazy_input+"flux_EAZY_total_w23.cat", phot_data['f200w'].loc[obj_cnd]['num'].values[idx_matched],
#                  z_spec2[idx_matched], np.column_stack([Fv2_hst[idx_spec], Fv2_jwst[idx_spec]]),
#                  np.column_stack([e_Fv2_hst[idx_spec], e_Fv2_jwst[idx_spec]]), id_flt_hst + id_flt_jwst)

# # RELICS-matched sources with RELICS fluxes
# from pystilts import wcs_match1

# dir_clu = "/data/jlee/DATA/JWST/First/SMACS0723/Ferreira+22/RELICS/cat/"
# catname = "hlsp_relics_hst_acs-wfc3ir_smacs0723-73_multi_v1_cat.txt"

# with open(dir_clu+catname, "r") as f:
#     ll = f.readlines()

# colnames = ll[128].split()[1:]
# df_relics = np.genfromtxt(dir_clu+catname, dtype=None, skip_header=129, names=colnames)
# # df_relics = pd.read_fwf(dir_clu+catname, comment="#", skiprows=129, header=None, names=colnames,
# #                         na_values='inf')

# tol = 0.5   # arcsec
# id_n22, id_relics, sepr = wcs_match1(phot_data['RA'].values, phot_data['DEC'].values,
#                                      df_relics['RA'], df_relics['Dec'], tol, ".")
# print(len(id_n22))

# band_relics = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
# id_flt_relics = [233, 236, 239, 202, 203, 204, 205]
# Fv_relics, e_Fv_relics = np.zeros((len(df_relics), len(band_relics))), np.zeros((len(df_relics), len(band_relics)))
# for i in range(len(band_relics)):
#     Fv_relics[:, i] = np.where(np.isnan(df_relics[band_relics[i]+'_fluxnJy']),
#                                np.nan, df_relics[band_relics[i]+'_fluxnJy'] * 1.0e-3)
#     e_Fv_relics[:, i] = np.where(np.isnan(df_relics[band_relics[i]+'_fluxnJyerr']),
#                                  np.nan, df_relics[band_relics[i]+'_fluxnJyerr'] * 1.0e-3)
# fault = ((Fv_relics < 0.) | (np.isnan(Fv_relics)) | (np.isinf(Fv_relics)) | \
#          (e_Fv_relics < 0.) | (np.isnan(e_Fv_relics)) | (np.isinf(e_Fv_relics)))
# Fv_relics[fault], e_Fv_relics[fault] = -99., -99.

# write_input_file(dir_eazy_input+"flux_EAZY_RELICS_hstacs.cat", phot_data['ID'].values[id_n22],
#                  z_spec[id_n22], Fv_relics[id_relics, :3], e_Fv_relics[id_relics, :3], id_flt_relics[:3])

# write_input_file(dir_eazy_input+"flux_EAZY_RELICS_hstnir.cat", phot_data['ID'].values[id_n22],
#                  z_spec[id_n22], Fv_relics[id_relics, 3:], e_Fv_relics[id_relics, 3:], id_flt_relics[3:])

# write_input_file(dir_eazy_input+"flux_EAZY_RELICS_total.cat", phot_data['ID'].values[id_n22],
#                  z_spec[id_n22], Fv_relics[id_relics, :], e_Fv_relics[id_relics, :], id_flt_relics)
