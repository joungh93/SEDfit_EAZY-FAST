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

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import extinction
from dustmaps.sfd import SFDQuery


# ----- Loading the photometry data ----- #
dir_phot = "/data01/jhlee/DATA/JWST/First/Valentino+23/Phot/"
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
bands_jwst = ['f090w', 'f150w', 'f200w',
              'f277w', 'f356w', 'f444w']
n_hst = len(bands_hst)
n_jwst = len(bands_jwst)

### Selecting sources (extended + point)
mag_cnd = np.ones_like(phot_data['f200w']['num'].values, dtype=bool)
merr_cnd = np.ones_like(phot_data['f200w']['num'].values, dtype=bool)
size_cnd = np.ones_like(phot_data['f200w']['num'].values, dtype=bool)
for band in bands_jwst[3:]:
    mag_cnd = (mag_cnd & (phot_data[band]['mag_aper'] < 30.0))
    merr_cnd = (merr_cnd & (phot_data[band]['e_mag_aper'] < 1.0))
    size_cnd = (size_cnd & (phot_data[band]['flxrad'] > 0.0))
eff_cnd = (mag_cnd & merr_cnd & size_cnd)
print(np.sum(eff_cnd))

col_cnd = ((phot_data['f200w']['mag_corr'] - phot_data['f277w']['mag_corr'] >= -1.0) & \
           (phot_data['f200w']['mag_corr'] - phot_data['f277w']['mag_corr'] <=  1.0) & \
           (phot_data['f277w']['mag_corr'] - phot_data['f356w']['mag_corr'] >= -1.0) & \
           (phot_data['f277w']['mag_corr'] - phot_data['f356w']['mag_corr'] <=  1.0))

# nir_flags = np.vstack([phot_data['f277w']['flag'].values,
#                        phot_data['f356w']['flag'].values,
#                        phot_data['f444w']['flag'].values])
gal_cnd = (mag_cnd & merr_cnd & size_cnd & col_cnd & \
           # (np.nanmedian(nir_flags, axis=0) <= 4) & \
           # (phot_data['f277w']['flag'] <= 4) & \
           (phot_data['nir_detect']['flag'] <= 4) & \
           (phot_data['nir_detect']['flxrad'] >= 2.25))    # (phot_data['f200w']['cl'] < 0.4)
print(np.sum(gal_cnd))

# poi_cnd = (mag_cnd & merr_cnd & size_cnd & \
#            (phot_data['f200w']['flag'] <= 4) & (phot_data['f200w']['cl'] > 0.8))
# print(np.sum(poi_cnd))


### Filter information & extinction
id_flt_hst = [233, 236, 239, 202, 203, 204, 205]    # from 'FILTER.RES.latest.info' file
id_flt_jwst = [363, 365, 366, 375, 376, 377]    # from 'FILTER.RES.latest.info' file

dir_img = "/data01/jhlee/DATA/JWST/First/Valentino+23/Reproject/"
hdr = fits.getheader(dir_img+"f277w.fits")
ra, dec = hdr['CRVAL1'], hdr['CRVAL2']

sfd = SFDQuery()
coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
ebv_sfd = sfd(coords)
R_V = 3.1
A_V = R_V * ebv_sfd
wave_hst = np.array([4318.9, 5921.1, 8057.0,
                     10545.0, 12471.0, 13924.0, 15396.0])
wave_jwst = np.array([9022.9, 15007.0, 19886.0,
                      27623.0, 35682.0, 44037.0])

Amags_hst = extinction.fitzpatrick99(wave_hst, A_V, R_V, unit='aa')
Amags_jwst = extinction.fitzpatrick99(wave_jwst, A_V, R_V, unit='aa')


# ----- Flux calculations ----- #
obj_cnd = gal_cnd  #(gal_cnd | poi_cnd)
n_obj = np.sum(obj_cnd)

# idx_ref = 2
mag_offset = 0.0
# mag_offset = np.maximum(0.0, phot_data[bands_jwst[idx_ref]].iloc[gids-1]['mag_auto_1/2'] - \
#                              phot_data[bands_jwst[idx_ref]].iloc[gids-1]['mag_corr'])
e_mag_offset = 0.0
# e_mag_offset = np.sqrt(phot_data[bands_jwst[idx_ref]].iloc[gids-1]['e_mag_auto_1/2']**2 + \
#                        phot_data[bands_jwst[idx_ref]].iloc[gids-1]['e_mag_corr']**2)

mag_AB_hst, e_mag_AB_hst = np.zeros((n_obj, n_hst)), np.zeros((n_obj, n_hst))
for i in range(n_hst):
    mag_AB_hst[:, i] = phot_data[bands_hst[i]].loc[obj_cnd]['mag_corr']  - Amags_hst[i]
    # mag_AB_hst[:, i] = phot_data[bands_hst[i]].iloc[gids-1]['mag_corr']  - Amags_hst[i]
    # mag_AB_hst[:, i] = phot_data[bands_hst[i]].iloc[gids-1]['mag_auto_1/2'] - mag_offset - Amags_hst[i]
    e_mag_AB_hst[:, i] = np.maximum(0.1, phot_data[bands_hst[i]].loc[obj_cnd]['e_mag_corr'])
    # e_mag_AB_hst[:, i] = np.maximum(0.1, phot_data[bands_hst[i]].iloc[gids-1]['e_mag_corr'])
    # e_mag_AB_hst[:, i] = np.maximum(0.1, np.sqrt(phot_data[bands_hst[i]].iloc[gids-1]['e_mag_auto_1/2']**2 + \
    #                                      e_mag_offset**2))

mag_AB_jwst, e_mag_AB_jwst = np.zeros((n_obj, n_jwst)), np.zeros((n_obj, n_jwst))
for i in range(n_jwst):
    mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].loc[obj_cnd]['mag_corr']  - Amags_jwst[i]
    # mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].iloc[gids-1]['mag_corr']  - Amags_jwst[i]
    # mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].iloc[gids-1]['mag_auto_1/2'] - mag_offset - Amags_jwst[i]
    e_mag_AB_jwst[:, i] = np.maximum(0.1, phot_data[bands_jwst[i]].loc[obj_cnd]['e_mag_corr'])
    # e_mag_AB_jwst[:, i] = np.maximum(0.1, phot_data[bands_jwst[i]].iloc[gids-1]['e_mag_corr'])
    # if (i == idx_ref):
    #     e_mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].iloc[gids-1]['e_mag_auto_1/2']
    # else:
    #     e_mag_AB_jwst[:, i] = np.maximum(0.1, np.sqrt(phot_data[bands_jwst[i]].iloc[gids-1]['e_mag_auto_1/2']**2 + \
    #                                           e_mag_offset**2))

### Flux to micro-Jansky
magzero = 23.90
Fv_min = 10.0 ** ((magzero-30.0)/2.5)

Fv_hst = 10.0 ** ((magzero-mag_AB_hst)/2.5)   # micro-Jansky
e_Fv_hst = Fv_hst * (np.log(10.0)/2.5) * e_mag_AB_hst
e_Fv_hst[np.isnan(e_Fv_hst)] = Fv_hst[np.isnan(e_Fv_hst)] / 10.
Fv_hst[(Fv_hst < 1.0e-10) | (e_Fv_hst < 1.0e-10) | (Fv_hst < Fv_min)] = np.nan
e_Fv_hst[(Fv_hst < 1.0e-10) | (e_Fv_hst < 1.0e-10) | (Fv_hst < Fv_min)] = np.nan
for i in range(Fv_hst.shape[0]):
    Fv_med = np.nanmedian(Fv_hst[i, :])
    if np.isnan(Fv_med):
        continue
    else:
        Fv_hst[i, :][Fv_hst[i, :] < Fv_med/100.] = np.nan
Fv_hst[(np.isnan(Fv_hst)) | (np.isnan(e_Fv_hst))] = -99.
e_Fv_hst[(np.isnan(Fv_hst)) | (np.isnan(e_Fv_hst))] = -99.

Fv_jwst = 10.0 ** ((magzero-mag_AB_jwst)/2.5)   # micro Jansky
e_Fv_jwst = Fv_jwst * (np.log(10.0)/2.5) * e_mag_AB_jwst
e_Fv_jwst[np.isnan(e_Fv_jwst)] = Fv_jwst[np.isnan(e_Fv_jwst)] / 10.
Fv_jwst[(Fv_jwst < 1.0e-10) | (e_Fv_jwst < 1.0e-10) | (Fv_jwst < Fv_min)] = np.nan
e_Fv_jwst[(Fv_jwst < 1.0e-10) | (e_Fv_jwst < 1.0e-10) | (Fv_jwst < Fv_min)] = np.nan
for i in range(Fv_jwst.shape[0]):
    Fv_med = np.nanmedian(Fv_jwst[i, :])
    if np.isnan(Fv_med):
        continue
    else:
        Fv_jwst[i, :][Fv_jwst[i, :] < Fv_med/100.] = np.nan
Fv_jwst[(np.isnan(Fv_jwst)) | (np.isnan(e_Fv_jwst))] = -99.
e_Fv_jwst[(np.isnan(Fv_jwst)) | (np.isnan(e_Fv_jwst))] = -99.


# ----- Reading the spectroscopic catalogs ----- #
dir_n22 = "/data01/jhlee/DATA/JWST/First/SMACS0723/"
df_n22 = pd.read_csv(dir_n22+"Noirot+22_redshift_catalog.txt", skiprows=2, sep=' ')
# df_n22.head(6)
z_spec = np.where(np.isnan(df_n22['Z_SPEC'].values), df_n22['Z_GRISM'].values, df_n22['Z_SPEC'].values)

### Matching
tol = 0.5   # arcsec
idx_matched, idx_spec, sepr = wcs_match1(phot_data['f200w'].loc[obj_cnd]['ra'].values,
                                         phot_data['f200w'].loc[obj_cnd]['dec'].values,
                                         df_n22['RA'].values, df_n22['DEC'].values, tol, ".")
print(len(idx_matched))


# ----- Write the region files ----- #
with open("n22_matched.reg", "w") as f:
    f.write('global color=magenta font="helvetica 10 normal" ')
    f.write("select=1 edit=1 move=1 delete=1 include=1 fixed=0 source width=2\n")
    f.write("fk5\n")
    for i in range(len(idx_matched)):
        f.write(f"circle({df_n22['RA'].values[idx_spec][i]:.6f}, {df_n22['DEC'].values[idx_spec][i]:.6f}, 1.0")
        f.write('")\n')  # text={'+f"{z_spec[idx_spec][i]:.4f}"+'}\n')


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
write_input_file(dir_eazy_input+"flux_EAZY_run1.cat",
                 phot_data['f200w'].loc[obj_cnd]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 np.column_stack([Fv_hst[z_spec2 > 0.], Fv_jwst[z_spec2 > 0.]]),
                 np.column_stack([e_Fv_hst[z_spec2 > 0.], e_Fv_jwst[z_spec2 > 0.]]),
                 id_flt_hst + id_flt_jwst)

# HST+JWST - RUN #2
write_input_file(dir_eazy_input+"flux_EAZY_run2.cat",
                 phot_data['f200w'].loc[obj_cnd]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 np.column_stack([Fv_hst[z_spec2 > 0.], Fv_jwst[z_spec2 > 0.]]),
                 np.column_stack([e_Fv_hst[z_spec2 > 0.], e_Fv_jwst[z_spec2 > 0.]]),
                 id_flt_hst + id_flt_jwst)

# HST+JWST - RUN #3
write_input_file(dir_eazy_input+"flux_EAZY_run3.cat",
                 phot_data['f200w'].loc[obj_cnd]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 np.column_stack([Fv_hst[z_spec2 > 0.], Fv_jwst[z_spec2 > 0.]]),
                 np.column_stack([e_Fv_hst[z_spec2 > 0.], e_Fv_jwst[z_spec2 > 0.]]),
                 id_flt_hst + id_flt_jwst)

# JWST - RUN #4
write_input_file(dir_eazy_input+"flux_EAZY_run4.cat",
                 phot_data['f200w'].loc[obj_cnd]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 Fv_jwst[z_spec2 > 0.],
                 e_Fv_jwst[z_spec2 > 0.],
                 id_flt_jwst)

# HST+JWST - RUN #5 (-F435W)
write_input_file(dir_eazy_input+"flux_EAZY_run5.cat",
                 phot_data['f200w'].loc[obj_cnd]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 np.column_stack([Fv_hst[z_spec2 > 0.][:, 1:], Fv_jwst[z_spec2 > 0.]]),
                 np.column_stack([e_Fv_hst[z_spec2 > 0.][:, 1:], e_Fv_jwst[z_spec2 > 0.]]),
                 id_flt_hst[1:] + id_flt_jwst)

# HST+JWST - RUN #6 (-F435W, Total)
write_input_file(dir_eazy_input+"flux_EAZY_run6.cat",
                 phot_data['f200w'].loc[obj_cnd]['num'].values, z_spec2,
                 np.column_stack([Fv_hst[:, 1:], Fv_jwst]),
                 np.column_stack([e_Fv_hst[:, 1:], e_Fv_jwst]),
                 id_flt_hst[1:] + id_flt_jwst)

