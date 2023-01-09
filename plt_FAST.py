#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:43:02 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import pandas as pd
import glob, os, copy
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import tqdm

from astropy.cosmology import FlatLambdaCDM


# ----- Basic settings ----- #
c = 2.99792e+10    # cm/s

# Directories
dir_eazy = "/home/jlee/Downloads/eazy-photoz/"
dir_fast = "/home/jlee/Downloads/FAST_v1.0/"
dir_src = dir_eazy+"src/"
dir_flt = dir_eazy+"filters/"

dir_FAST_input = "FAST_INPUT/"
dir_FAST_output = "FAST_OUTPUT/"
dir_FAST_bestfit = dir_FAST_output+"BEST_FITS/"

f = open(dir_flt+"FILTER.RES.latest","r")
flt_res_lines = f.readlines()
f.close()

g = open(dir_flt+"FILTER.RES.latest.info","r")
flt_info_lines = g.readlines()
g.close()

# Filter names
# galex = ["galex_FUV", "galex_NUV"]
# hst_acs = ["acs_wfc_f435w", "acs_wfc_f606w", "acs_wfc_f814w"]
# hst_wfc3_ir = ["wfc3_ir_f110w", "wfc3_ir_f140w"]
# spitzer = ["spitzer_irac_ch1", "spitzer_irac_ch2"]

# filternames = galex + hst_acs + hst_wfc3_ir + spitzer
# filternum = [120, 121, 233, 236, 239, 241, 204, 18, 19]    # EAZY & FAST

id_flt_hst = [233, 236, 239, 202, 203, 204, 205]    # from 'FILTER.RES.latest.info' file
id_flt_jwst = [363, 365, 366, 375, 376, 377]    # from 'FILTER.RES.latest.info' file\
filternum = id_flt_hst + id_flt_jwst    # EAZY & FAST


# Calculating effective wavelengths
lambda_c = np.array([])
for i in np.arange(len(filternum)):
    line_split = flt_info_lines[filternum[i]-1].split(" ")
    fltname_FAST = np.array(line_split)[np.array(line_split) != ''][2]
    # print(fltname_FAST)
    fidx = flt_res_lines.index([w for w in flt_res_lines if fltname_FAST in w][0])
    # print(fidx)
    line_split = flt_res_lines[fidx].split(' ')
    Nres = int(np.array(line_split)[np.array(line_split) != ''][0])
    # print(Nres)
    wav, res = np.array([]), np.array([])
    for j in flt_res_lines[fidx+1:fidx+1+Nres]:
        wav = np.append(wav, float(j.split(" ")[-2]))
        res = np.append(res, float(j.split(" ")[-1].split("\n")[0]))
    lambda_c = np.append(lambda_c, np.sum(wav*res)/np.sum(res))
lambda_c = lambda_c * 1.0e-4    # micrometer


# ----- Reading input catalog ----- #
def read_incat(input_catalog, num_flt_arr):
    f = open(input_catalog, "r")
    ll = f.readline()
    f.close()
    colnames = ll.split(' ')[1:-1]

    dc = np.genfromtxt(input_catalog, dtype=None, encoding="ascii",
                       comments="#", names=tuple(colnames))
    
    flx_cat, e_flx_cat = [], []    # microJy
    for i in np.arange(len(num_flt_arr)):
        flx_cat.append(dc[f"F{num_flt_arr[i]:d}"]*1)
        e_flx_cat.append(dc[f"E{num_flt_arr[i]:d}"]*1)

    return [dc['id'], np.array(flx_cat), np.array(e_flx_cat)] 

fluxes, e_fluxes = {}, {}
objid, fluxes['hst'], e_fluxes['hst'] = read_incat(dir_FAST_input+"photz_hst.cat", id_flt_hst)
_, fluxes['jwst'], e_fluxes['jwst'] = read_incat(dir_FAST_input+"photz_jwst.cat", id_flt_jwst)
_, fluxes['total'], e_fluxes['total'] = read_incat(dir_FAST_input+"photz_total.cat", filternum)


# ----- Reading output catalogs ----- #
def read_outcat(fout_file, bestfit_files, objids):
    dic = {}
    for i in range(len(bestfit_files)):
        dic[str(objids[i])] = {}
        if (glob.glob(bestfit_files[i]) == []):
            flx_fit = np.nan
            wav_fit = np.nan
        else:
            ds = np.genfromtxt(bestfit_files[i], dtype=None, encoding="ascii", comments="#", names=("wl","fl"))
            flx_fit = ds['fl'] * ((ds['wl'])**2./c) * 100.0   # microJy 
            wav_fit = ds['wl'] * 1.0e-4    # micrometer

        obj_key = f"{objids[i]:05d}"
        dic[obj_key] = {}
        dic[obj_key]['flux'] = flx_fit
        dic[obj_key]['wave'] = wav_fit

    df = np.genfromtxt(fout_file, dtype=None, comments="#",
                       names=("id","z","ltau","metal","lage","Av","lmass",
                              "lsfr","lssfr","la2t","chi2"))

    return [dic, df]

dict_fit = {}
dict_fit['hst_fit'], dict_fit['hst_fout'] = read_outcat(dir_FAST_output+"FAST_RESULT_hst.fout",
                                                        [dir_FAST_bestfit+"photz_hst_"+str(i)+".fit" for i in objid],
                                                        objid)
dict_fit['jwst_fit'], dict_fit['jwst_fout'] = read_outcat(dir_FAST_output+"FAST_RESULT_jwst.fout",
                                                          [dir_FAST_bestfit+"photz_jwst_"+str(i)+".fit" for i in objid],
                                                          objid)
dict_fit['total_fit'], dict_fit['total_fout'] = read_outcat(dir_FAST_output+"FAST_RESULT_total.fout",
                                                            [dir_FAST_bestfit+"photz_total_"+str(i)+".fit" for i in objid],
                                                            objid)


# ----- Figure setting ----- #
def draw_sed(objids, out, lambda_c, wav_fit, flx_fit, flx_cat, e_flx_cat, df_output, suffix_label):

    # Establish bounds
    xmin, xmax = np.min(lambda_c)*0.3, np.max(lambda_c)/0.3
    idx_xmin, idx_xmax = np.abs(wav_fit-xmin).argmin()-1, np.abs(wav_fit-xmax).argmin()+2
    nu_fit = 1.0e+4*c/wav_fit    # Hz
    fphot = 1.0e+4*c/lambda_c    # Hz
    specE = nu_fit*flx_fit*1.0e-29
    # print(specE)
    ymin = specE[idx_xmin:idx_xmax][specE[idx_xmin:idx_xmax] > 0.].min()*0.2
    ymax = specE[idx_xmin:idx_xmax][specE[idx_xmin:idx_xmax] > 0.].max()/0.2

    # --------------- #
    fig = plt.figure(1, figsize=(9,5))
    gs = GridSpec(1, 1, left=0.13, bottom=0.15, right=0.78, top=0.95)
    ax1 = fig.add_subplot(gs[0,0])
    ax = ax1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=15.0, pad=8.0)
    ax.set_xticks([1.0e-1, 5.0e-1, 1.0e+0, 5.0e+0, 1.0e+1])
    ax.set_xticklabels(["0.1", "0.5", "1", "5", "10"])
    ax.set_xlabel(r"Wavelength ${\rm [\mu m]}$", fontsize=15.0, labelpad=7.0)
    ax.set_ylabel(r"$\nu F_{\nu}~{\rm [erg~s^{-1}~cm^{-2}]}$", fontsize=15.0, labelpad=7.0)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.tick_params(width=1.5, length=9.0)
    ax.tick_params(width=1.5, length=5.0, which="minor")
    for axis in ["top","bottom","left","right"]:
        ax.spines[axis].set_linewidth(1.5)
    # --------------- #

    # Plotting model + data
    ax.plot(wav_fit, nu_fit*flx_fit*1.0e-29,
            label="Model spectrum",
            lw=1.5, color="navy", alpha=0.5)
    ax.errorbar(lambda_c, fphot*flx_cat*1.0e-29,
                yerr=fphot*e_flx_cat*1.0e-29,
                label="Observed photometry",
                marker="o", markersize=8, alpha=0.9, ls="", lw=2,
                ecolor="tomato", markerfacecolor="none", markeredgecolor="tomato", 
                markeredgewidth=2)

    # Figure texts
    ax.text(1.05, 0.95, f"ID-{objids:05d}", fontsize=16.0, fontweight="bold", color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.05, 0.85, suffix_label, fontsize=14.0, fontweight="bold", color="gray",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.70, r"$z_{\rm phot}=%.4f$" \
            %(df_output['z'][df_output['id'] == objids]), fontsize=12.0, color="blue",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.62, r"${\rm log} (M_{{\rm star}}/M_{\odot})=%.3f$" \
            %(df_output['lmass'][df_output['id'] == objids]), fontsize=12.0, color="blue",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.54, r"$[Z/Z_{\odot}]=%.2f$ (fixed)" \
            %(np.log10(df_output['metal'][df_output['id'] == objids]/0.02)), fontsize=12.0, color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.46, r"$A_{V}=%.1f$ mag" \
            %(df_output['Av'][df_output['id'] == objids]), fontsize=12.0, color="blue",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.38, r"$t_{\rm age}=%.2f$ Gyr" \
            %(10.0**(df_output['lage'][df_output['id'] == objids]-9.0)), fontsize=12.0, color="blue",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.30, r"$\tau=%.2f$ Gyr" \
            %(10.0**(df_output['ltau'][df_output['id'] == objids]-9.0)), fontsize=12.0, color="blue",
            ha="left", va="top", transform=ax.transAxes)
    # chisq = np.sum(((obs["maggies"]-pphot)/obs["maggies_unc"])**2.0)
    # dof = np.sum(obs['phot_mask']) - len(model.theta)
    # rchisq = chisq / dof
    ax.text(1.02, 0.10, r"$\chi^{2}=%.3f$" \
            %(df_output['chi2'][df_output['id'] == objids]), fontsize=12.0, color="red",
            ha="left", va="top", transform=ax.transAxes)
    # ax.text(1.02, 0.25, r"$\chi_{v}^{2}=%.3f$" \
    #         %(rchisq), fontsize=20.0, color="red",
    #         ha="left", va="top", transform=ax.transAxes)

    ax.legend(loc="best", fontsize=12)

    plt.savefig(out, dpi=300)
    # plt.savefig("./fig_FAST.png", dpi=300)
    plt.close()


dir_fig = dir_FAST_output+"Figures/"
if (glob.glob(dir_fig) == []):
    os.system("mkdir "+dir_fig)
else:
    os.system("rm -rfv "+dir_fig+"*")

# draw_sed(objids, out, lambda_c, wav_fit, flx_fit, flx_cat, e_flx_cat, df_output)

# HST only
for i in tqdm.trange(len(objid)):
    obj_key = f"{objid[i]:05d}"
    if (np.isnan(dict_fit['hst_fit'][obj_key]['wave']).any()):
        continue
    else:
        out = dir_fig+"Fig-SED"+obj_key+"_hst.png"
        draw_sed(objid[i], out, lambda_c[:7], dict_fit['hst_fit'][obj_key]['wave'], dict_fit['hst_fit'][obj_key]['flux'],
                 fluxes['hst'][:, i], e_fluxes['hst'][:, i], dict_fit['hst_fout'], suffix_label="(HST only)")

# # JWST only
for i in tqdm.trange(len(objid)):
    obj_key = f"{objid[i]:05d}"
    if (np.isnan(dict_fit['jwst_fit'][obj_key]['wave']).any()):
        continue
    else:
        out = dir_fig+"Fig-SED"+obj_key+"_jwst.png"
        draw_sed(objid[i], out, lambda_c[7:], dict_fit['jwst_fit'][obj_key]['wave'], dict_fit['jwst_fit'][obj_key]['flux'],
                 fluxes['jwst'][:, i], e_fluxes['jwst'][:, i], dict_fit['jwst_fout'], suffix_label="(JWST only)")

# # HST + JWST
for i in tqdm.trange(len(objid)):
    obj_key = f"{objid[i]:05d}"
    if (np.isnan(dict_fit['total_fit'][obj_key]['wave']).any()):
        continue
    else:
        out = dir_fig+"Fig-SED"+obj_key+"_total.png"
        draw_sed(objid[i], out, lambda_c, dict_fit['total_fit'][obj_key]['wave'], dict_fit['total_fit'][obj_key]['flux'],
                 fluxes['total'][:, i], e_fluxes['total'][:, i], dict_fit['total_fout'], suffix_label="(HST + JWST)")


# Printing the running time
print(f"--- {time.time()-start_time:.4f} sec ---")
