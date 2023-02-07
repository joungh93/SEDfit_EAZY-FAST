#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:31:54 2023

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.visualization import ZScaleInterval
interval = ZScaleInterval()
from astropy.io import fits
import pickle
from matplotlib.patches import Ellipse
import tqdm

c = 2.99792e+10    # cm/s

def read_zout(file):
    with open(file, "r") as f:
        ll = f.readlines()
    cols_zfile = ll[0].split()[1:]
    cols_zfile = tuple(cols_zfile)

    dz = np.genfromtxt(file, dtype=None, comments="#", names=cols_zfile)
    dz = pd.DataFrame(dz)
    return dz


def plot_eazy_sed(objid, prefix, out, z_phot=0.1, z_spec=-1.0, peak_prob=1.0,
                  dir_output="FAST_INPUT", dir_fig="EAZY_Figure",
                  n_inst=1, wav_eff=[], color_inst=[], label_inst=[],
                  cut_img=None, pixel_scale=None,
                  f200w_mag=None, r_h=None, color1=None, color2=None,
                  r_kron=None, axis_ratio=None, theta=None,
                  cut_img2=None):
    
    if (dir_output[-1] == "/"):
        dir_output = dir_output[-1]
    if (dir_fig[-1] == "/"):
        dir_fig = dir_fig

    if not os.path.exists(dir_fig):
        os.system("mkdir "+dir_fig)

    # dz = read_zout(dir_output+"/"+prefix+".zout")
    obs_sed = np.genfromtxt(dir_output+"/"+prefix+f"_{objid:d}.obs_sed", dtype=None, encoding='ascii',
                            names=('lambda','flux_cat','err_cat','err_full','tempa_z'))
    temp_sed = np.genfromtxt(dir_output+"/"+prefix+f"_{objid:d}.temp_sed", dtype=None, encoding='ascii',
                             names=('lambda','tempflux'))
    pz = np.genfromtxt(dir_output+"/"+prefix+f"_{objid:d}.pz", dtype=None, encoding='ascii',
                       names=('z','chi2'))

    lambda_c = obs_sed['lambda'] * 1.0e-4
    flx_cat = obs_sed['flux_cat']
    e_flx_cat = obs_sed['err_full']
    wav_fit = temp_sed['lambda'] * 1.0e-4
    flx_fit = temp_sed['tempflux']

    if (n_inst > 1):
        idx_inst = []
        for i in range(n_inst):
            idx_inst.append(np.in1d(obs_sed['lambda'].astype('int'), wav_eff[i]))

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
    fig = plt.figure(1, figsize=(9,8))
    gs = GridSpec(2, 1, left=0.15, bottom=0.10, right=0.78, top=0.95, height_ratios=[6,3], hspace=0.30)
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
    if (n_inst == 1):
        ax.errorbar(lambda_c, fphot*flx_cat*1.0e-29,
                    yerr=fphot*e_flx_cat*1.0e-29,
                    label=label_inst[0],
                    marker="o", markersize=8, alpha=0.9, ls="", lw=2,
                    ecolor=color_inst[0], markerfacecolor="none", markeredgecolor=color_inst[0], 
                    markeredgewidth=2)
    else:
        for i in range(n_inst):
            ax.errorbar(lambda_c[idx_inst[i]], fphot[idx_inst[i]]*flx_cat[idx_inst[i]]*1.0e-29,
                        yerr=fphot[idx_inst[i]]*e_flx_cat[idx_inst[i]]*1.0e-29,
                        label=label_inst[i],
                        marker="o", markersize=8, alpha=0.9, ls="", lw=2,
                        ecolor=color_inst[i], markerfacecolor="none", markeredgecolor=color_inst[i], 
                        markeredgewidth=2)        

    # Figure texts
    ax.text(1.05, 0.95, f"ID-{objid:05d}", fontsize=18.0, fontweight="bold", color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.03, 0.78, r"$z_{\rm spec}=$"+f"{z_spec:.4f}",
            fontsize=14.0, color='red', ha="left", va="top", transform=ax.transAxes)
    ax.text(1.03, 0.70, r"$z_{\rm phot}=$"+f"{z_phot:.4f}",
            fontsize=14.0, color='blue', ha="left", va="top", transform=ax.transAxes)
    ax.text(1.03, 0.62, r"$p(z)=$"+f"{peak_prob:.4f}",
            fontsize=14.0, color='blue', ha="left", va="top", transform=ax.transAxes)
    ax.legend(loc="best", fontsize=12)

    # --------------- #
    dchi2 = pz['chi2'].max() - pz['chi2'].min()
    chi2min, chi2max = pz['chi2'].min()-0.2*dchi2, pz['chi2'].max()+0.2*dchi2

    ax2 = fig.add_subplot(gs[1,0])
    ax = ax2
    ax.set_xlabel(r"$z$", fontsize=15.0, labelpad=7.0)
    ax.set_ylabel(r"$\chi^{2}$", fontsize=15.0, labelpad=7.0)
    # ax.set_xlim([xmin, xmax])
    ax.set_ylim([chi2min, chi2max])
    ax.set_xscale('log')
    ax.tick_params(axis="both", labelsize=15.0, pad=8.0)
    ax.tick_params(width=1.5, length=9.0)
    ax.tick_params(width=1.5, length=5.0, which="minor")
    for axis in ["top","bottom","left","right"]:
        ax.spines[axis].set_linewidth(1.5)
    # --------------- #
    ax.plot(pz['z'], pz['chi2'], lw=2.0, color="dimgray", alpha=0.7)
    ax.axvline(z_phot, 0., 1.,
               ls='--', lw=1.5, color='blue', alpha=0.7)
    ax.axvline(z_spec, 0., 1.,
               ls='--', lw=1.5, color='red', alpha=0.7)

    if (cut_img is not None):
        axins = inset_axes(ax, width="100%", height="100%",
                           bbox_to_anchor=(1.00, 0.70, 0.35, 0.80),
                           bbox_transform=ax.transAxes, borderpad=0)
        axins.tick_params(left=False, right=False, labelleft=False, labelright=False,
                          top=False, bottom=False, labeltop=False, labelbottom=False)
        for axis in ['top','bottom','left','right']:
            axins.spines[axis].set_linewidth(0.8)

        vmin, vmax = interval.get_limits(cut_img)
        axins.imshow(cut_img, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
        # fib = Circle((round(rth/pixel_scale), round(rth/pixel_scale)), radius=0.75/pixel_scale,
        #              linewidth=1.5, edgecolor='magenta', fill=False, alpha=0.9)
        # axins.add_artist(fib)

        e = Ellipse((cut_img.shape[1] // 2, cut_img.shape[0] // 2),
                    width=2*r_kron, height=2*r_kron*axis_ratio, angle=theta,
                    fill=False, color='magenta', linestyle='-', linewidth=2.0, zorder=10, alpha=0.8) 
        axins.add_patch(e)

    ax.text(1.02, 0.48, f"F200W={f200w_mag:.2f}",
            fontsize=12.5, color='dimgray', ha="left", va="bottom", transform=ax.transAxes)
    ax.text(1.02, 0.32, r"$r_{h}=$"+f"{r_h*pixel_scale:.2f}"+'"',
            fontsize=12.5, color='dimgray', ha="left", va="bottom", transform=ax.transAxes)
    ax.text(1.02, 0.16, f"F200W-F356W={color1:.2f}",
            fontsize=12.5, color='dimgray', ha="left", va="bottom", transform=ax.transAxes)
    ax.text(1.02, 0.00, f"F150W-F277W={color2:.2f}",
            fontsize=12.5, color='dimgray', ha="left", va="bottom", transform=ax.transAxes)

    if (cut_img2 is not None):
        axins = inset_axes(ax, width="100%", height="100%",
                           bbox_to_anchor=(1.00, 1.60, 0.35, 0.80),
                           bbox_transform=ax.transAxes, borderpad=0)
        axins.tick_params(left=False, right=False, labelleft=False, labelright=False,
                          top=False, bottom=False, labeltop=False, labelbottom=False)
        for axis in ['top','bottom','left','right']:
            axins.spines[axis].set_linewidth(0.8)

        try:
            vmin, vmax = interval.get_limits(cut_img2)
            axins.imshow(cut_img2, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
            e = Ellipse((cut_img.shape[1] // 2, cut_img.shape[0] // 2),
                        width=2*r_kron, height=2*r_kron*axis_ratio, angle=theta,
                        fill=False, color='magenta', linestyle='-', linewidth=2.0, zorder=10, alpha=0.8) 
            axins.add_patch(e)
        except:
            axins.imshow(np.zeros_like(cut_img), origin='lower', cmap='gray_r')


    plt.savefig(dir_fig+"/"+out, dpi=300)
    # plt.savefig("./fig_FAST.png", dpi=300)
    plt.close() 



# Image data
dir_align = "/data01/jhlee/DATA/JWST/A2744/Weaver+23/Reproject/"
totimg = fits.getdata(dir_align+"f277w.fits")
hstimg = fits.getdata(dir_align+"f814w.fits")
rth = 125

# Photometric data
dir_phot = "/data01/jhlee/DATA/JWST/A2744/Weaver+23/Phot/"
with open(dir_phot+"phot_data.pickle", 'rb') as fr:
    phot_data = pickle.load(fr)

# Plotting the EAZY results
dir_ezout = "FAST_INPUT/"
prefixes = ['photz_total', 'photz_total_w23']
for i in range(len(prefixes)):
    dz = read_zout(dir_ezout+prefixes[i]+".zout")
    objid = dz['id'].values
    
    if (i == 0):
        wav_eff = [[4328, 5959, 8084, 10577, 12500, 13971, 15418],
                   [11570, 15039, 19933, 27699, 35766, 40842, 44153]]
        color_inst = ["tomato", "darkorange"]
        label_inst = ["Observed HST photometry", "Observed JWST photometry"]

    # for j in tqdm.trange(len(objid)):
    #     objID = objid[j]

    for objID in [34014, 34967, 38415, 42523, 43310,
                  45844, 46047, 46841, 47425,
                  47978, 50062, 52243]:
        
        try:
            plot_eazy_sed(objID, prefixes[i], prefixes[i]+f"_{objID:05d}.png",
                          z_phot=dz.loc[dz['id'] == objID]['z_peak'].values[0],
                          z_spec=dz.loc[dz['id'] == objID]['z_spec'].values[0],
                          peak_prob=dz.loc[dz['id'] == objID]['peak_prob'].values[0],
                          dir_output="FAST_INPUT", dir_fig="EAZY_Figure", pixel_scale=0.04, 
                          n_inst=len(label_inst), wav_eff=wav_eff, color_inst=color_inst, label_inst=label_inst,
                          cut_img=totimg[round(phot_data['f200w']['y'].values[objID-1]-1-rth):round(phot_data['f200w']['y'].values[objID-1]-1+rth),
                                          round(phot_data['f200w']['x'].values[objID-1]-1-rth):round(phot_data['f200w']['x'].values[objID-1]-1+rth)],
                          cut_img2=hstimg[round(phot_data['f200w']['y'].values[objID-1]-1-rth):round(phot_data['f200w']['y'].values[objID-1]-1+rth),
                                           round(phot_data['f200w']['x'].values[objID-1]-1-rth):round(phot_data['f200w']['x'].values[objID-1]-1+rth)],
                          f200w_mag=phot_data['f200w']['mag_auto'].values[objID-1],
                          r_h=phot_data['f200w']['flxrad'].values[objID-1],
                          color1=phot_data['f200w']['mag_auto'].values[objID-1]-phot_data['f356w']['mag_auto'].values[objID-1],
                          color2=phot_data['f150w']['mag_auto'].values[objID-1]-phot_data['f277w']['mag_auto'].values[objID-1],
                          r_kron=phot_data['f200w']['kron'].values[objID-1]*phot_data['f200w']['a'].values[objID-1],
                          axis_ratio=phot_data['f200w']['b'].values[objID-1]/phot_data['f200w']['a'].values[objID-1],
                          theta=phot_data['f200w']['theta'].values[objID-1])

        except:
            pass      


# Printing the running time
print(f"--- {time.time()-start_time:.4f} sec ---")

