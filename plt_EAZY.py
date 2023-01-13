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
                  dir_output="FAST_INPUT", dir_fig="EAZY_Figure"):
    
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
    fig = plt.figure(1, figsize=(8,8))
    gs = GridSpec(2, 1, left=0.15, bottom=0.10, right=0.80, top=0.95, height_ratios=[6,3], hspace=0.30)
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
    ax.text(1.04, 0.95, f"ID-{objid:05d}", fontsize=16.0, fontweight="bold", color="black",
            ha="left", va="top", transform=ax.transAxes)
    # ax.text(1.05, 0.85, suffix_label, fontsize=14.0, fontweight="bold", color="gray",
    #         ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.78, r"$z_{\rm spec}=$"+f"{z_spec:.4f}",
        # dz.loc[dz['id'] == objid]['z_spec'].values[0]:.4f}",
            fontsize=12.0, color='red', ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.70, r"$z_{\rm phot}=$"+f"{z_phot:.4f}",
        # dz.loc[dz['id'] == objid]['z_peak'].values[0]:.4f}",
            fontsize=12.0, color='blue', ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.62, r"$p(z)=$"+f"{peak_prob:.4f}",
        # dz.loc[dz['id'] == objid]['peak_prob'].values[0]:.4f}",
            fontsize=12.0, color='blue', ha="left", va="top", transform=ax.transAxes)
    # ax.text(1.02, 0.62, r"${\rm log} (M_{{\rm star}}/M_{\odot})=%.3f$" \
    #         %(df_output['lmass'][df_output['id'] == objids]), fontsize=12.0, color="blue",
    #         ha="left", va="top", transform=ax.transAxes)
    # ax.text(1.02, 0.54, r"$[Z/Z_{\odot}]=%.2f$ (fixed)" \
    #         %(np.log10(df_output['metal'][df_output['id'] == objids]/0.02)), fontsize=12.0, color="black",
    #         ha="left", va="top", transform=ax.transAxes)
    # ax.text(1.02, 0.46, r"$A_{V}=%.1f$ mag" \
    #         %(df_output['Av'][df_output['id'] == objids]), fontsize=12.0, color="blue",
    #         ha="left", va="top", transform=ax.transAxes)
    # ax.text(1.02, 0.38, r"$t_{\rm age}=%.2f$ Gyr" \
    #         %(10.0**(df_output['lage'][df_output['id'] == objids]-9.0)), fontsize=12.0, color="blue",
    #         ha="left", va="top", transform=ax.transAxes)
    # ax.text(1.02, 0.30, r"$\tau=%.2f$ Gyr" \
    #         %(10.0**(df_output['ltau'][df_output['id'] == objids]-9.0)), fontsize=12.0, color="blue",
    #         ha="left", va="top", transform=ax.transAxes)
    # # chisq = np.sum(((obs["maggies"]-pphot)/obs["maggies_unc"])**2.0)
    # # dof = np.sum(obs['phot_mask']) - len(model.theta)
    # # rchisq = chisq / dof
    # ax.text(1.02, 0.10, r"$\chi^{2}=%.3f$" \
    #         %(df_output['chi2'][df_output['id'] == objids]), fontsize=12.0, color="red",
    #         ha="left", va="top", transform=ax.transAxes)
    # # ax.text(1.02, 0.25, r"$\chi_{v}^{2}=%.3f$" \
    # #         %(rchisq), fontsize=20.0, color="red",
    # #         ha="left", va="top", transform=ax.transAxes)
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
               # dz.loc[dz['id'] == objid]['z_peak'].values[0], 0., 1.,
               ls='--', lw=1.5, color='blue', alpha=0.7)
    ax.axvline(z_spec, 0., 1.,
               # dz.loc[dz['id'] == objid]['z_spec'].values[0], 0., 1.,
               ls='--', lw=1.5, color='red', alpha=0.7)

    plt.savefig(dir_fig+"/"+out, dpi=300)
    # plt.savefig("./fig_FAST.png", dpi=300)
    plt.close() 


dir_ezout = "FAST_INPUT/"
prefixes = ['photz_hst', 'photz_jwst', 'photz_total']
for i in range(len(prefixes)):
    dz = read_zout(dir_ezout+prefixes[i]+".zout")
    objid = dz['id'].values
    for j in tqdm.trange(len(objid)):
    # range(len(objid)):
        try:
            plot_eazy_sed(objid[j], prefixes[i], prefixes[i]+f"_{objid[j]:05d}.png",
                          z_phot=dz.loc[dz['id'] == objid[j]]['z_peak'].values[0],
                          z_spec=dz.loc[dz['id'] == objid[j]]['z_spec'].values[0],
                          peak_prob=dz.loc[dz['id'] == objid[j]]['peak_prob'].values[0])
        except:
            pass   


# Printing the running time
print(f"--- {time.time()-start_time:.4f} sec ---")


# dz_hst   = read_zout(dir_ezout+'photz_hst.zout')
# dz_jwst  = read_zout(dir_ezout+'photz_jwst.zout')
# dz_total = read_zout(dir_ezout+'photz_total.zout')

# objid = dz_hst['id'].values
# prefixes = ['photz_hst', 'photz_jwst', 'photz_total']
# for i in range(len(objid)):
#     for j, prefix in enumerate(prefixes):
#         if (j == 0):
#             dz = dz_hst
#         if (j == 1):
#             dz = dz_jwst
#         if (j == 2):
#             dz = dz_total
#         plot_eazy_sed(objid[i], prefix, prefix+f"_{objid[i]:05d}.png",
#                       z_phot=dz['z_peak'], z_spec=dz['z_spec'], peak_prob=dz['peak_prob'])





# for i in range(len(prefixes)):
#     dz = read_zout(dir_ezout+prefixes+".zout")


# read_zout(file)



# plot_eazy_sed(2900, 'photz_jwst', 'test.png', z_phot=0.1, z_spec=-1.0, peak_prob=1.0,
#               dir_output="FAST_INPUT", dir_fig="EAZY_Figure")

# objid, prefix, out, z_phot=0.1, z_spec=-1.0, peak_prob=1.0,
#                   dir_output="FAST_INPUT", dir_fig="EAZY_Figure"







# import os
# import numpy as np

# def readEazyBinary(MAIN_OUTPUT_FILE='photz', OUTPUT_DIRECTORY='./OUTPUT', CACHE_FILE='Same'):
#     """
# tempfilt, coeffs, temp_sed, pz = readEazyBinary(MAIN_OUTPUT_FILE='photz', \
#                                                 OUTPUT_DIRECTORY='./OUTPUT', \
#                                                 CACHE_FILE = 'Same')

#     Read Eazy BINARY_OUTPUTS files into structure data.
    
#     If the BINARY_OUTPUTS files are not in './OUTPUT', provide either a relative or absolute path
#     in the OUTPUT_DIRECTORY keyword.
    
#     By default assumes that CACHE_FILE is MAIN_OUTPUT_FILE+'.tempfilt'.
#     Specify the full filename if otherwise. 
#     """
        
#     root = OUTPUT_DIRECTORY+'/'+MAIN_OUTPUT_FILE
    
#     ###### .tempfilt
#     if CACHE_FILE == 'Same':
#         CACHE_FILE = root+'.tempfilt'
    
#     if os.path.exists(CACHE_FILE) is False:
#         print(('File, %s, not found.' %(CACHE_FILE)))
#         return -1,-1,-1,-1
    
#     f = open(CACHE_FILE,'rb')
    
#     s = np.fromfile(file=f,dtype=np.int32, count=4)
#     NFILT=s[0]
#     NTEMP=s[1]
#     NZ=s[2]
#     NOBJ=s[3]
#     tempfilt = np.fromfile(file=f,dtype=np.double,count=NFILT*NTEMP*NZ).reshape((NZ,NTEMP,NFILT)).transpose()
#     lc = np.fromfile(file=f,dtype=np.double,count=NFILT)
#     zgrid = np.fromfile(file=f,dtype=np.double,count=NZ)
#     fnu = np.fromfile(file=f,dtype=np.double,count=NFILT*NOBJ).reshape((NOBJ,NFILT)).transpose()
#     efnu = np.fromfile(file=f,dtype=np.double,count=NFILT*NOBJ).reshape((NOBJ,NFILT)).transpose()
    
#     f.close()
    
#     tempfilt  = {'NFILT':NFILT,'NTEMP':NTEMP,'NZ':NZ,'NOBJ':NOBJ,\
#                  'tempfilt':tempfilt,'lc':lc,'zgrid':zgrid,'fnu':fnu,'efnu':efnu}
    
#     ###### .coeff
#     f = open(root+'.coeff','rb')
    
#     s = np.fromfile(file=f,dtype=np.int32, count=4)
#     NFILT=s[0]
#     NTEMP=s[1]
#     NZ=s[2]
#     NOBJ=s[3]
#     coeffs = np.fromfile(file=f,dtype=np.double,count=NTEMP*NOBJ).reshape((NOBJ,NTEMP)).transpose()
#     izbest = np.fromfile(file=f,dtype=np.int32,count=NOBJ)
#     tnorm = np.fromfile(file=f,dtype=np.double,count=NTEMP)
    
#     f.close()
    
#     coeffs = {'NFILT':NFILT,'NTEMP':NTEMP,'NZ':NZ,'NOBJ':NOBJ,\
#               'coeffs':coeffs,'izbest':izbest,'tnorm':tnorm}
              
#     ###### .temp_sed
#     f = open(root+'.temp_sed','rb')
#     s = np.fromfile(file=f,dtype=np.int32, count=3)
#     NTEMP=s[0]
#     NTEMPL=s[1]
#     NZ=s[2]
#     templam = np.fromfile(file=f,dtype=np.double,count=NTEMPL)
#     temp_seds = np.fromfile(file=f,dtype=np.double,count=NTEMPL*NTEMP).reshape((NTEMP,NTEMPL)).transpose()
#     da = np.fromfile(file=f,dtype=np.double,count=NZ)
#     db = np.fromfile(file=f,dtype=np.double,count=NZ)
    
#     f.close()
    
#     temp_sed = {'NTEMP':NTEMP,'NTEMPL':NTEMPL,'NZ':NZ,\
#               'templam':templam,'temp_seds':temp_seds,'da':da,'db':db}
              
#     ###### .pz
#     if os.path.exists(root+'.pz'):
#         f = open(root+'.pz','rb')
#         s = np.fromfile(file=f,dtype=np.int32, count=2)
#         NZ=s[0]
#         NOBJ=s[1]
#         chi2fit = np.fromfile(file=f,dtype=np.double,count=NZ*NOBJ).reshape((NOBJ,NZ)).transpose()

#         ### This will break if APPLY_PRIOR No
#         s = np.fromfile(file=f,dtype=np.int32, count=1)
        
#         if len(s) > 0:
#             NK = s[0]
#             kbins = np.fromfile(file=f,dtype=np.double,count=NK)
#             priorzk = np.fromfile(file=f, dtype=np.double, count=NZ*NK).reshape((NK,NZ)).transpose()
#             kidx = np.fromfile(file=f,dtype=np.int32,count=NOBJ)
#             pz = {'NZ':NZ,'NOBJ':NOBJ,'NK':NK, 'chi2fit':chi2fit, 'kbins':kbins, 'priorzk':priorzk,'kidx':kidx}
#         else:
#             pz = None
        
#         f.close()
        
#     else:
#         pz = None
    
#     if False:
#         f = open(root+'.zbin','rb')
#         s = np.fromfile(file=f,dtype=np.int32, count=1)
#         NOBJ=s[0]
#         z_a = np.fromfile(file=f,dtype=np.double,count=NOBJ)
#         z_p = np.fromfile(file=f,dtype=np.double,count=NOBJ)
#         z_m1 = np.fromfile(file=f,dtype=np.double,count=NOBJ)
#         z_m2 = np.fromfile(file=f,dtype=np.double,count=NOBJ)
#         z_peak = np.fromfile(file=f,dtype=np.double,count=NOBJ)
#         f.close()
        
#     ###### Done.    
#     return tempfilt, coeffs, temp_sed, pz


# def generate_sed_arrays(MAIN_OUTPUT_FILE='photz', OUTPUT_DIRECTORY='./OUTPUT', CACHE_FILE='Same'):
#     """
#     Generate full "obs_sed" and "temp_sed" arrays from the stored
#     EAZY binary files.
    
#     Returns
#     -------
#     tempfilt : dict
#         Dictionary read from the `tempfilt` file.  Keys include the input 
#         photometry ('fnu', 'efnu') and the filter central wavelengths
#         ('lc').
    
#     z_grid : array
#         Redshift on the input grid nearest to the best output photo-z.  The
#         SEDs are generated here based on the fit coefficients.
    
#     obs_sed : (NFILT, NOBJ) array
#         Template photometry.  Has fnu units with the same scaling as the 
#         input photometry (i.e., AB with zeropoint `PRIOR_ABZP`).
    
#     templam : (NTEMPL) array
#         Rest-frame wavelengths of the full template spectra
    
#     temp_sed : (NTEMPL, NOBJ) array
#         Full best-fit template spectra in same units as `obs_sed`.
        
#     """
    
#     import os
#     #import threedhst.eazyPy as eazy
    
#     out = readEazyBinary(MAIN_OUTPUT_FILE=MAIN_OUTPUT_FILE, OUTPUT_DIRECTORY=OUTPUT_DIRECTORY, CACHE_FILE=CACHE_FILE)
#     tempfilt, coeffs, temp_seds, pz = out
    
#     # The redshift gridpoint where the SEDs are evaluated
#     z_grid = tempfilt['zgrid'][coeffs['izbest']]
    
#     obs_sed = np.zeros_like(tempfilt['fnu'])
#     for i in range(tempfilt['NOBJ']):
#         obs_sed[:,i] = np.dot(tempfilt['tempfilt'][:,:,coeffs['izbest'][i]],
#                          coeffs['coeffs'][:,i])
    
    
#     ### Temp_sed
#     temp_sed = (np.dot(temp_seds['temp_seds'],coeffs['coeffs']).T*(temp_seds['templam']/5500.)**2).T
    
#     ## IGM
#     lim1 = np.where(temp_seds['templam'] < 912)
#     temp_sed[lim1,:] *= 0
    
#     lim2 = np.where((temp_seds['templam'] >= 912) & (temp_seds['templam'] < 1026))
#     db = 1.-temp_seds['db'][coeffs['izbest']]
#     temp_sed[lim2,:] *= db
    
#     lim3 = np.where((temp_seds['templam'] >= 1026) & (temp_seds['templam'] < 1216))
#     da = 1.-temp_seds['da'][coeffs['izbest']]
#     temp_sed[lim3,:] *= da
    
#     templam = temp_seds['templam']
    
#     return tempfilt, z_grid, obs_sed, templam, temp_sed
    
# def demo():
#     """ 
#     Demo on the GOODS-N test data
#     """
#     import matplotlib.pyplot as plt
    
#     tempfilt, z_grid, obs_sed, templam, temp_sed = generate_sed_arrays(MAIN_OUTPUT_FILE='photz', OUTPUT_DIRECTORY='./OUTPUT', CACHE_FILE='Same')
    
#     idx = 17
#     plt.errorbar(tempfilt['lc'], tempfilt['fnu'][:,idx],
#                  tempfilt['efnu'][:,idx], marker='.', color='k',
#                   linestyle='None', zorder=1000, label='Data')
                  
#     plt.plot(tempfilt['lc'], obs_sed[:,idx], marker='.', color='r', alpha=0.8,
#              label='Template photometry')
             
#     plt.plot(templam*(1+z_grid[idx]), temp_sed[:,idx], color='b', alpha=0.8,
#              label='Template spectrum')
    
#     plt.loglog(); plt.xlim(2000,3.e4)
#     plt.legend(loc='lower right')
#     plt.xlabel(r'Observed wavelength, $\mathrm{\AA}$')
#     plt.ylabel(r'Observed flux, $f_\nu$ @ PRIOR_ABZP')
#     plt.tight_layout()
#     
