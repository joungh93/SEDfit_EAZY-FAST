### Imports

# Printing the versions of packages
from importlib_metadata import version
for pkg in ['numpy', 'matplotlib', 'pandas']:
    print(pkg+": ver "+version(pkg))

# importing necessary modules
import numpy as np
import glob, os
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from pystilts import wcs_match1
from astropy.io import fits
from matplotlib.patches import Rectangle

c = 2.99792e+5    # km/s

### Load the Data

# ----- Loading the photometry data ----- #
dir_phot = "/data01/jhlee/DATA/JWST/A2744/Weaver+23/Phot/"

# load data
with open(dir_phot+"phot_data.pickle", 'rb') as fr:
    phot_data = pickle.load(fr)

dir_w23 = "/data01/jhlee/DATA/JWST/A2744/Weaver+23/"
df_w23 = fits.getdata(dir_w23+"UNCOVER_DR1_LW_D070_catalog.fits")
df_w23 = pd.DataFrame(df_w23)

z_spec = df_w23['z_spec'].values


# ----- Reading *.cat ----- #
def read_cat(file):
    with open(file, "r") as f:
        ll = f.readline()
    colnames = ll.split()[1:]
    
    df = np.genfromtxt(file, dtype=None, encoding='ascii', names=colnames)
    df = pd.DataFrame(df)
    return df

df_tot_run1 = read_cat("EAZY_INPUT/flux_EAZY_total_run1.cat")
df_tot_run2 = read_cat("EAZY_INPUT/flux_EAZY_total_run2.cat")
df_tot_run3 = read_cat("EAZY_INPUT/flux_EAZY_total_run3.cat")
df_tot_run4 = read_cat("EAZY_INPUT/flux_EAZY_total_run4.cat")
df_tot_run5 = read_cat("EAZY_INPUT/flux_EAZY_total_run5.cat")

id_Cat = df_tot_run1['id'].values


# ----- Reading *.zout ----- #
def read_zout(file):
    f = open(file, "r")
    ll = f.readlines()
    f.close()
    cols_zfile = ll[0].split()[1:]
    cols_zfile = tuple(cols_zfile)
    
    dz = np.genfromtxt(file, dtype=None, comments="#", names=cols_zfile)
    dz = pd.DataFrame(dz)
    return dz

dz_tot_run1 = read_zout("FAST_INPUT/photz_total_run1.zout")
dz_tot_run2 = read_zout("FAST_INPUT/photz_total_run2.zout")
dz_tot_run3 = read_zout("FAST_INPUT/photz_total_run3.zout")
dz_tot_run4 = read_zout("FAST_INPUT/photz_total_run4.zout")
dz_tot_run5 = read_zout("FAST_INPUT/photz_total_run5.zout")


# ----- WCS matching ----- #
tol = 0.5   # arcsec
idx_matched, idx_spec, sepr = wcs_match1(phot_data['f200w'].iloc[id_Cat-1]['ra'].values,
                                         phot_data['f200w'].iloc[id_Cat-1]['dec'].values,
                                         df_w23['ra'].values, df_w23['dec'].values, tol, ".")
print(len(idx_matched))


# ----- Redshift comparison ----- #
def plot_comp(z_spec, z_phot, out, label_x='', label_y='', title='',
              z_clu=0.30, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.10,
              check_highz=True):
    z_cnd = ((z_phot > 0) & (z_spec > 0))
    z_mem = (c*np.abs(z_spec-z_clu)/(1+z_clu) < dv_mem)
    print(f"Objects : {np.sum(z_cnd):d}")
    print(f"Members : {np.sum(z_cnd & z_mem):d}")
    
    dz = np.abs(z_spec-z_phot)/(1+z_spec)
    sigma = 1.48*np.median(np.abs(dz[z_cnd]-np.median(dz[z_cnd]))/(1+z_spec[z_cnd]))
    if slope is None:
        slope = 5.0*sigma
        print(f"Slope: {slope:.3f}")
    
    outlier = (z_cnd & (dz >= slope))
    print(f"Outliers: {np.sum(outlier):d}")
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(z_spec[z_cnd & ~z_mem], z_phot[z_cnd & ~z_mem], 'o', ms=3.0, mew=0.5,
            color='tab:blue', alpha=0.6)
    ax.plot(z_spec[z_cnd & z_mem],  z_phot[z_cnd & z_mem],  'o', ms=3.0, mew=0.5,
            color='tab:red', alpha=0.6)
    sym_1, = ax.plot(1.0e-10, 1.0e-10, 'o', ms=4.0, mew=0.8,
                     color='tab:blue', alpha=0.7, label="Spec-z sample")
    sym_2, = ax.plot(1.0e-10, 1.0e-10, 'o', ms=4.0, mew=0.8,
                     color='tab:red', alpha=0.7, label="Cluster member")
    ax.plot([xmin, xmax], [xmin, xmax], '-', lw=1.5, color='gray', alpha=0.75)
    # ax.plot([(xmin+slope)/(1.0-slope), xmax], [xmin, (1.0-slope)*xmax-slope],
    #       '--', lw=1.2, color='gray', alpha=0.7)
    # ax.plot([xmin, xmax], [(1.0+slope)*xmin+slope, (1.0+slope)*xmax+slope],
    #       '--', lw=1.2, color='gray', alpha=0.7)
    # ax.plot([0.1, 10.], [(1.0+slope)*0.1+slope, (1.0+slope)*10.+slope],
    #         '--', lw=1.2, color='k', alpha=0.7)
    xx = np.logspace(np.log10(xmin), np.log10(xmax), 1000)
    ax.plot(xx, (1.0-slope)*xx-slope, '--', lw=1.2, color='gray', alpha=0.7)
    ax.plot(xx, (1.0+slope)*xx+slope, '--', lw=1.2, color='gray', alpha=0.7)
    # ax.set_xticks([0., 2., 4., 6., 8.])
    # ax.set_yticks([0., 2., 4., 6., 8.])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([xmin, xmax])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(label_x, fontsize=10.0)
    ax.set_ylabel(label_y, fontsize=10.0)
    ax.tick_params(axis='both', labelsize=10.0)
    ax.text(0.05, 0.95, title,
            fontsize=12.0, fontweight="bold", color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(0.03, 0.88, r"$N$"+f" = {np.sum(z_cnd):d} ({np.sum(outlier):d},"+ \
            f" {100.*np.sum(outlier)/np.sum(z_cnd):.1f}%)",
            fontsize=9.0, color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(0.03, 0.83, r"$\sigma_{\rm NMAD}$"+ \
            f" = {sigma:.3f}",
            fontsize=9.0, color="black",
            ha="left", va="top", transform=ax.transAxes)
    if check_highz:
        highz_cnd = (z_spec > 1.5)
        ax.axvline(1.5, 0.0, 1.0, ls='-.', lw=1.2, color='gray', alpha=0.5)
        ax.text(0.95, 0.09, r"$N(z_{\rm s}>1.5)$"+ \
                f" = {np.sum(z_cnd & highz_cnd):d} "+ \
                f"({np.sum(outlier & highz_cnd):d},"+ \
                f" {100.*np.sum(outlier & highz_cnd)/np.sum(z_cnd & highz_cnd):.1f}%)",
                fontsize=9.0, color="black",
                ha="right", va="bottom", transform=ax.transAxes                )
        sigma_highz = 1.48*np.median(np.abs(dz[z_cnd & highz_cnd]- \
                      np.median(dz[z_cnd & highz_cnd]))/(1+z_spec[z_cnd & highz_cnd]))
        ax.text(0.95, 0.03, r"$\sigma_{\rm NMAD}~(z_{\rm s}>1.5)$"+ \
                f" = {sigma_highz:.3f}",
                fontsize=9.0, color="black",
                ha="right", va="bottom", transform=ax.transAxes)
    # ax.text(0.95, 0.05, r"$\delta z~/~(1+z)$"+ \
    #         f" = {np.mean(np.abs(z_spec[z_cnd]-z_phot[z_cnd])/(1+z_spec[z_cnd])):.3f}",
    #         fontsize=10.0, color="black",
    #         ha="right", va="bottom", transform=ax.transAxes)
    # ax.plot(z_spec[outlier],  z_phot[outlier],  'o', ms=3.0, mew=0.5,
    #         color='tab:purple', alpha=0.9)
    ax.legend(handles=[sym_1, sym_2], fontsize=8.0, loc=(0.03, 0.03),
              handlelength=0, frameon=True, borderpad=0.8, handletextpad=0.8,
              framealpha=0.6, edgecolor='gray')
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

# Diagrams
dzs = [dz_tot_run1, dz_tot_run2, dz_tot_run3, dz_tot_run4, dz_tot_run5]
run_names = ["run1", "run2", "run3", "run4", "run5"]

### z_spec vs. z_phot from EAZY
for i, dz in enumerate(dzs):
    plot_comp(dz['z_spec'], dz['z_peak'], "Fig1-comp_z_total_"+run_names[i]+".png",
              label_x=r"$z_{\rm spec}$", label_y=r"$z_{\rm phot}$",
              title="Phot-z from EAZY (HST+JWST)",
              z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.15,
              check_highz=True)

# # 1) z_spec vs. z_phot from EAZY (run1)
# plot_comp(dz_tot_run1['z_spec'], dz_tot_run1['z_peak'],
#           "Fig1-comp_z_total_run1.png",
#           label_x=r"$z_{\rm spec}$", label_y=r"$z_{\rm phot}$",
#           title="Phot-z from EAZY (HST+JWST)",
#           z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.15,
#           check_highz=True)

# # 2) z_spec vs. z_phot from EAZY (run2)
# plot_comp(dz_tot_run2['z_spec'], dz_tot_run2['z_peak'],
#           "Fig1-comp_z_total_run2.png",
#           label_x=r"$z_{\rm spec}$", label_y=r"$z_{\rm phot}$",
#           title="Phot-z from EAZY (HST+JWST)",
#           z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.15,
#           check_highz=True)

# # 3) z_spec vs. z_phot from EAZY (run3)
# plot_comp(dz_tot_run3['z_spec'], dz_tot_run3['z_peak'],
#           "Fig1-comp_z_total_run3.png",
#           label_x=r"$z_{\rm spec}$", label_y=r"$z_{\rm phot}$",
#           title="Phot-z from EAZY (HST+JWST)",
#           z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.15,
#           check_highz=True)

# # 4) z_spec vs. z_phot from EAZY (run4)
# plot_comp(dz_tot_run4['z_spec'], dz_tot_run4['z_peak'],
#           "Fig1-comp_z_total_run4.png",
#           label_x=r"$z_{\rm spec}$", label_y=r"$z_{\rm phot}$",
#           title="Phot-z from EAZY (HST+JWST)",
#           z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.15,
#           check_highz=True)

# # 4) z_spec vs. z_phot from EAZY (run4)
# plot_comp(dz_tot_run4['z_spec'], dz_tot_run4['z_peak'],
#           "Fig1-comp_z_total_run4.png",
#           label_x=r"$z_{\rm spec}$", label_y=r"$z_{\rm phot}$",
#           title="Phot-z from EAZY (HST+JWST)",
#           z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.15,
#           check_highz=True)


# ----- Color-color diagrams ----- #
def plot_ccd(z_spec, z_phot, out, ids, title='',
             z_clu=0.30, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.10,
             check_box=False, box_x0=[], box_y0=[], box_width=[], box_height=[]):
    z_cnd = ((z_phot > 0) & (z_spec > 0))
    z_mem = (c*np.abs(z_spec-z_clu)/(1+z_clu) < dv_mem)

    dz = np.abs(z_spec-z_phot)/(1+z_spec)
    sigma = 1.48*np.median(np.abs(dz[z_cnd]-np.median(dz[z_cnd]))/(1+z_spec[z_cnd]))
    if slope is None:
        slope = 5.0*sigma
        print(f"Slope: {slope:.3f}")
    
    outlier = (z_cnd & (dz >= slope))
    
    match_zs = (z_cnd & (dz < slope))
    high_zs  = (outlier & (z_spec > z_phot))
    low_zs   = (outlier & (z_spec < z_phot))

    id_member   = ids[z_mem]-1
    id_match_zs = ids[match_zs & ~z_mem]-1
    id_high_zs  = ids[high_zs & ~z_mem]-1
    id_low_zs   = ids[low_zs & ~z_mem]-1

    c1 = ['f200w', 'f356w']
    c2 = ['f150w', 'f277w']
    color1 = phot_data[c1[0]]['mag_auto'].values - phot_data[c1[1]]['mag_auto'].values
    color2 = phot_data[c2[0]]['mag_auto'].values - phot_data[c2[1]]['mag_auto'].values

    syms = []

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.plot(color1[id_match_zs], color2[id_match_zs],
            'o', ms=3.0, color='dodgerblue', mew=0.5, alpha=0.6)
    sym_match_zs, = ax.plot(-100., -100., 'o', ms=4.0, color='dodgerblue', mew=0.8, alpha=0.7,
                            label="z_phot ~ z_spec")
    if (len(id_match_zs) > 0.):
        syms.append(sym_match_zs)

    ax.plot(color1[id_high_zs], color2[id_high_zs],
            'o', ms=3.0, color='magenta', mew=0.5, alpha=0.6)
    sym_high_zs, = ax.plot(-100., -100., 'o', ms=4.0, color='magenta', mew=0.8, alpha=0.7,
                           label="z_phot < z_spec")
    if (len(id_high_zs) > 0.):
        syms.append(sym_high_zs)

    ax.plot(color1[id_low_zs], color2[id_low_zs],
            'o', ms=3.0, color='darkorange', mew=0.5, alpha=0.6)
    sym_low_zs, = ax.plot(-100., -100., 'o', ms=4.0, color='darkorange', mew=0.8, alpha=0.7,
                          label="z_phot > z_spec")
    if (len(id_low_zs) > 0.):
        syms.append(sym_low_zs)

    ax.plot(color1[id_member], color2[id_member],
            'o', ms=3.0, color='tab:red', mew=0.5, alpha=0.8)
    sym_mem, = ax.plot(-100., -100., 'o', ms=4.0, color='tab:red', mew=0.8, alpha=0.8,
                       label="Cluster memeber")
    if (len(id_member) > 0.):
        syms.append(sym_mem)
    
    id_box = []
    if check_box:
        for i in range(len(box_x0)):
            box = Rectangle((box_x0[i], box_y0[i]),
                            width=box_width[i], height=box_height[i],
                            ls='-', lw=1.0, edgecolor='gray', fill=False, alpha=0.6)
            ax.add_artist(box)
            color_cnd = ((color1[id_high_zs] >= box_x0[i]) & \
                         (color1[id_high_zs] <= box_x0[i]+box_width[i]) & \
                         (color2[id_high_zs] >= box_y0[i]) & \
                         (color2[id_high_zs] <= box_y0[i]+box_height[i]))
            id_box.append(id_high_zs.values[color_cnd])

    ax.text(0.05, 0.95, title,
            fontsize=12.0, fontweight="bold", color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.set_xlim([-1.9, 1.9])
    ax.set_ylim([-1.9, 1.9])
    ax.set_xlabel(c1[0].upper()+"-"+c1[1].upper())
    ax.set_ylabel(c2[0].upper()+"-"+c2[1].upper())
    plt.legend(handles=syms, fontsize=8.0, loc=(0.03, 0.03),
               handlelength=0, frameon=True, borderpad=0.8, handletextpad=0.8,
               framealpha=0.6, edgecolor='gray')
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    return id_box


### Color-color diagram
for i, dz in enumerate(dzs):
	plot_ccd(dz['z_spec'], dz['z_peak'],
	         "Fig1-ccd_"+run_names[i]+".png", dz['id'], title="Color-color Diagram (All)",
	         z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.15)

# id_check = plot_ccd(dz_tot_jlee['z_spec'].loc[dz_tot_jlee['z_spec'] > 1.5],
#                     dz_tot_jlee['z_peak'].loc[dz_tot_jlee['z_spec'] > 1.5],
#                     "Fig1-ccd_jlee_highz.png", dz_tot_jlee['id'].loc[dz_tot_jlee['z_spec'] > 1.5],
#                     title=r'Color-color Diagram ($z_{\rm s} > 1.5$)',
#                     z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.10,
#                     check_box=True, box_x0=[-1.2,-1.0,0.2], box_y0=[0.05,-1.0,-0.5],
#                     box_width=[0.7,1.0,0.8], box_height=[0.45,0.6,0.6])
#     # box region 1: upper left
#     # box region 2: lower left
#     # box region 3: lower right

