# Imports
import eazy

# Module versions
import importlib
import sys
import time
print(time.ctime() + '\n')

print(sys.version + '\n')

for module in ['numpy', 'scipy', 'matplotlib','astropy','eazy']:#, 'prospect']:
    #print(module)
    mod = importlib.import_module(module)
    print('{0:>20} : {1}'.format(module, mod.__version__))

import glob, os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import tqdm


def plot_seds(dir_eazyout, run_name, objids=None):
    
    if not (dir_eazyout[-1] == "/"):
        dir_eazyout += "/"

    dir_figs = dir_eazyout+run_name+".sed/"
    if not os.path.exists(dir_figs):
        os.system("mkdir "+dir_figs)

    dz, hz = fits.getdata(dir_eazyout+run_name+".eazypy.zout.fits", ext=1, header=True)

    with open(dir_eazyout+run_name+".eazypy.zphot.pickle", "rb") as fr:
        pred = pickle.load(fr)

    if objids is None:
        objids = dz['id']
    n_obj = len(objids)

    for i in tqdm.trange(n_obj):
        objid = objids[i]
        fig, data = pred.show_fit(objid, id_is_idx=False, show_fnu=1,
                                  xlim=[0.3, 9], show_components=True)
        plt.savefig(dir_figs+f"ID-{objid:05d}.png", dpi=300)
        plt.close()        


dir_output = "EAZY_OUTPUT/"
run_name = "run6"
plot_seds(dir_output, run_name, objids=None) ## save data?

