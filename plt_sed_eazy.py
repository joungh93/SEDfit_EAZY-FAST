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
from astropy.io import fits
import tqdm


def plot_seds(dir_eazyout, run_name, objids=None, save_data=False):
    
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

        if save_data:
            with open(dir_figs+f"ID-{objid:05d}.data", "wb") as fw:
                pickle.dump(data, fw)                    



dir_output = "EAZY_OUTPUT/"
run_name = "run6"

idc = np.array([  410,   460,   485,   747,   769,   875,  1178,  1227,  1316,
                 1327,  1337,  1947,  1960,  1976,  2171,  2201,  3093,  3247,
                 3297,  3303,  3381,  3515,  3528,  3768,  4050,  4062,  4198,
                 4334,  4675,  4778,  4928,  5039,  5264,  5322,  5379,  5780,
                 5788,  5840,  5875,  6009,  6376,  6482,  6573,  6706,  6780,
                 6919,  7000,  7019,  7191,  7364,  7455,  7626,  7680,  7796,
                 8001,  8051,  8074,  8199,  8256,  8364,  8565,  8715,  9089,
                 9114,  9218,  9389,  9453,  9479,  9499,  9508,  9552,  9628,
                 9630, 10017, 10131, 10183, 10250, 10368, 10372, 10383, 10474,
                10506, 10521, 10559, 10605, 10680, 10706, 10734, 10786, 10810,
                10970, 10998, 11112, 11239, 11325, 11416, 11470, 11545, 11613,
                11719, 11860, 12020, 12141, 12179, 12500, 12557, 12629, 12706,
                12732, 12739, 12805, 12883, 12997, 13029, 13060, 13066, 13477,
                13518, 13554, 13606, 13627, 13650, 13676, 13822, 13854, 13888,
                13955, 13989, 14029, 14078, 14128, 14172, 14235, 14250, 14332,
                14363, 14379, 14382, 14564, 14635])

plot_seds(dir_output, run_name, objids=idc, save_data=True)


