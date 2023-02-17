#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:58:18 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os, copy


# Copying the template file to the current diretory
dir_eazy = "/data01/jhlee/Downloads/eazy-photoz/"
dir_temp = dir_eazy+"templates/PEGASE2.0/"
# dir_temp = dir_eazy+"templates/fsps_full/"

temp_file = "pegase13.spectra.param"
# temp_file = "tweak_fsps_QSF_12_v3.param"
os.system("cp -rpv "+dir_temp+temp_file+" .")


# Reading & rewriting the template file with right path
with open(temp_file, "r") as f:
	ll = f.readlines()

nll = []
for i in range(len(ll)):
	ls = ll[i].split()
	datfile = ls[1]
	ls[1] = dir_temp+datfile.split("/")[-1]
	nll.append("  ".join(ls)+"\n")


with open(temp_file, "w") as f:
	f.writelines(nll)

# t = np.genfromtxt(temp_file, dtype=None, encoding="ascii")

# f = open(temp_file, "w")
# for i in np.arange(len(t)):
# 	f.write(f"{t['f0'][i]:d} ")
# 	f.write(dir_temp+t['f1'][i].split("/")[-1]+" ")
# 	f.write(f"{t['f2'][i]:.1f} ")
# 	f.write(f"{t['f3'][i]:.1f} ")
# 	f.write(f"{t['f4'][i]:.1f}\n")
# f.close()


# Printing the running time
print(f"--- {time.time()-start_time:.4f} sec ---")
