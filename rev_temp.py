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
dir_eazy = "/home/jlee/Downloads/eazy-photoz/"
dir_temp = dir_eazy+"templates/fsps_full/"

temp_file = "tweak_fsps_QSF_12_v3.param"
os.system("cp -rpv "+dir_temp+temp_file+" .")


# Reading & rewriting the template file with right path
t = np.genfromtxt(temp_file, dtype=None, encoding="ascii")

f = open(temp_file, "w")
for i in np.arange(len(t)):
	f.write(f"{t['f0'][i]:d} ")
	f.write(dir_temp+t['f1'][i].split("/")[-1]+" ")
	f.write(f"{t['f2'][i]:.1f} ")
	f.write(f"{t['f3'][i]:.1f} ")
	f.write(f"{t['f4'][i]:.1f}\n")
f.close()


# Printing the running time
print(f"--- {time.time()-start_time:.4f} sec ---")