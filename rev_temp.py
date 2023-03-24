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
dirs_temp = [dir_eazy+"templates/fsps_full/",
             dir_eazy+"templates/PEGASE2.0/",
             dir_eazy+"templates/sfhz/"]
temp_files = ["tweak_fsps_QSF_12_v3.param",
              "pegase13.spectra.param",
              "carnall_sfhz_13.param"]

for i in range(len(temp_files)):
	os.system("cp -rpv "+dirs_temp[i]+temp_files[i]+" .")
	os.system("cp -rpv "+dirs_temp[i]+temp_files[i]+".fits .")


# Reading & rewriting the template file with right path
for i in range(len(temp_files)):
	with open(temp_files[i], "r") as f:
		ll = f.readlines()

	nll = []
	for j in range(len(ll)):
		ls = ll[j].split()
		datfile = ls[1]
		ls[1] = dirs_temp[i]+datfile.split("/")[-1]
		nll.append("  ".join(ls)+"\n")

	with open(temp_files[i], "w") as f:
		f.writelines(nll)


# Printing the running time
print(f"--- {time.time()-start_time:.4f} sec ---")
