import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
# necessaty for pure ssh connection
# plt.switch_backend('agg')
# ----------------------------
# ########################################################
# 
# Check distributiob of training labels
# (c) 2019 DrSdl
# 
# ########################################################

import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils_06 import *
from sys import exit

np.random.seed(2018)

# get number of samples in hdf5 file
def GetTrainSamples(filename):
    training_dataset = h5py.File(filename, "r")
    liste=list(training_dataset.keys())
    anzahl=len(liste)
    return anzahl

print("Loading testing labels")
filen='GraphTrainIds_CUB.hdf5' # Parameters
filen='GraphTrainIds.hdf5'   # Type


Y_test_orig = load_chara_dataset_param(filen)
Y_test_orig = Y_test_orig.T

print(type(Y_test_orig)," ",Y_test_orig.shape)
#exit(0)
# ###################################################################


# Number of parameters to be trained
NumParameters=1  # TYPE
#NumParameters=2  # LIN
#NumParameters=3  # QUA
#NumParameters=4  # CUB
#NumParameters=3  # EXP
#NumParameters=2  # LOG


if NumParameters == 1:
    plt.subplot(221)
    plt.hist(Y_test_orig[0,:], bins=5, range=(1,+5),  facecolor='b', alpha=1.00)


if NumParameters == 2 and 'LIN' in filen:
    plt.subplot(221)
    plt.hist(Y_test_orig[1,:], bins=50, range=(-10,+10),  facecolor='b', alpha=1.00)

    plt.subplot(222)
    plt.hist(Y_test_orig[2,:], bins=50, range=(0,+10),  facecolor='b', alpha=1.00)



if NumParameters == 3 and 'QUA' in filen:
    plt.subplot(221)
    plt.hist(Y_test_orig[1,:], bins=50, range=(0,10),  facecolor='b', alpha=1.00)

    plt.subplot(222)
    plt.hist(Y_test_orig[2,:], bins=50, range=(-10,+10),  facecolor='b', alpha=1.00)

    plt.subplot(223)
    plt.hist(Y_test_orig[3,:], bins=50, range=(-4,+4),  facecolor='b', alpha=1.00)


if NumParameters == 4 and 'CUB' in filen:
    plt.subplot(221)
    plt.hist(Y_test_orig[1,:], bins=50, range=(0,+10),  facecolor='b', alpha=1.00)

    plt.subplot(222)
    plt.hist(Y_test_orig[2,:], bins=50, range=(0,+10),  facecolor='b', alpha=1.00)

    plt.subplot(223)
    plt.hist(Y_test_orig[3,:], bins=50, range=(-5,+5),  facecolor='b', alpha=1.00)

    plt.subplot(224)
    plt.hist(Y_test_orig[4,:], bins=50, range=(-2,+2),  facecolor='b', alpha=1.00)



if NumParameters == 3 and 'EXP' in filen:
    plt.subplot(221)
    plt.hist(Y_test_orig[1,:], bins=50, range=(-0.5,+0.5),  facecolor='b', alpha=1.00)

    plt.subplot(222)
    plt.hist(Y_test_orig[2,:], bins=50, range=(-1,+1),  facecolor='b', alpha=1.00)

    plt.subplot(223)
    plt.hist(Y_test_orig[3,:], bins=50, range=(-5,+5),  facecolor='b', alpha=1.00)


if NumParameters == 2 and 'LOG' in filen:
    plt.subplot(221)
    plt.hist(Y_test_orig[1,:], bins=50, range=(-5,+5),  facecolor='b', alpha=1.00)

    plt.subplot(222)
    plt.hist(Y_test_orig[2,:], bins=50, range=(-5,+5),  facecolor='b', alpha=1.00)



plt.show()