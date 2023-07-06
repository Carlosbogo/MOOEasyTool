import numpy as np
import sobol_seq
import minepy
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm

from utils.ADFAlgorithm import ADF


## Function to call other acquisition functions
def MultiObjectiveAcqGrid(function, GP, N : int = 12, M : int = 50, p : float = -1):
    xx = tf.convert_to_tensor((GP.upperBounds-GP.lowerBounds)*sobol_seq.i4_sobol_generate(GP.d,2**N)+GP.lowerBounds)
    if p>0:
        zz = function(GP, xx, N = N, M = M, p = p)
    else:
        zz = function(GP, xx, N = N, M = M)
    return xx[tf.math.argmax(zz)], tf.reduce_max(zz)