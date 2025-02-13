"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
from scipy.stats import norm
import tensorflow as tf

def ADF(mean, var, pareto):
    for y in pareto:
        z = (y-mean)/tf.math.sqrt(var)
        print("Z = ", Z)
        Z = tf.reshape(1-tf.math.reduce_prod(norm.cdf(z),axis=1), (z.shape[0],1))
        Z = tf.where(Z<10**-6, 10**-6, Z)

        Z_m = -(Z-1)/Z*norm.pdf(z)/norm.cdf(z)/tf.math.sqrt(var)
        Z_v = -(Z-1)/Z*norm.pdf(z)/norm.cdf(z)*z/2/var
        Z_m = tf.where(tf.math.is_nan(Z_m), 0, Z_m)
        Z_v = tf.where(tf.math.is_nan(Z_v), 0, Z_v)

        mean = mean + var*Z_m

        var = var - var*var * (Z_m*Z_m - 2*Z_v)

    return mean, var