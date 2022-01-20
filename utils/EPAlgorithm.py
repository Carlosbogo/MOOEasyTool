"""
@author: Miguel Taibo Martínez

Date: Jan 2021
"""
import pdb
from scipy.stats import norm
import tensorflow as tf

def EP(m_x, v_x, means, vars):

    m_is = [0 for _ in range(len(means))]
    v_is = [10e10 for _ in range(len(vars))]

    for m_i, v_i, m_a, v_a in zip(m_is,v_is, means, vars):
        
        v_x_i = v_x * v_i / (v_i - v_x)
        m_x_i = m_x + (m_x-m_i) * v_x_i / v_i

        z = (m_a-m_x_i)/tf.sqrt(v_a+v_x_i)
        Z = 1-tf.math.reduce_prod(norm.cdf(z))

        Z_m = -(Z-1)/Z*norm.pdf(z)/norm.cdf(z)/tf.math.sqrt(v_a+v_x_i)
        Z_v = -(Z-1)/Z*norm.pdf(z)/norm.cdf(z)*z/2/(v_a+v_x_i)
        Z_m = tf.where(tf.math.is_nan(Z_m), 0, Z_m)
        Z_v = tf.where(tf.math.is_nan(Z_v), 0, Z_v)

        m_x = m_x_i + v_x_i*Z_m
        v_x = v_x_i - v_x_i*v_x_i * (Z_m*Z_m - 2*Z_v)

        v_i = v_x * v_x_i / (v_x_i - v_x)
        m_i = m_x + v_i /v_x_i* (m_x-m_x_i)

    return m_x, v_x