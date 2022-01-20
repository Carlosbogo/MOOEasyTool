from math import sqrt
import numpy as np
import gpflow
import sobol_seq

def XSquared(x,d):
    res = 0
    for i in range(d):
        res+=x[i]*x[i]
    return res

def XRoot(x,d):
    res = 0
    for i in range(d):
        res+=sqrt(x[i])
    return res

def f1(x,d):
    res = 0
    for i in range(d):
        res+=(x[i]+0.5)*(x[i]+0.5)
        res-=7/12
    return res

def f2(x,d):
    res = 0
    for i in range(d):
        res+=(x[i]-0.5)*(x[i]-0.5)
        res-=7/12
    return res

def f3(x,d):
    res = 0
    for i in range(d):
        res+=x[i]*x[i]
        res-=2/6
    return res

def GPRandomFunction(O:int, d:int, lowerBounds: float, upperBounds: float):

    k = gpflow.kernels.SquaredExponential()
    GPR = gpflow.models.GPR([np.array([[0.]*d]),np.array([[0.]*O])], kernel = k)

    xx = sobol_seq.i4_sobol_generate(d,100)
    sample = GPR.predict_f_samples(xx,1)

    GPR = gpflow.models.GPR([xx,sample[0]], kernel = k)

    return GPR.predict_y


