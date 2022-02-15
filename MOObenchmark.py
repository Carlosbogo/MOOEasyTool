import numpy as np
from benchmarkFunctions.ackeley import ackley
from benchmarkFunctions.crossit import crossitNdim
from benchmarkFunctions.gramacy import gramacy

def MOOquadratic(xx, c1=-0.5, c2=0.5):
    return np.array([np.sum((xx-c1)**2),np.sum((xx-c2)**2)])

def MOOexponential(xx, c1=-0.5, c2=0.5):
    return np.array([1-np.exp(-np.sum((xx-c1)**2)), 1-np.exp(-np.sum((xx-c2)**2))])

def MOOackley(xx, c1=0.5, c2=-0.5, a=20, b=0.2, c=2*np.pi):
    return np.array([ackley(np.array([xx-c1])),ackley(np.array([xx-c2]))])

def MOOcrossit(xx, c1=0.5, c2=-0.5):
    return np.array([crossitNdim(xx-c1), crossitNdim(xx-c2)])
    
def MOOgramacy(x , c1=0.5, c2=-0.5):
  return np.array([gramacy(x-c1), gramacy(x-c2)])
