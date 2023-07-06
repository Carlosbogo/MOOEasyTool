import numpy as np
from benchmarkFunctions.ackley import ackley
from benchmarkFunctions.crossit import crossitNdim
from benchmarkFunctions.gramacy import gramacy
from benchmarkFunctions.NNHeart import fit_nn_heart_precision, fit_nn_heart_recall, fit_nn_heart, fit_logreg_heart
from benchmarkFunctions.NNHeartNeurons import fit_nn_heart_neurons
from benchmarkFunctions.NNHeartNeuronsF1 import fit_nn_heart_neurons_f1
from benchmarkFunctions.NNHeartNeuronsProb import fit_nn_heart_neurons_prob
from benchmarkFunctions.NNHeartNeuronsProbF1 import fit_nn_heart_neurons_prob_f1
from benchmarkFunctions.NNhouses_regression import fit_nn_houses_regression
from benchmarkFunctions.NNhouses_regression_prob import fit_nn_houses_regression_prob
from benchmarkFunctions.NNhouses_regression_mixed import fit_nn_houses_regression_mixed
from benchmarkFunctions.NNhouses_regression_mixed_prob import fit_nn_houses_regression_mixed_prob

def MOOquadratic(xx, c1=-0.5, c2=0.5):
    return np.array([np.sum((xx-c1)**2),np.sum((xx-c2)**2)])

def MOOackley(xx, c1=0.5, c2=-0.5, a=20, b=0.2, c=2*np.pi):
    return np.array([ackley(np.array([xx-c1])),ackley(np.array([xx-c2]))])

def MOOexponential_ackley(xx, c1=-0.5, c2=0.5):
    return np.array([1-np.exp(-np.sum((xx-c1)**2)), ackley(np.array([xx-c2]))])

def MOOquadratic_ackley(xx, c1=-0.5, c2=0.5):
    return np.array([np.sum((xx-c1)**2),ackley(np.array([xx-c2]))])

def MOOexponential(xx, c1=-0.5, c2=0.5):
    return np.array([1-np.exp(-np.sum((xx-c1)**2)), 1-np.exp(-np.sum((xx-c2)**2))])

def MOOcrossit(xx, c1=0.5, c2=-0.5):
    return np.array([crossitNdim(xx-c1), crossitNdim(xx-c2)])
    
def MOOgramacy(x, c1=0.5, c2=-0.5):
    return np.array([gramacy(x-c1), gramacy(x-c2)])

def MOOnnHeart_precision(C):
    return fit_nn_heart_precision(C)

def MOOnnHeart_recall(C):
    return fit_nn_heart_recall(C)

def MOOnnHeart(C):
    return fit_nn_heart(C)

def MOOnnHeartNeurons(C):
    return fit_nn_heart_neurons(C)

def MOOnnHeartNeuronsF1(C):
    return fit_nn_heart_neurons_f1(C)

def MOOnnHeartNeuronsProb(C):
    return fit_nn_heart_neurons_prob(C)

def MOOnnHeartNeuronsProbF1(C):
    return fit_nn_heart_neurons_prob_f1(C)

def MOOnnHeartLogReg(C):
    return fit_logreg_heart(C)

def MOOnnHousesRegression(C):
    return fit_nn_houses_regression(C)

def MOOnnHousesRegressionProb(C):
    return fit_nn_houses_regression_prob(C)

def MOOnnHousesRegressionMixed(C):
    return fit_nn_houses_regression_mixed(C)

def MOOnnHousesRegressionMixedProb(C):
    return fit_nn_houses_regression_mixed_prob(C)
