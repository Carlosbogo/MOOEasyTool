# # Main Notebook
# 
# File to perform experiments

# ## Imports
import os
import numpy as np
np.set_printoptions(formatter={'float': '{: 2.3f}'.format})
import gpflow
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

from models.SOGP import GaussianProcess
from acquisition_functions.SingleObjective import SingleObjectiveAcq, SingleObjectiveAcqGrid

from acquisition_functions.SingleObjective import random, pi, ei, ucb, mes,   \
                simulated_mes_bins, simulated_mes_correlation, simulated_mes_covariance, \
                simulated_mes_distances_correlation, simulated_mes_distances_covariance, \
                simulated_mes_distancecorrelation,   simulated_mes_distancecovariance, \
                simulated_mes_spearmanr_correlation, simulated_mes_mic_e, simulated_mes_mic_approx
from benchmarkFunctions.eggholder import eggholder
from benchmarkFunctions.ackley import ackley

### Argparser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rootfolder", help="folder to save all", type=str, default="SingleObjectiveExperiments")
parser.add_argument("--acq", help="select acq function", type=str, default="pi")
parser.add_argument("--test", help="select test function", type=str, default="exponential")
parser.add_argument("--d", help="select input dimension of test function", type=int, default=2)
parser.add_argument("--M", help="fix number of samples", type=int, default=-1)
parser.add_argument("--device", help="GPU to use (0 or 1)", type=str, default="1")
parser.add_argument("--total-iter", help="Number of evaluations to perform", type=int, default=38)
parser.add_argument("--verbose", help="How much information to show", type=int, default=0)


args = parser.parse_args()

def acq_multiplex(acq):
    if acq=="random":
        return random
    if acq=="pi":
        return pi
    if acq=="ei":
        return ei
    if acq=="ucb":
        return ucb
    if acq=="mes":
        return mes
    if acq=="simulated_mes_bins":
        return simulated_mes_bins
    if acq=="simulated_mes_correlation":
        return simulated_mes_correlation
    if acq=="simulated_mes_covariance":
        return simulated_mes_covariance
    if acq=="simulated_mes_distances_correlation":
        return simulated_mes_distances_correlation
    if acq=="simulated_mes_distances_covariance":
        return simulated_mes_distances_covariance
    if acq=="simulated_mes_distancecorrelation":
        return simulated_mes_distancecorrelation
    if acq=="simulated_mes_distancecovariance":
        return simulated_mes_distancecovariance
    if acq=="simulated_mes_spearmanr_correlation":
        return simulated_mes_spearmanr_correlation
    if acq=="simulated_mes_mic_e":
        return simulated_mes_mic_e
    if acq=="simulated_mes_mic_approx":
        return simulated_mes_mic_approx
    return None

# ## Algorithm Arguments
seed = 1
np.random.seed(seed)
if args.test == "exponential":
    total_iter = 23
    initial_iter = 2

    lower_bound = -2
    upper_bound = 2
if args.test == "ackley":
    total_iter = args.total_iter
    initial_iter = 2

    lower_bound = -2
    upper_bound = 2
    
    opt_arg, opt_val = np.array([0, 0]), 0
    
if args.test == "eggholder":
    total_iter = 98
    initial_iter = 2

    lower_bound = -512
    upper_bound = 512


# ## Evaluation
d = args.d

def evaluation(x):
    if args.test == "exponential":
        return np.array([1-np.exp(-np.sum(x**2))])
    if args.test == "eggholder":
        return eggholder(x)
    if args.test == "ackley":
        return np.array([ackley(x)])


def random_acq(GP):
    while True:
        x_rand = np.random.uniform(GP.lowerBound, GP.upperBound, GP.d)
        if GP.X is None or not x_rand in GP.X:
                break
    return x_rand


# ## N experiments
root_folder = args.rootfolder
testF = args.test +str(args.d)
acqF = args.acq
if args.M>0:
    acqF = acqF+str(args.M)
function = acq_multiplex(args.acq)
print(root_folder+"/"+testF+"/"+acqF+".csv")
print("Global minum", opt_val, "in", opt_arg)


df = None
n_experiments = 20
with tf.device('/GPU:'+args.device):
    for i in range(n_experiments):
        print(i, time.ctime())

        ### GPs Initialization
        GP = GaussianProcess(d, lower_bound, upper_bound, noise_variance=2e-6)

        #### Initial samples, at least 1
        for l in range(initial_iter):
            ## Get random evaluation point
            x_rand = random_acq(GP)

            ## EVALUATION OF THE OUTSIDE FUNCTION
            y_rand = evaluation(x_rand)
            GP.addSample(x_rand,y_rand)

        GP.updateGP()
        GP.optimizeKernel()
        if False:
            GP.plotSamples()

        row = {
            'exp_id' : i,
            'testF' : testF,
            'acqF': acqF,
            'time': 0,
            'ns' : len(GP.X),
            'x'  : x_rand,
            'y'  : y_rand,
            'acq': 0
        }
        metrics = GP.evaluateOptimum(opt_val)
        row.update(metrics)
        if df is None:
            df = pd.DataFrame({k: [v] for k, v in row.items()})
        else:
            df = pd.concat([df, pd.DataFrame({k: [v] for k, v in row.items()})])        

        for l in range(total_iter):

            ## Search of the best acquisition function
            start = time.time()
            if "simulated_mes" in args.acq:
                x_best, acq = function(GP, M = args.M)
            else:
                x_best, acq = SingleObjectiveAcqGrid(function, GP)
            end = time.time()

            ## EVALUATION OF THE OUTSIDE FUNCTION
            y_best = evaluation(x_best)
            
            #print("  ", x_best, "->", y_best, "----", acq, "<-->", function(GP,np.array([x_best])), "   in", float(end-start))
            #print("  ", l, time.ctime())
            if args.verbose>0:
                print("  ", x_best, "->", y_best, "----", "{:.5f}".format(acq.numpy()), "<-->"," in", "{:.2f}".format(float(end-start)))
            #print(GP.X, GP.Y)
            
            ## UPDATE
            GP.addSample(x_best,y_best)     ## Add new sample to the model
            GP.updateGP()                   ## Update data on the GP regressor
            GP.optimizeKernel()             ## Optimize kernel hyperparameters

            ## Evaluate Pareto (distances and hypervolumes)
            row = {
                'exp_id' : i,
                'testF' : testF,
                'acqF': acqF,
                'time': float(end-start),
                'ns' : len(GP.X),
                'x'  : x_best,
                'y'  : y_best,
                'acq': acq
            }
            metrics = GP.evaluateOptimum(opt_val)
            row.update(metrics)

            df = pd.concat([df, pd.DataFrame({k: [v] for k, v in row.items()})])


    df.to_csv(root_folder+"/"+testF+"/"+acqF+".csv")

