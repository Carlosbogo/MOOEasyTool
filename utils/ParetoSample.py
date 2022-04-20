import sobol_seq

from models.GaussianProcess import GaussianProcess
import numpy as np
import tensorflow as tf
from utils.calc_pareto import get_pareto_frontier, get_pareto_undominated_by, getSetfromFront

def generateLinspaceGrid(N: int, D: int):

    grid = [tf.linspace(0,1,N) for _ in range(D)]
    for o in range(D-1):
        grid = [ tf.tile(g,[N]) if idx<=o else tf.repeat(g,N) for idx,g in enumerate(grid)]

    grid = tf.convert_to_tensor(grid)
    
    comp = np.unique(np.array([grid[:,a] for a in range(grid.shape[1])]),axis=0).shape
    if not (comp[0]==N**D and comp[1]==D):
        raise Exception("Grid has not correct dimensions or return values")

    return grid

def getParetoFrontSamples(GP: GaussianProcess, N: int = 1_000, M: int = 5):

    Paretos = []
    xx = sobol_seq.i4_sobol_generate(GP.d,N)
    samples = GP.multiGPR.predict_f_samples(xx,M)
    maxs = tf.math.reduce_max(tf.math.reduce_max(samples, axis=0),axis=0)
    grid = np.transpose(generateLinspaceGrid(100,GP.O-1).numpy())
    for sample in samples:
        for o, gold_medal in enumerate(tf.gather(sample, tf.math.argmin(sample, axis=0), axis=0)):
            new_sample = grid*(np.append(maxs[:o], maxs[o+1:])-np.append(gold_medal[:o], gold_medal[o+1:]))+np.append(gold_medal[:o], gold_medal[o+1:])
            new_sample = np.insert(new_sample, o, gold_medal[o]*np.ones(new_sample.shape[0]), axis=1)
            sample = np.append(sample, new_sample, axis=0)
            
        pareto = get_pareto_undominated_by(sample)
        Paretos.append(pareto)

    return Paretos

def getParetoSetSamples(GP: GaussianProcess, N: int = 1_000, M: int = 5):

    Paretos = []
    xx = sobol_seq.i4_sobol_generate(GP.d,N)
    samples = GP.multiGPR.predict_f_samples(xx,M)
    for sample in samples:

        pareto_front = get_pareto_undominated_by(sample)
        pareto_set = getSetfromFront(xx,samples,pareto_front)
        Paretos.append(pareto_set)

    return Paretos

def getParetoSamples(GP: GaussianProcess, N: int = 1_000, M: int = 5):

    Pareto_fronts, Pareto_sets = [], []
    xx = sobol_seq.i4_sobol_generate(GP.d,N)
    samples = GP.multiGPR.predict_f_samples(xx,M)

    for sample in samples:
        pareto_front = get_pareto_undominated_by(sample.numpy())
        Pareto_fronts.append(pareto_front)

        pareto_set = getSetfromFront(xx,sample,pareto_front)
        Pareto_sets.append(pareto_set)

        # fig, axs = plt.subplots(1,2)
        # axs[0].plot(pareto_set[:,0],pareto_set[:,1],'ro', markersize=2)
        # axs[0].plot(xx[:,0],xx[:,1],'o', markersize=1)

        # axs[1].plot(pareto_front[:,0],pareto_front[:,1],'ro', markersize=2)
        # axs[1].plot(sample[:,0],sample[:,1],'o', markersize=1)

        # plt.show()
    return Pareto_sets, Pareto_fronts

