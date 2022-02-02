import numpy as np

crossit_description = {
    "desciption": "2D function with many local minima in a row",
    "problem": "Algorithm stuck in local minima",
    "dimensions": 2,
    "x*": [[1.3491, -1.3491], [1.3491, 1.3491], [-1.3491, 1.3491], [-1.3491, -1.3491]],
    "f(x*)": -2.06261
}

def crossit(xx):
    if len(xx)!=2:
        ValueError('Bukin6 function must have dimension 2, dimension',len(xx),'was found.')

    x1 = xx[0]
    x2 = xx[1]

    fact1 = np.sin(x1)*np.sin(x2)
    fact2 = np.exp(abs(100 - np.sqrt(x1**2+x2**2)/np.pi))

    return -0.0001 * (abs(fact1*fact2)+1)**0.1
