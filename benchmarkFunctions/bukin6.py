import numpy as np

bukin6_description = {
    "desciption": "2D function with many local minima in a row",
    "problem": "Algorithm stuck in local minima",
    "dimensions": 2,
    "x*": [-10,1],
    "f(x*)": 0
}


def bukin6(xx):
    if len(xx)!=2:
        ValueError('Bukin6 function must have dimension 2, dimension',len(xx),'was found.')

    x1 = xx[0]
    x2 = xx[1]

    term1 = 100 * np.sqrt(np.abs(x2 - 0.01*(x1**2)))
    term2 = 0.01 * np.abs(x1+10)

    return term1 +term2