import numpy as np

ackley_description = {
    "desciption": "Multidimensional function with many local minima",
    "problem": "Algorithm stuck in local minima",
    "dimensions": -1,
    "input_range": [-32.768, 32.768],
    "x*": [0,0],
    "f(x*)": 0
}

def ackley(xx, a=20, b=0.2, c=2*np.pi):

    d = len(xx)

    sum1 = np.sum(xx**2)
    sum2 = np.sum(np.cos(c*xx))

    term1 = -a * np.exp(-b*np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)

    return term1 + term2 + a + np.exp(1)