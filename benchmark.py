from math import sqrt


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