import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# -------------------------------------------
# The GraphAI project (c) 2017-2018 Dr. Sdl
# 070218 unconstrained graph generation
# -------------------------------------------

# a linear function y=a*x+c
def flin(t,a,c):
    return a*t+c

# a quadratic function y=a*x^2+b*x+c
def fquad(t,a,b,c):
    return a*t*t+b*t+c

# a cubic function y=a*x^3+b*x^2+c*x+d
def fcubic(t,a,b,c,d):
    return a*t*t*t+b*t*t+c*t+d

# an exponential function
def fexp(t,a,b,c):
    return b*np.exp(a*t)+c

# a logarithmic function
def flog(t,a,b):
    return a*np.log(t)+b


rnd.seed(2018)
N=100

for k in range(0,N):
    
    # decide plot range
    x0=rnd.randint(-100,-1)
    x1=rnd.randint(0,100)
    t1 = np.arange(x0, x1, (x1-x0)/50.0)

    # decide which function type to use -----
    # 1: lin
    # 2: quad
    # 3: cubic
    # 4: exp
    # 5: log
    # ---------------------------------------
    co=rnd.randint(1,5)

    if co==1:
        a=rnd.randint(-100,100)
        c=rnd.randint(-100,100)
        plt.plot(t1, flin(t1,a,c), 'k',)
    elif co==2:
        a=rnd.randint(-100,100)
        b=rnd.randint(-100,100)
        c=rnd.randint(-100,100)
        plt.plot(t1, fquad(t1,a,b,c), 'k',)
    elif co==3:
        a=rnd.randint(-100,100)
        b=rnd.randint(-100,100)
        c=rnd.randint(-100,100)
        d=rnd.randint(-100,100)
        plt.plot(t1, fcubic(t1,a,b,c,d), 'k',)
    elif co==4:
        x0=rnd.randint(-10,-1)
        x1=rnd.randint(0,10)
        t1 = np.arange(x0, x1, (x1-x0)/50.0)
        a=rnd.randint(-5,5)
        b=rnd.randint(-10,10)
        c=rnd.randint(-10,10)
        plt.plot(t1, fexp(t1,a,b,c), 'k',)
    elif co==5:
        t1 = np.arange(0.1, x1, (x1+0.1)/50.0)
        a=rnd.randint(-100,100)
        c=rnd.randint(-100,100)
        plt.plot(t1, flog(t1,a,c), 'k',)



    #plt.show()
    plt.savefig('lin' + str(k) +'.jpg')
    plt.clf()

