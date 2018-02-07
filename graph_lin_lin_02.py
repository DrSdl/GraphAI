import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from pathlib import Path

# -------------------------------------------
# The GraphAI project (c) 2017-2018 Dr. Sdl
# 070218 simple, constrained graph generation
# -------------------------------------------

# a linear function y=a*x+c
def flin(t,a,c):
    return a*(t-c)

# a quadratic function y=a*x^2+b*x+c
def fquad(t,a,b,c):
    return c*(t-a)*(t-b)

# a cubic function y=a*x^3+b*x^2+c*x+d
def fcubic(t,a,b,c,d):
    return d*(t-a)*(t-b)*(t-c)

# an exponential function
def fexp(t,a,b,c):
    return b*np.exp(a*t)+c

# a logarithmic function
def flog(t,a,b):
    return a*np.log(t)+b

# we want to enforce that at least one zeros of the quad function is in the plotting range
def check2para(a,b):
    ret=0
    if (a>0) or (b>0):
        ret=1
    return ret

# we want to enforce that at least two zeros of the cubic function are in the plotting range
def check3para(a,b,c):
    ret=0
    if (a>0 and b>0) or (a>0 and c>0) or(b>0 and c>0):
        ret=1
    return ret

# write file with graph target information
def AppendData(myname,id,co,a,b,c,d):
    my_file = Path('./%s' % myname)
    if my_file.is_file():
        f = open('./%s' % myname, 'a')
        f.write(str(id))
        f.write(", ")
        f.write(str(co))
        f.write(", ")
        f.write(str(a))
        f.write(", ")
        f.write(str(b))
        f.write(", ")
        f.write(str(c))
        f.write(", ")
        f.write(str(d))
        f.write("\n")
    else:
        f = open('./%s' % myname, 'w')
        f.write(str(id))
        f.write(", ")
        f.write(str(co))
        f.write(", ")
        f.write(str(a))
        f.write(", ")
        f.write(str(b))
        f.write(", ")
        f.write(str(c))
        f.write(", ")
        f.write(str(d))
        f.write("\n")
    f.close()
    
#import csv
#with open(<path to output_csv>, "wb") as csv_file:
#        writer = csv.writer(csv_file, delimiter=',')
#        for line in data:
#            writer.writerow(line)

rnd.seed(2018)
N=1000

# decide plot range
x0=0.01
x1=10.01
t1 = np.arange(x0, x1, (x1-x0)/100.0)

myfile="GraphAIrun.csv"

for k in range(0,N):
    
    print("case: ",k)
    # decide which function type to use -----
    # 1: lin
    # 2: quad
    # 3: cubic
    # 4: exp
    # 5: log
    # ---------------------------------------
    # in this version of the graph generator
    # we select graph parameters in such a way so
    # that curves fit into a given standard box.
    # This standard box is assumed to be [0,10]x[-10,10].
    co=rnd.randint(1,5)

    if co==1:
        a=rnd.uniform(-1,1)
        c=rnd.uniform(0,5)
        plt.axis((0,10,-10,10))
        plt.plot(t1, flin(t1,a,c), 'k',)
        AppendData(myfile,k,co,a,c,0,0)
    elif co==2:
        a=rnd.uniform(-5,+5)
        b=rnd.uniform(-a,8)
        while check2para(a,b) == 0:
            a=rnd.uniform(-5,+5)
            b=rnd.uniform(-a,8)
        c=rnd.uniform(-1,1)
        plt.axis((0,10,-10,10))
        plt.plot(t1, fquad(t1,a,b,c), 'k',)
        AppendData(myfile,k,co,a,b,c,0)
    elif co==3:
        a=rnd.uniform(-2,+5)
        b=rnd.uniform(-2,+8)
        c=rnd.uniform(+2,+8)
        while check3para(a,b,c) == 0:
            a=rnd.uniform(-2,+5)
            b=rnd.uniform(-2,+8)
            c=rnd.uniform(+2,+8)
        d=rnd.uniform(-0.3,0.3)
        plt.axis((0,10,-10,10))
        plt.plot(t1, fcubic(t1,a,b,c,d), 'k',)
        AppendData(myfile,k,co,a,b,c,d)
    elif co==4:
        while True:
            a=rnd.uniform(0.2,0.3)
            b=rnd.uniform(-1,1)
            if b<0:
                a=-1.0*a
                b=rnd.uniform(-5,5)
            else:
                b=rnd.uniform(-1,1)
            c=rnd.uniform(-2,2)
            x1=fexp(0,a,b,c)
            x2=fexp(10,a,b,c)
            if(np.abs(x2-x1)>3):
                break
        plt.axis((0,10,-10,10))
        plt.plot(t1, fexp(t1,a,b,c), 'k',)
        AppendData(myfile,k,co,a,b,c,0)
    elif co==5:
        a=rnd.uniform(-3,3)
        c=rnd.uniform(-3,3)
        plt.axis((0,10,-10,10))
        #print(a,c)
        plt.plot(t1, flog(t1,a,c), 'k',)
        AppendData(myfile,k,co,a,c,0,0)



    #plt.show()
    plt.savefig('lin' + str(k) +'.jpg')
    plt.clf()

