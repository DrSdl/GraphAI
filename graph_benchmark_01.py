import numpy as np
import matplotlib.pyplot as plt
# necessaty for pure ssh connection
# plt.switch_backend('agg')
# ########################################################
# 
# GENERATOR for the benchmarking data
# (c) 2019 DrSdl
# 
# ########################################################
import matplotlib.image as mpimg
import random as rnd
from scipy.optimize import curve_fit

########################################################
# generate 2000 benchmark images in the form
# of 50 (x,y) datapoints
# and run a least-squares minimisation algo to identify
# parameters from it; assume some gaussian noise
# to simulate a researcher extracting the data
# manually from a scanned graph
########################################################

########################################################
# generate benchmark data
# 100
#fhdf5 = h5py.File('GraphTestData_ALL_pts_simple.hdf5', 'w')
#ghdf5 = h5py.File('GraphTestIds_ALL_pts_simple.hdf5', 'w')
#myfile="GraphAItest_ALL_pts_simple.csv"
#mypoints="GraphAIpnttest_ALL_pts_simple.csv"
#
rnd.seed(2006) #testing
N=2000   # number of benchmark graphs
########################################################


# -------------------------------------------
# The GraphAI project (c) 2017-2019 Dr. Sdl
# 170119 benchmark generation
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

# do some sanity checks if quadratic plot makes sense, i.e. extremum of function is within the plot range
def check2para(a,b,c):
    ret=0
    r=(a+b)/2.0
    v=c*(r-a)*(r-b)

    if abs(v<10):
        ret=1

    return ret

# do some sanity checks if cubic plot makes sense, i.e. extrema of function are within the plot range
def check3para(a,b,c,d):
    ret=0
    r1=(a+b+c-np.sqrt(a*a-a*b+b*b-a*c-b*c+c*c))/3.0
    r2=(a+b+c+np.sqrt(a*a-a*b+b*b-a*c-b*c+c*c))/3.0
    v1=d*(r1-a)*(r1-b)*(r1-c)
    v2=d*(r2-a)*(r2-b)*(r2-c)

    if (abs(v1)<10) and (abs(v2)<10) :
        ret=1
    if b<0:
        ret=0
    return ret


# do some sanity checks if expplot makes sense, i.e. slope of function not too small at zero crossing
def check4para(a,b,c):
    ret=0
    r1=a*b
    if (abs(a*b)>0.3) :
        ret=1
    return ret

# create the y-data for the function and add Gaussian noise
def MakeBench(x_set, flin, sigm, ranges):
    n=len(x_set)
    noise=np.random.normal(0.0,sigm,n)
    y_set=flin+noise
    x_s=[]
    y_s=[]
    #
    for i in range(n):
        x=x_set[i]
        y=y_set[i]
        if y>ranges[2] and y<ranges[3]:
            x_s.append(x)
            y_s.append(y)
    #
    return np.array(x_s), np.array(y_s)

# #################################################################
# decide plot range
x0=0.01
x1=10.01
NN=50 # number of points per graph
t1 = np.arange(x0, x1, (x1-x0)/NN) # create x points
Params=[]   # array holding the true parameters per function
PCov1=[]    # array to hold the covariance for lin fits
PCov2=[]    # array to hold the covariance for qua fits
PCov3=[]    # array to hold the covariance for cub fits
PCov4=[]    # array to hold the covariance for exp fits
PCov5=[]    # array to hold the covariance for log fits
sigm=0.3    # sigma of noise from simulated data extraction
confuse=np.zeros((5,5))
# #################################################################

for k in range(0,N):
    
    print("case: ",k)
    fig = plt.figure()

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
    #co=1 # for debugging purposes
    #
    #######################################################


    # create the graph data
    if co==1:
        a=rnd.uniform(-10,10)
        c=rnd.uniform(0,10)   
        plt.axis((0,10,-10,10))
        #plt.plot(t1, flin(t1,a,c), 'k')
        x_set, y_set = MakeBench(t1, flin(t1,a,c), sigm, [0,10,-10,+10])
        #plt.plot(x_set,y_set,'o')
        Params.append([co,a,c,0,0])

    elif co==2:
        a=rnd.uniform(0,+10)
        b=rnd.uniform(-10,a-1)
        r=abs(40/((b-a)*(a-b)))
        c=rnd.uniform(-r,r)
        if abs(c)<0.1:
            c=0.1
        plt.axis((0,10,-10,10))
        x_set, y_set = MakeBench(t1, fquad(t1,a,b,c), sigm, [0,10,-10,+10])
        #plt.plot(x_set,y_set,'o')
        Params.append([co,a,b,c,0])

    elif co==3:
        a=rnd.uniform(0,+10)
        b=rnd.uniform(-1,a)
        c=rnd.uniform(-5,b)
        d=rnd.uniform(-2,2)
        while check3para(a,b,c,d) == 0:
            a=rnd.uniform(0,+10)
            b=rnd.uniform(-8,+a)
            c=rnd.uniform(-10,b)
            d=rnd.uniform(-2,2)
        plt.axis((0,10,-10,10))
        x_set, y_set = MakeBench(t1, fcubic(t1,a,b,c,d), sigm, [0,10,-10,+10])
        #plt.plot(x_set,y_set,'o')
        Params.append([co,a,b,c,d])

    elif co==4:
        a=rnd.uniform(-0.5,0.5)
        b=rnd.uniform(-1.0,1.0)
        c=rnd.uniform(-5,5)
        while check4para(a,b,c) == 0:
            a=rnd.uniform(-0.5,0.5)
            b=rnd.uniform(-1.0,1.0)
            c=rnd.uniform(-5,5)            
        plt.axis((0,10,-10,10))
        x_set, y_set = MakeBench(t1, fexp(t1,a,b,c), sigm, [0,10,-10,+10])
        #plt.plot(x_set,y_set,'o')
        Params.append([co,a,b,c,0])  

    elif co==5:
        a=rnd.uniform(-5,5)
        c=rnd.uniform(-5,5)
        plt.axis((0,10,-10,10))
        x_set, y_set = MakeBench(t1, flog(t1,a,c), sigm, [0,10,-10,+10])
        #plt.plot(x_set,y_set,'o')
        Params.append([co,a,c,0,0]) 

    # plt.show()
    # now start the least square fit for all potential function types
    #
    # https://docs.scipy.org/doc/scipy/reference/optimize.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit
    # https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting
    # 
    # the try/except procedure is necessary because sometimes the least-squares fit does
    # not converge - but we need to let the loop continue without breaking
    #
    # sorry, not very nice code, will be improved later
    # ##############################################################################
    # find the covariance data from parameter fitting ##############################        
    #
    if co==1:
        try:
            popt, pcov = curve_fit(flin, x_set, y_set)       # 2 paranmeters
        except:
            pcov=np.identity(2)
            popt=np.array([0,0])
        pcov=np.abs(pcov)
        perr = np.sqrt(np.diag(pcov))
        if np.mean(perr)>1:              # reduce impact of unreasonable fit
            perr=np.array([1,1])
        PCov1.append(perr)
    elif co==2:
        try:
            popt, pcov = curve_fit(fquad, x_set, y_set)      # 3 parameters
        except:
            pcov=np.identity(3)
            popt=np.array([0,0,0])
        pcov=np.abs(pcov)
        perr = np.sqrt(np.diag(pcov))
        if np.mean(perr)>1:              # reduce impact of unreasonable fit
            perr=np.array([1,1,1])
        PCov2.append(perr)
    elif co==3:
        try:
            popt, pcov = curve_fit(fcubic, x_set, y_set)     # 4 parameters
        except:
            pcov=np.identity(4)
            popt=np.array([0,0,0,0])
        pcov=np.abs(pcov)
        perr = np.sqrt(np.diag(pcov))
        if np.mean(perr)>1:              # reduce impact of unreasonable fit
            perr=np.array([1,1,1,1])
        PCov3.append(perr)
    elif co==4:
        try:
            popt, pcov = curve_fit(fexp, x_set, y_set)       # 3 parameters
        except:
            pcov=np.identity(3)
            popt=np.array([0,0,0])
        pcov=np.abs(pcov)
        perr = np.sqrt(np.diag(pcov))
        if np.mean(perr)>1:              # reduce impact of unreasonable fit
            perr=np.array([1,1,1])
        PCov4.append(perr)
    elif co==5:
        try:
            popt, pcov = curve_fit(flog, x_set, y_set)       # 2 parameters
        except:
            pcov=np.identity(2)
            popt=np.array([0,0])
        pcov=np.abs(pcov)
        perr = np.sqrt(np.diag(pcov))
        if np.mean(perr)>1:              # reduce impact of unreasonable fit
            perr=np.array([1,1])
        PCov5.append(perr)
    #print('-----------------------------------------------------------')
    #print(perr)
    #print(popt, " :: ", Params[k])
    #print('-----------------------------------------------------------')
    # ##############################################################################
    # find the confusion matrix ####################################################
    #
    #
    pgues=[0,0,0,0,0]
    #
    try:
        popt, pcov = curve_fit(flin, x_set, y_set)
        pcov=np.abs(pcov)
        pcov=np.mean(np.sqrt(np.diag(pcov)))
        pgues[0]=pcov
    except:
        pgues[0]=1000.0
    try: 
        popt, pcov = curve_fit(fquad, x_set, y_set) 
        pcov=np.abs(pcov)
        pcov=np.mean(np.sqrt(np.diag(pcov)))
        pgues[1]=pcov
    except:
        pgues[1]=1000.0
    try:
        popt, pcov = curve_fit(fcubic, x_set, y_set)
        pcov=np.abs(pcov)
        pcov=np.mean(np.sqrt(np.diag(pcov)))
        pgues[2]=pcov
    except:
        pgues[2]=1000.0 
    try:
        popt, pcov = curve_fit(fexp, x_set, y_set)
        pcov=np.abs(pco)
        pcov=np.mean(np.sqrt(np.diag(pcov)))
        pgues[3]=pcov
    except:
        pgues[3]=1000.0 
    try:
        popt, pcov = curve_fit(flog, x_set, y_set) 
        pcov=np.abs(pco)
        pcov=np.mean(np.sqrt(np.diag(pcov)))
        pgues[4]=pcov
    except:
        pgues[4]=1000.0
    #
    #print(pgues)
    pgues=np.array(pgues)
    coguess=np.argmin(pgues)+1
    #print('true type: ', co, 'estimated type: ', coguess)
    confuse[co-1,coguess-1] = confuse[co-1,coguess-1] +1
    

#print(PCov1)
PCov1=np.array(PCov1)
PCov1=np.mean(PCov1,axis=0)
PCov2=np.array(PCov2)
PCov2=np.mean(PCov2,axis=0)
PCov3=np.array(PCov3)
PCov3=np.mean(PCov3,axis=0)
PCov4=np.array(PCov4)
PCov4=np.mean(PCov4,axis=0)
PCov5=np.array(PCov5)
PCov5=np.mean(PCov5,axis=0)

print('confusion matrix: ', confuse)

print('mean std for lin parameters: ',PCov1)
print('mean std for qua parameters: ',PCov2)
print('mean std for cub parameters: ',PCov3)
print('mean std for exp parameters: ',PCov4)
print('mean std for log parameters: ',PCov5)
