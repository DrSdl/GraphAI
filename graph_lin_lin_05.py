import numpy as np
import matplotlib.pyplot as plt
# necessaty for pure ssh connection
#plt.switch_backend('agg')
# ----------------------------
# 12052018: added experimental data point plotting

import matplotlib.image as mpimg
import random as rnd
from pathlib import Path
import h5py

# Either generate Training Data or Testing Data
# Change random seed and sample number accordingly
fhdf5 = h5py.File('GraphTrainData.hdf5', 'w')
ghdf5 = h5py.File('GraphTrainIds.hdf5', 'w')

#fhdf5 = h5py.File('GraphTestData.hdf5', 'w')
#ghdf5 = h5py.File('GraphTestIds.hdf5', 'w')

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


# do some sanity checks if expplot makes sense, i.e. slope of function not too small
def check4para(a,b,c):
    ret=0
    r1=a*b
    if (abs(a*b)>0.1) :
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

# write file with experimental data points
def AppendPoints(myname, xline, yline):
    my_file = Path('./%s' % myname)
    if my_file.is_file():
        f = open('./%s' % myname, 'a')
        f.write(str(xline))
        f.write("\n")
        f.write(str(yline))
        f.write("\n")
    else:
        f = open('./%s' % myname, 'w')
        f.write(str(xline))
        f.write("\n")
        f.write(str(yline))
        f.write("\n")
    f.close()

  
#import csv
#with open(<path to output_csv>, "wb") as csv_file:
#        writer = csv.writer(csv_file, delimiter=',')
#        for line in data:
#            writer.writerow(line)

rnd.seed(2018) #training
#rnd.seed(2006) #testing
N=200

# decide plot range
x0=0.01
x1=10.01
t1 = np.arange(x0, x1, (x1-x0)/100.0)

myfile="GraphAIrun.csv"
mypoints="GraphAIpnt.csv"
# 0: no experimental data points
# 1: experimentl data points are added
experi=1
style=['o','v','^','s','D','*','+','x']

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
    characterSimple=np.array([co])

    if co==1:
        a=rnd.uniform(-10,10)
        c=rnd.uniform(0,10)
        plt.axis((0,10,-10,10))
        plt.plot(t1, flin(t1,a,c), 'k')
        AppendData(myfile,k,co,a,c,0,0)
        character=np.array([co,a,c,0,0])
        if experi==1 :
            Npoints=np.random.randint(5,30)
            Nxlocat=10.0*np.random.sample(Npoints)
            Nylocat=flin(Nxlocat,a,c) + np.random.normal(0,2,Npoints)
            plt.plot(Nxlocat, Nylocat, rnd.sample(style,1)[0])
            AppendPoints(mypoints,Nxlocat,Nylocat)
    elif co==2:
        a=rnd.uniform(0,+10)
        b=rnd.uniform(-10,a-1)
        r=abs(40/((b-a)*(a-b)))
        c=rnd.uniform(-r,r)
        if abs(c)<0.1:
            c=0.1
        plt.axis((0,10,-10,10))
        plt.plot(t1, fquad(t1,a,b,c), 'k')
        AppendData(myfile,k,co,a,b,c,0)
        character=np.array([co,a,b,c,0])
        if experi==1 :
            Npoints=np.random.randint(5,30)
            Nxlocat=10.0*np.random.sample(Npoints)
            Nylocat=fquad(Nxlocat,a,b,c) + np.random.normal(0,2,Npoints)
            plt.plot(Nxlocat, Nylocat, rnd.sample(style,1)[0])
            AppendPoints(mypoints,Nxlocat,Nylocat)
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
        plt.plot(t1, fcubic(t1,a,b,c,d), 'k')
        AppendData(myfile,k,co,a,b,c,d)
        character=np.array([co,a,b,c,d])
        if experi==1 :
            Npoints=np.random.randint(5,30)
            Nxlocat=10.0*np.random.sample(Npoints)
            Nylocat=fcubic(Nxlocat,a,b,c,d) + np.random.normal(0,2,Npoints)
            plt.plot(Nxlocat, Nylocat, rnd.sample(style,1)[0])
            AppendPoints(mypoints,Nxlocat,Nylocat)
    elif co==4:
        a=rnd.uniform(-0.5,0.5)
        b=rnd.uniform(-1.0,1.0)
        c=rnd.uniform(-5,5)
        while check4para(a,b,c) == 0:
            a=rnd.uniform(-0.5,0.5)
            b=rnd.uniform(-1.0,1.0)
            c=rnd.uniform(-5,5)            
        plt.axis((0,10,-10,10))
        plt.plot(t1, fexp(t1,a,b,c), 'k')
        AppendData(myfile,k,co,a,b,c,0)
        character=np.array([co,a,b,c,0])  
        if experi==1 :
            Npoints=np.random.randint(5,30)
            Nxlocat=10.0*np.random.sample(Npoints)
            Nylocat=fexp(Nxlocat,a,b,c) + np.random.normal(0,2,Npoints)
            plt.plot(Nxlocat, Nylocat, rnd.sample(style,1)[0])
            AppendPoints(mypoints,Nxlocat,Nylocat)
    elif co==5:
        a=rnd.uniform(-5,5)
        c=rnd.uniform(-5,5)
        plt.axis((0,10,-10,10))
        #print(a,c)
        plt.plot(t1, flog(t1,a,c), 'k')
        AppendData(myfile,k,co,a,c,0,0)
        character=np.array([co,a,c,0,0]) 
        if experi==1 :
            Npoints=np.random.randint(5,30)
            Nxlocat=10.0*np.random.sample(Npoints)
            Nylocat=flog(Nxlocat,a,c) + np.random.normal(0,2,Npoints)
            plt.plot(Nxlocat, Nylocat, rnd.sample(style,1)[0])
            AppendPoints(mypoints,Nxlocat,Nylocat)

    #plt.show()
    plt.savefig('lin' + str(k) +'.jpg') # for debugging purposes
    # see: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    #      https://matplotlib.org/examples/pylab_examples/agg_buffer.html
    #      https://media.readthedocs.org/pdf/h5py/latest/h5py.pdf
    fig.canvas.draw()
    #print(fig.canvas.tostring_rgb())
    image_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # laternative data preparation
    #image_data = np.fromstring(fig.canvas.tostring_rgb())
    #ncols, nrows = fig.canvas.get_width_height()
    #image_data = np.fromstring(image_data, dtype=np.uint8).reshape(nrows, ncols, 3)
    image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    print(type(image_data)," ",image_data.shape," ",image_data[0,0,0], np.amin(image_data)," ",np.amax(image_data))
    name_data='lin' + str(k)
    fhdf5.create_dataset(name_data,data=image_data)
    # Write complete training data
    # ghdf5.create_dataset(name_data,data=character)
    # Write simple classification training data
    ghdf5.create_dataset(name_data,data=characterSimple)
    #plt.imshow(image_data)
    #plt.show()
    plt.clf()
    plt.close()

# print hdf5 file content
#fhdf5.keys()
#for name in fhdf5:
#    print(name)

#ghdf5.keys()
#for name in ghdf5:
#    print(name)

fhdf5.close()
ghdf5.close()

