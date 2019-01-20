import numpy as np
import matplotlib.pyplot as plt
# necessaty for pure ssh connection
#
plt.switch_backend('agg')
# ########################################################
# 
# GENERATOR for all the training and testing graphs
# 12052018: added experimental data point plotting
# 20012019: added write routine for OpenCV training
# (c) 2019 DrSdl
# 
# ########################################################
import matplotlib.image as mpimg
import random as rnd
from pathlib import Path
import h5py

# Either generate Training Data or Testing Data
# Change random seed and sample number accordingly
########################################################
# generate Training Data init
#fhdf5 = h5py.File('GraphTrainData_LIN.hdf5', 'w')
#ghdf5 = h5py.File('GraphTrainIds_LIN.hdf5', 'w')
#myfile="GraphAIrun_LIN.csv"
#mypoints="GraphAIpnt_LIN.csv"
#rnd.seed(2018) #training
#N=20000 #N=20000 # number of graph images
########################################################

########################################################
# generate Testing Data init

# 100
fhdf5 = h5py.File('GraphTestData_ALL_pts_simple.hdf5', 'w')
ghdf5 = h5py.File('GraphTestIds_ALL_pts_simple.hdf5', 'w')
myfile="GraphAItest_ALL_pts_simple.csv"
mypoints="GraphAIpnttest_ALL_pts_simple.csv"

# 2000
#fhdf5 = h5py.File('GraphTestData_ALL_pts.hdf5', 'w')
#ghdf5 = h5py.File('GraphTestIds_ALL_pts.hdf5', 'w')
#myfile="GraphAItest_ALL_pts.csv"
#mypoints="GraphAIpnttest_ALL_pts.csv"

# 20000
#fhdf5 = h5py.File('GraphTrainData_ALL_pts.hdf5', 'w')
#ghdf5 = h5py.File('GraphTrainIds_ALL_pts.hdf5', 'w')
#myfile="GraphAItrain_ALL_pts.csv"
#mypoints="GraphAIpnttrain_ALL_pts.csv"

#
rnd.seed(2006) #testing
N=100 #N=100, 2000, 20000  # number of graph images
########################################################


# -------------------------------------------
# The GraphAI project (c) 2017-2019 Dr. Sdl
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


# do some sanity checks if expplot makes sense, i.e. slope of function not too small at zero crossing
def check4para(a,b,c):
    ret=0
    r1=a*b
    if (abs(a*b)>0.3) :
        ret=1
    return ret

##################################################
# write file with graph target information
# the ID data consists of:
# id: number of graph
# co: type of graph
# a,b,c,d: parameters of function
##################################################
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

##################################################
# write file with experimental data points
# the data for experimental points consists of:
##################################################
def AppendPoints(myname, xline, yline, fname):
    my_file = Path('./%s' % myname)
    # convert graph position into pixel position
    # and only print points which fall into the graph boundary
    X1=0.0
    Y1=-10.0
    X2=10.0
    Y2=10.0
    #
    if my_file.is_file():
        f = open('./%s' % myname, 'a')
    else:
        f = open('./%s' % myname, 'w')

    n=len(xline)
    f.write(fname+ ' ')
    #
    sum=0
    # count visible number of points
    for i in range(n):
        if xline[i]>X1 and xline[i]<X2 and yline[i]>Y1 and yline[i]<Y2:
            sum +=1
    #
    f.write(str(sum)+ ' ')
    #
    # experimental points are usually 10x10 ; we go back (-5,-5) and assume a 20x20 frame for training
    for i in range(n):
        if xline[i]>X1 and xline[i]<X2 and yline[i]>Y1 and yline[i]<Y2:
            # the specific numbers used correspond to the pixel corner positions of the image
            f.write(str(int(xline[i]*(575.0-80.0)/10.0+80.0 - 5))+ ' '+ str(int(-1.0*yline[i]*(428.0-60.0)/20.0+(428.0+60.0)*0.5 - 5))+ ' 20 20 ' )  
    #
    f.write("\n")
    #f.write(str(xline)+ '\n')
    f.close()

  
#import csv
#with open(<path to output_csv>, "wb") as csv_file:
#        writer = csv.writer(csv_file, delimiter=',')
#        for line in data:
#            writer.writerow(line)


# decide plot range
x0=0.01
x1=10.01
t1 = np.arange(x0, x1, (x1-x0)/100.0)

###########################################################
# Switch for experimental data points
# 0: no experimental data points
# 1: experimentl data points are added
experi=0
style=['o','v','^','s','D','*','+','x'] # data point markers
###########################################################


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
    #######################################################
    # in simple training scenarios we only want to classify
    # the graph type => "characterSimple" array
    # in a more complete approach we want to identify
    # graph type and graph parameters simultaneously => "character" array
    characterSimple=np.array([co])
    #NNtype=0 # categorisation of graphs
    NNtype=1 # extracting parameters from graphs
    # attention: swap characterSimple with character at EOF!
    #######################################################

    funame = 'graphAI_' + str(k) +'.jpg'

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
            AppendPoints(mypoints,Nxlocat,Nylocat,funame)
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
            AppendPoints(mypoints,Nxlocat,Nylocat,funame)
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
            AppendPoints(mypoints,Nxlocat,Nylocat,funame)
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
            AppendPoints(mypoints,Nxlocat,Nylocat,funame)
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
            AppendPoints(mypoints,Nxlocat,Nylocat,funame)

    # plt.show()
    # 
    plt.savefig(funame) # for debugging purposes save figure
    # see: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    #      https://matplotlib.org/examples/pylab_examples/agg_buffer.html
    #      https://media.readthedocs.org/pdf/h5py/latest/h5py.pdf
    fig.canvas.draw()
    #print(fig.canvas.tostring_rgb())
    image_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # aternative data preparation
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
    if NNtype==0:
        ghdf5.create_dataset(name_data,data=characterSimple)
    if NNtype==1:
        ghdf5.create_dataset(name_data,data=character)
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

