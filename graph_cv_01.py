import numpy as np
import matplotlib.pyplot as plt
# necessaty for pure ssh connection
# plt.switch_backend('agg')
# ################################################################
# 
# Simple Test procedure for the trained CASCADE OpenCV classifier
# (c) 2019 DrSdl
# 
# ################################################################

import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('.\graphAI_pts_examples\graphAI_21.jpg',1)
imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

point_cascade = cv2.CascadeClassifier('.\graphAI_pts_examples\data\cascade.xml')

points = point_cascade.detectMultiScale(imgg, 1.3, 5, maxSize=(30,30))

for (x,y,w,h) in points:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ##------------------------------------------------------------------------------------------------------------------------------------------------------
# the following training commands were used:
# G:\OpenCV\opencv_345\build\x64\vc15\bin\opencv_createsamples.exe   -info .\GraphAIpnttest_ALL_pts_simple.csv -vec samples.vec -w 20 -h 20
# G:\OpenCV\opencv_345\build\x64\vc15\bin\opencv_traincascade.exe -data data -vec samples.vec -bg bg.dat -numPos 900 -numNe 80 -numStages=2 -w 30 -h 30
# G:\OpenCV\opencv_345\build\x64\vc15\bin\opencv_createsamples.exe   -vec samples.vec -w 20 -h 20
# ##------------------------------------------------------------------------------------------------------------------------------------------------------

# next: we want to check the total performance: i.e. testing and training performance
# we open "GraphAIpnttest_ALL_pts_simple.csv" which contains the image location and the "true" point data

# determine the minimum distance between measured and true point position
def MiniDist(xp,yp,points):
    # len(points)<=len(xp)
    mins=[]
    for (x,y,w,h) in points:
        cont=[]
        for k in range(len(xp)):
            xt=xp[k]
            yt=yp[k]
            d=np.sqrt(np.power((xt-x),2)+np.power((yt-y),2))
            cont.append(d)
        mins.append(np.min(cont))
    return mins

mypath='.\\graphAI_pts_examples\\'
tfile='GraphAIpnttest_ALL_pts_simple.csv'
fileout = mypath + tfile

PointsT=[]  # true number of points
PointsI=[]  # detected number of points
PointsD=[]  # Distance between detected and true points

with open(fileout) as f:
    for line in f:
        #print(line)
        li=line.split()
        graph=li[0]
        npoints=int(li[1])
        #print(graph,';',npoints)
        xp=[]
        yp=[]
        for k in range(npoints):
            xp.append(int(li[2+4*k]))
            yp.append(int(li[3+4*k]))
        #print(xp,";",yp)
        img = cv2.imread(mypath+graph,1)
        imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        points = point_cascade.detectMultiScale(imgg, 1.3, 5, maxSize=(30,30))
        #print(k,';',len(points))
        PointsT.append(npoints)
        PointsI.append(len(points))
        mins=MiniDist(xp,yp,points)
        PointsD.append(mins)

PointsD_flat = [val for sublist in PointsD for val in sublist] 

plt.subplot(211)
plt.xlim((0,30))
plt.ylim((0,30))
plt.plot(PointsT, PointsI, 'bo')

plt.subplot(212)
#plt.xlim((0,30))
#plt.ylim((0,30))
#plt.plot(PointsT, PointsI, 'bo')
plt.hist(PointsD_flat, 50, facecolor='b', alpha=1.00)
plt.show()


