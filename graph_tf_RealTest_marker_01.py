import numpy as np
import matplotlib.pyplot as plt
# necessaty for pure ssh connection
# plt.switch_backend('agg')
# ################################################################
# 
# Real image test for the trained CASCADE OpenCV classifier
# (c) 2019 DrSdl
# 
# ################################################################

import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('test01_640x480.jpg',1)
#img = cv2.imread('test01tfx.jpg',1)
imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

point_cascade = cv2.CascadeClassifier('.\graphAI_pts_examples\data\cascade.xml')

points = point_cascade.detectMultiScale(imgg, 1.3, 5, maxSize=(30,30))
#
for (x,y,w,h) in points:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
