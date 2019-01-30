import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# necessaty for pure ssh connection
# plt.switch_backend('agg')
# ----------------------------
# ########################################################
# 
# Test model on real image: parameter estimate
# (c) 2019 DrSdl
# 
# ########################################################
# restore trained and saved tensorflow model
# load real image from a scientific journal

import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils_06 import *
from sys import exit

# real image
X_image_name = './test01a_640x480fx.jpg'
img_orig=plt.imread(X_image_name)
#print(img.shape)
#print(img)
plt.imshow(img_orig)
plt.axis('off')
plt.show()



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


tf.reset_default_graph()

img=img_orig/255.
img=np.array([img])
print(img.shape)

with tf.Session() as sess:
    # restore trained model
    new_saver = tf.train.import_meta_graph('./model3C_02/model3C_02.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./model3C_02'))
    # get graph
    saved_graph = tf.get_default_graph()

    # ###########################################################
    # get some tensors: input to network
    X = saved_graph.get_tensor_by_name('Placeholder:0')
    Y = saved_graph.get_tensor_by_name('Placeholder_1:0')
    # 
    Z3 = saved_graph.get_tensor_by_name('fully_connected/BiasAdd:0')
    #
    # get prediction of parameters
    #
    pre = Z3.eval({X: img})
    print('prediction for graph parameters: ',pre[0])


#
#plt.imshow(img_orig)
#plt.axis((0,10,-10,10))
#plt.axis('off')
#
x0=0.01
x1=10.01
t1 = np.arange(x0, x1, (x1-x0)/100.0)
plt.plot(t1, fquad(t1,pre[0][0],pre[0][2],pre[0][1]), 'blue')

plt.show()
