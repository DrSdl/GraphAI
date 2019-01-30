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
# Test model on real image: categorization
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
X_image_name = './test01a_640x480.jpg'
img=plt.imread(X_image_name)
#print(img.shape)
#print(img)
plt.imshow(img)
plt.axis('off')
plt.show()

tf.reset_default_graph()

img=img/255.
img=np.array([img])
print(img.shape)

with tf.Session() as sess:
    # restore trained model
    new_saver = tf.train.import_meta_graph('./model3L_01/model3L_01.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./model3L_01'))
    # get graph
    saved_graph = tf.get_default_graph()

    # ###########################################################
    # get some tensors: input to network
    X = saved_graph.get_tensor_by_name('Placeholder:0')
    Y = saved_graph.get_tensor_by_name('Placeholder_1:0')
    # 
    Z3 = saved_graph.get_tensor_by_name('fully_connected/BiasAdd:0')
    #
    forward_prop  = tf.argmax(Z3,1) 
    #
    print('prediction for graph type: ',forward_prop.eval({X: img}))


#




