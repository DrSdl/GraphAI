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
# Recover trained NN model 
# (c) 2019 DrSdl
# 
# ########################################################
# restore trained and saved tensorflow model
# 29092018: first version
# 30092018: batch loading of testing data to prevent memory overflow


import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils_06 import *
from sys import exit

np.random.seed(2018)

# get number of samples in hdf5 file
def GetTrainSamples(filename):
    training_dataset = h5py.File(filename, "r")
    liste=list(training_dataset.keys())
    anzahl=len(liste)
    return anzahl

# load small, fixed subset of figure image data from HDF5 file
# quick info: "http://docs.h5py.org/en/latest/quick.html"
def load_graph_dataset_small(filename, cnt, macro_length):
    training_dataset = h5py.File(filename, "r")
    liste=list(training_dataset.keys())
    anzahl=len(liste)
    Nm1=int(anzahl/macro_length)
    if cnt<Nm1:
        liste_sample=liste[cnt*macro_length:(cnt+1)*macro_length]
    else:
        liste_sample=liste[cnt*macro_length:anzahl]
    hx=0
    hy=0
    hd=0
    train_set=[]
    for myname in liste_sample:
        #print(myname)
        dats=np.array([training_dataset[myname]],dtype=np.uint8)
        #print(dats.shape," ",np.amin(dats)," ",np.amax(dats))
        if hx==0:
            hx=dats.shape[0]
            hy=dats.shape[1]
            hd=dats.shape[2]
            #train_set=np.empty((1,hx,hy,hd))
            train_set=dats
        else:
            train_set=np.concatenate((train_set,dats),axis=0)

    return train_set

# load small, fixed samnle of figure image characterisation from HDF5 file
# quick info: "http://docs.h5py.org/en/latest/quick.html"
def load_chara_dataset_small(filename, cnt, macro_length):
    training_dataset = h5py.File(filename, "r")
    liste=list(training_dataset.keys())
    anzahl=len(liste)
    Nm1=int(anzahl/macro_length)
    if cnt<Nm1:
        liste_sample=liste[cnt*macro_length:(cnt+1)*macro_length]
    else:
        liste_sample=liste[cnt*macro_length:anzahl]

    train_set=[]
    sum=0
    for myname in liste_sample:
        #print(myname)
        dats=np.array([training_dataset[myname]],dtype=np.int16)
        #print(dats.shape," ",np.amin(dats)," ",np.amax(dats))
        if sum==0:
            train_set=dats
            sum +=1
        else:
            train_set=np.concatenate((train_set,dats),axis=0)
  
    #print(train_set)
    return train_set

# number of batches simultaneously loaded into memory
m_macro=4

X_train_name = 'GraphTestData.hdf5'
Y_train_name = 'GraphTestIds.hdf5'
Nm=GetTrainSamples(Y_train_name);
print("magnitude of test set: ",Nm)

# Number of categories to be trained
NumCategories=6

tf.reset_default_graph()

predict=[]
correct=[]

with tf.Session() as sess:
    # restore trained model
    new_saver = tf.train.import_meta_graph('./model3L_01.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    # get graph
    saved_graph = tf.get_default_graph()

    # ###########################################################
    # print graph operations to identify names of variables to re-load
    ## for i in tf.get_default_graph().get_operations():
    ##     print(i.name)
    ## exit(0)
    # ###########################################################

    # so, where do we get the names "PlaceholderXXX" from? => By doing the above
    # print job first to see what has been stored...
    # get some tensors: input to network
    X = saved_graph.get_tensor_by_name('Placeholder:0')
    Y = saved_graph.get_tensor_by_name('Placeholder_1:0')
    # get accuracy tensor
    accuracy = saved_graph.get_tensor_by_name('Mean_1:0')
    # now we would also like to see which class the network predicts versus the true class
    # get raw output from forward propagation, last op from fully_connected layer
    Z3 = saved_graph.get_tensor_by_name('fully_connected/BiasAdd:0')

    num_minibatches = int(Nm / m_macro)

    test_accuracy = 0.0
    sum=0

    for i in range(num_minibatches):
        print("working on batch; ",i)
        # we want to load a batch of size "m_macro" sequentially from a larger dataset
        # into memory and see how well our trained TF graph can do the predictions
        X_test_orig = load_graph_dataset_small(X_train_name, i, m_macro)
        Y_test_orig = load_chara_dataset_small(Y_train_name, i, m_macro)
        Y_test_orig = Y_test_orig.T
        X_test = X_test_orig/255.
        Y_test = convert_to_one_hot(Y_test_orig, NumCategories).T

        # ##########################################################################
        # the TF "path" to "accuracy" lets us compare predictions with expected values
        temp_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        # ##########################################################################
        forward_prop  = tf.argmax(Z3,1) 
        # print("prediction: ", forward_prop.eval({X: minibatch_X}), " true value: ", tf.argmax(minibatch_Y,1).eval())
        print("case: ", sum, " ; result ", temp_accuracy)
        #predict.append(forward_prop.eval({X: X_test})[0])
        #correct.append(tf.argmax(Y_test,1).eval()[0])
        # ##########################################################################
        # we want to store "predict" and "correct" separately here in order to
        # determine the confusion matrix later 
        predict=np.append(predict,forward_prop.eval({X: X_test}))
        correct=np.append(correct,tf.argmax(Y_test,1).eval())
        test_accuracy += temp_accuracy / num_minibatches
        # ##########################################################################
        sum=sum+1
    print("Test Accuracy:", test_accuracy)


# initialize confusion matrix 
print("number of cases in test matrix:" ,len(correct))
cmatrix = confusion_matrix(y_true=correct, y_pred=predict)
print('this is the confusion matrix:')
print(cmatrix)

# plot confusion matrix 
plt.matshow(cmatrix)

plt.colorbar()
tick_marks = np.arange(NumCategories-1)
plt.xticks(tick_marks, range(NumCategories-1))
plt.yticks(tick_marks, range(NumCategories-1))
plt.xlabel('Predicted')
plt.ylabel('True')

plt.savefig("Confusion_matrix.svg")

# #####################################################
# original output of stored graphs names:
""" 

Placeholder
Placeholder_1
W1/Initializer/random_uniform/shape
W1/Initializer/random_uniform/min
W1/Initializer/random_uniform/max
W1/Initializer/random_uniform/RandomUniform
W1/Initializer/random_uniform/sub
W1/Initializer/random_uniform/mul
W1/Initializer/random_uniform
W1
W1/Assign
W1/read
W2/Initializer/random_uniform/shape
W2/Initializer/random_uniform/min
W2/Initializer/random_uniform/max
W2/Initializer/random_uniform/RandomUniform
W2/Initializer/random_uniform/sub
W2/Initializer/random_uniform/mul
W2/Initializer/random_uniform
W2
W2/Assign
W2/read
Conv2D
Relu
MaxPool
Conv2D_1
Relu_1
MaxPool_1
Flatten/Shape
Flatten/Slice/begin
Flatten/Slice/size
Flatten/Slice
Flatten/Slice_1/begin
Flatten/Slice_1/size
Flatten/Slice_1
Flatten/Const
Flatten/Prod
Flatten/ExpandDims/dim
Flatten/ExpandDims
Flatten/concat/axis
Flatten/concat
Flatten/Reshape
fully_connected/weights/Initializer/random_uniform/shape
fully_connected/weights/Initializer/random_uniform/min
fully_connected/weights/Initializer/random_uniform/max
fully_connected/weights/Initializer/random_uniform/RandomUniform
fully_connected/weights/Initializer/random_uniform/sub
fully_connected/weights/Initializer/random_uniform/mul
fully_connected/weights/Initializer/random_uniform
fully_connected/weights
fully_connected/weights/Assign
fully_connected/weights/read
fully_connected/biases/Initializer/Const
fully_connected/biases
fully_connected/biases/Assign
fully_connected/biases/read
fully_connected/MatMul
fully_connected/BiasAdd
Rank
Shape
Rank_1
Shape_1
Sub/y
Sub
Slice/begin
Slice/size
Slice
concat/values_0
concat/axis
concat
Reshape
Rank_2
Shape_2
Sub_1/y
Sub_1
Slice_1/begin
Slice_1/size
Slice_1
concat_1/values_0
concat_1/axis
concat_1
Reshape_1
SoftmaxCrossEntropyWithLogits
Sub_2/y
Sub_2
Slice_2/begin
Slice_2/size
Slice_2
Reshape_2
Const
Mean
gradients/Shape
gradients/Const
gradients/Fill
gradients/Mean_grad/Reshape/shape
gradients/Mean_grad/Reshape
gradients/Mean_grad/Shape
gradients/Mean_grad/Tile
gradients/Mean_grad/Shape_1
gradients/Mean_grad/Shape_2
gradients/Mean_grad/Const
gradients/Mean_grad/Prod
gradients/Mean_grad/Const_1
gradients/Mean_grad/Prod_1
gradients/Mean_grad/Maximum/y
gradients/Mean_grad/Maximum
gradients/Mean_grad/floordiv
gradients/Mean_grad/Cast
gradients/Mean_grad/truediv
gradients/Reshape_2_grad/Shape
gradients/Reshape_2_grad/Reshape
gradients/zeros_like
gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim
gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
gradients/SoftmaxCrossEntropyWithLogits_grad/mul
gradients/Reshape_grad/Shape
gradients/Reshape_grad/Reshape
gradients/fully_connected/BiasAdd_grad/BiasAddGrad
gradients/fully_connected/BiasAdd_grad/tuple/group_deps
gradients/fully_connected/BiasAdd_grad/tuple/control_dependency
gradients/fully_connected/BiasAdd_grad/tuple/control_dependency_1
gradients/fully_connected/MatMul_grad/MatMul
gradients/fully_connected/MatMul_grad/MatMul_1
gradients/fully_connected/MatMul_grad/tuple/group_deps
gradients/fully_connected/MatMul_grad/tuple/control_dependency
gradients/fully_connected/MatMul_grad/tuple/control_dependency_1
gradients/Flatten/Reshape_grad/Shape
gradients/Flatten/Reshape_grad/Reshape
gradients/MaxPool_1_grad/MaxPoolGrad
gradients/Relu_1_grad/ReluGrad
gradients/Conv2D_1_grad/Shape
gradients/Conv2D_1_grad/Conv2DBackpropInput
gradients/Conv2D_1_grad/Shape_1
gradients/Conv2D_1_grad/Conv2DBackpropFilter
gradients/Conv2D_1_grad/tuple/group_deps
gradients/Conv2D_1_grad/tuple/control_dependency
gradients/Conv2D_1_grad/tuple/control_dependency_1
gradients/MaxPool_grad/MaxPoolGrad
gradients/Relu_grad/ReluGrad
gradients/Conv2D_grad/Shape
gradients/Conv2D_grad/Conv2DBackpropInput
gradients/Conv2D_grad/Shape_1
gradients/Conv2D_grad/Conv2DBackpropFilter
gradients/Conv2D_grad/tuple/group_deps
gradients/Conv2D_grad/tuple/control_dependency
gradients/Conv2D_grad/tuple/control_dependency_1
beta1_power/initial_value
beta1_power
beta1_power/Assign
beta1_power/read
beta2_power/initial_value
beta2_power
beta2_power/Assign
beta2_power/read
W1/Adam/Initializer/Const
W1/Adam
W1/Adam/Assign
W1/Adam/read
W1/Adam_1/Initializer/Const
W1/Adam_1
W1/Adam_1/Assign
W1/Adam_1/read
W2/Adam/Initializer/Const
W2/Adam
W2/Adam/Assign
W2/Adam/read
W2/Adam_1/Initializer/Const
W2/Adam_1
W2/Adam_1/Assign
W2/Adam_1/read
fully_connected/weights/Adam/Initializer/Const
fully_connected/weights/Adam
fully_connected/weights/Adam/Assign
fully_connected/weights/Adam/read
fully_connected/weights/Adam_1/Initializer/Const
fully_connected/weights/Adam_1
fully_connected/weights/Adam_1/Assign
fully_connected/weights/Adam_1/read
fully_connected/biases/Adam/Initializer/Const
fully_connected/biases/Adam
fully_connected/biases/Adam/Assign
fully_connected/biases/Adam/read
fully_connected/biases/Adam_1/Initializer/Const
fully_connected/biases/Adam_1
fully_connected/biases/Adam_1/Assign
fully_connected/biases/Adam_1/read
Adam/learning_rate
Adam/beta1
Adam/beta2
Adam/epsilon
Adam/update_W1/ApplyAdam
Adam/update_W2/ApplyAdam
Adam/update_fully_connected/weights/ApplyAdam
Adam/update_fully_connected/biases/ApplyAdam
Adam/mul
Adam/Assign
Adam/mul_1
Adam/Assign_1
Adam
init
save/Const
save/SaveV2/tensor_names
save/SaveV2/shape_and_slices
save/SaveV2
save/control_dependency
save/RestoreV2/tensor_names
save/RestoreV2/shape_and_slices
save/RestoreV2
save/Assign
save/RestoreV2_1/tensor_names
save/RestoreV2_1/shape_and_slices
save/RestoreV2_1
save/Assign_1
save/RestoreV2_2/tensor_names
save/RestoreV2_2/shape_and_slices
save/RestoreV2_2
save/Assign_2
save/RestoreV2_3/tensor_names
save/RestoreV2_3/shape_and_slices
save/RestoreV2_3
save/Assign_3
save/RestoreV2_4/tensor_names
save/RestoreV2_4/shape_and_slices
save/RestoreV2_4
save/Assign_4
save/RestoreV2_5/tensor_names
save/RestoreV2_5/shape_and_slices
save/RestoreV2_5
save/Assign_5
save/RestoreV2_6/tensor_names
save/RestoreV2_6/shape_and_slices
save/RestoreV2_6
save/Assign_6
save/RestoreV2_7/tensor_names
save/RestoreV2_7/shape_and_slices
save/RestoreV2_7
save/Assign_7
save/RestoreV2_8/tensor_names
save/RestoreV2_8/shape_and_slices
save/RestoreV2_8
save/Assign_8
save/RestoreV2_9/tensor_names
save/RestoreV2_9/shape_and_slices
save/RestoreV2_9
save/Assign_9
save/RestoreV2_10/tensor_names
save/RestoreV2_10/shape_and_slices
save/RestoreV2_10
save/Assign_10
save/RestoreV2_11/tensor_names
save/RestoreV2_11/shape_and_slices
save/RestoreV2_11
save/Assign_11
save/RestoreV2_12/tensor_names
save/RestoreV2_12/shape_and_slices
save/RestoreV2_12
save/Assign_12
save/RestoreV2_13/tensor_names
save/RestoreV2_13/shape_and_slices
save/RestoreV2_13
save/Assign_13
save/restore_all
ArgMax/dimension
ArgMax
ArgMax_1/dimension
ArgMax_1
Equal
Cast_1
Const_1
Mean_1

"""
# #####################################################






