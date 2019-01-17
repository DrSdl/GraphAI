import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# necessaty for pure ssh connection
# 
plt.switch_backend('agg')
# ----------------------------
# ########################################################
# 
# Recover trained NN model for parameter prediction
# (c) 2019 DrSdl
# 
# ########################################################
# restore trained and saved tensorflow model for parameters
# 01012019: first version
# 


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

# load small, fixed sample subset of figure image characterisation from HDF5 file
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

# load small, fixed sample subset of figure image parametrisation from HDF5 file
# quick info: "http://docs.h5py.org/en/latest/quick.html"
def load_chara_dataset_param_small(filename, cnt, macro_length):
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
        dats=np.array([training_dataset[myname]],dtype=np.float)
        if sum==0:
            train_set=dats
            sum +=1
        else:
            train_set=np.concatenate((train_set,dats),axis=0)
  
    #print(train_set)
    return train_set



# number of batches simultaneously loaded into memory
m_macro=4

#X_train_name = 'GraphTestData_LIN.hdf5'
#Y_train_name = 'GraphTestIds_LIN.hdf5'

#X_train_name = 'GraphTestData_QUA.hdf5'
#Y_train_name = 'GraphTestIds_QUA.hdf5'

#X_train_name = 'GraphTestData_CUB.hdf5'
#Y_train_name = 'GraphTestIds_CUB.hdf5'

X_train_name = 'GraphTestData_EXP.hdf5'
Y_train_name = 'GraphTestIds_EXP.hdf5'

#X_train_name = 'GraphTestData_LOG.hdf5'
#Y_train_name = 'GraphTestIds_LOG.hdf5'



Nm=GetTrainSamples(Y_train_name);
print("magnitude of test set: ",Nm)

# ATTENTION: update parameters depending on LIN, QUAD etc. case
# Number of parameters to be trained

#NumParameters=2 # LIN
#NumParameters=3 # QUA
#NumParameters=4 # CUB
NumParameters=3 # EXP
#NumParameters=2 # EXP

tf.reset_default_graph()

predict=[]
correct=[]

with tf.Session() as sess:
    # restore trained model
    # ATTENTION:                                     |||
    new_saver = tf.train.import_meta_graph('/home/drsdl/GraphAI/model3C_04/model3C_04.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('/home/drsdl/GraphAI/model3C_04/'))

    # get graph
    saved_graph = tf.get_default_graph()

    # ###########################################################
    # print graph operations to identify names of variables to re-load
    ##for i in tf.get_default_graph().get_operations():
    ##    print(i.name)
    ##exit(0)
    # ###########################################################

    # so, where do we get the names "PlaceholderXXX" from? => By doing the above
    # print job first to see what has been stored...
    # get some tensors: input to network
    X = saved_graph.get_tensor_by_name('Placeholder:0')
    Y = saved_graph.get_tensor_by_name('Placeholder_1:0')
    # get accuracy tensor
    accuracy = saved_graph.get_tensor_by_name('truediv_1:0')
    # now we would also like to see which class the network predicts versus the true class
    # get raw output from forward propagation, last op from fully_connected layer
    Z3 = saved_graph.get_tensor_by_name('fully_connected/BiasAdd:0')
    print('retrieved tensor X: ', X)  # Tensor("Placeholder:0", shape=(?, 480, 640, 3), dtype=float32)
    print('retrieved tensor Z: ', Z3) # Tensor("fully_connected/BiasAdd:0", shape=(?, 2), dtype=float32)
    num_minibatches = int(Nm / m_macro)

    test_accuracy = 0.0
    sum=0

    for i in range(num_minibatches):
        print("working on batch; ",i)
        # we want to load a batch of size "m_macro" sequentially from a larger dataset
        # into memory and see how well our trained TF graph can do the predictions
        X_test_orig = load_graph_dataset_small(X_train_name, i, m_macro)
        Y_test_orig = load_chara_dataset_param_small(Y_train_name, i, m_macro)
        Y_test_orig = Y_test_orig.T
        X_test = X_test_orig/255.
        Y_test = convert_to_target(Y_test_orig, NumParameters).T
        
        # ##########################################################################
        # the TF "path" to "accuracy" lets us compare predictions with expected values
        temp_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        # ##########################################################################
        #
        # ##########################################################################
        # we want to store "predict" and "correct" separately here in order to
        # determine the confusion matrix later 
        # print(Z3.eval({X: X_test}))
        if i==0:
            predict = Z3.eval({X: X_test})
            correct = Y_test
        else:
            predict=np.vstack((predict,Z3.eval({X: X_test})))
            correct=np.vstack((correct,Y_test))
        test_accuracy += temp_accuracy / num_minibatches
        # ##########################################################################
        sum=sum+1
    print("Test Accuracy - root mean square error of parameter difference: ", test_accuracy)


debugg=0
if NumParameters == 2:
    # plot parameter prediction performance 
    plt.subplot(211)
    plt.plot(predict[:,0])   # Parameter 1 prediction
    plt.plot(correct[:,0])   # Parameter 1 correct value
    plt.subplot(212)
    plt.plot(predict[:,1])   # Parameter 2 prediction
    plt.plot(correct[:,1])   # Parameter 2 correct value


if NumParameters == 3:
    # plot parameter prediction performance
    plt.subplot(311)
    plt.plot(predict[:,0])   # Parameter 1 prediction
    plt.plot(correct[:,0])   # Parameter 1 correct value
    plt.subplot(312)
    plt.plot(predict[:,1])   # Parameter 2 prediction
    plt.plot(correct[:,1])   # Parameter 2 correct value
    plt.subplot(313)
    plt.plot(predict[:,2])   # Parameter 3 prediction
    plt.plot(correct[:,2])   # Parameter 3 correct value

if NumParameters == 4:
    # plot parameter prediction performance
    plt.subplot(411)
    plt.plot(predict[:,0])   # Parameter 1 prediction
    plt.plot(correct[:,0])   # Parameter 1 correct value
    plt.subplot(412)
    plt.plot(predict[:,1])   # Parameter 2 prediction
    plt.plot(correct[:,1])   # Parameter 2 correct value
    plt.subplot(413)
    plt.plot(predict[:,2])   # Parameter 3 prediction
    plt.plot(correct[:,2])   # Parameter 3 correct value
    plt.subplot(414)
    plt.plot(predict[:,3])   # Parameter 4 prediction
    plt.plot(correct[:,3])   # Parameter 4 correct value





if debugg==1:
    plt.show()
else:
    plt.savefig('graphAI_EXP_parameter_100119.fig1.jpeg')
    plt.clf()

# plot parameter prediction performance histogram ---------------------------------------
# https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib/36479941
if NumParameters == 2:
    plt.subplot(121)
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    x1=[-10,+10]
    y1=[-10,+10]
    x2=[0,+10]
    y2=[0,+10]

    plt.scatter(correct[:,0], predict[:,0])  # correct value versus predict value scatter plot
    plt.plot(x1,y1, color='black')
    plt.subplot(122)
    plt.scatter(correct[:,1], predict[:,1])
    plt.plot(x2,y2, color='black')

if NumParameters == 3:
    plt.subplot(221)
    plt.xlim(0,10)
    plt.ylim(0,10)
    x1=[0,+10]
    y1=[0,+10]
    x2=[0,+10]
    y2=[0,+10]
    x3=[-10,+10]
    y3=[-10,+10]
 
    plt.scatter(correct[:,0], predict[:,0])  
    plt.plot(x1,y1, color='black')
    plt.subplot(222)
    plt.scatter(correct[:,1], predict[:,1])
    plt.plot(x2,y2, color='black')
    plt.subplot(223)
    plt.scatter(correct[:,2], predict[:,2])
    plt.plot(x3,y3, color='black')

if NumParameters == 4:
    plt.subplot(221)
    plt.xlim(0,10)
    plt.ylim(0,10)
    x1=[-10,+10]
    y1=[-10,+10]
    x2=[0,+10]
    y2=[0,+10]
    x3=[-10,+10]
    y3=[-10,+10]
    x4=[-5,+5]
    y4=[-5,+5]

    plt.scatter(correct[:,0], predict[:,0])
    plt.plot(x1,y1, color='black')
    plt.subplot(222)
    plt.scatter(correct[:,1], predict[:,1])
    plt.plot(x2,y2, color='black')
    plt.subplot(223)
    plt.scatter(correct[:,2], predict[:,2])
    plt.plot(x3,y3, color='black')
    plt.subplot(224)
    plt.scatter(correct[:,3], predict[:,3])
    plt.plot(x4,y4, color='black')



if debugg==1:
    plt.show()
else:
    plt.savefig('graphAI_EXP_parameter_100119.fig2.jpeg')
    plt.clf()




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
Shape
strided_slice/stack
strided_slice/stack_1
strided_slice/stack_2
strided_slice
sub
Pow/y
Pow
Const
Sum
Cast
mul/x
mul
truediv
gradients/Shape
gradients/Const
gradients/Fill
gradients/truediv_grad/Shape
gradients/truediv_grad/Shape_1
gradients/truediv_grad/BroadcastGradientArgs
gradients/truediv_grad/RealDiv
gradients/truediv_grad/Sum
gradients/truediv_grad/Reshape
gradients/truediv_grad/Neg
gradients/truediv_grad/RealDiv_1
gradients/truediv_grad/RealDiv_2
gradients/truediv_grad/mul
gradients/truediv_grad/Sum_1
gradients/truediv_grad/Reshape_1
gradients/truediv_grad/tuple/group_deps
gradients/truediv_grad/tuple/control_dependency
gradients/truediv_grad/tuple/control_dependency_1
gradients/Sum_grad/Reshape/shape
gradients/Sum_grad/Reshape
gradients/Sum_grad/Shape
gradients/Sum_grad/Tile
gradients/mul_grad/Shape
gradients/mul_grad/Shape_1
gradients/mul_grad/BroadcastGradientArgs
gradients/mul_grad/mul
gradients/mul_grad/Sum
gradients/mul_grad/Reshape
gradients/mul_grad/mul_1
gradients/mul_grad/Sum_1
gradients/mul_grad/Reshape_1
gradients/mul_grad/tuple/group_deps
gradients/mul_grad/tuple/control_dependency
gradients/mul_grad/tuple/control_dependency_1
gradients/Pow_grad/Shape
gradients/Pow_grad/Shape_1
gradients/Pow_grad/BroadcastGradientArgs
gradients/Pow_grad/mul
gradients/Pow_grad/sub/y
gradients/Pow_grad/sub
gradients/Pow_grad/Pow
gradients/Pow_grad/mul_1
gradients/Pow_grad/Sum
gradients/Pow_grad/Reshape
gradients/Pow_grad/Greater/y
gradients/Pow_grad/Greater
gradients/Pow_grad/Log
gradients/Pow_grad/zeros_like
gradients/Pow_grad/Select
gradients/Pow_grad/mul_2
gradients/Pow_grad/mul_3
gradients/Pow_grad/Sum_1
gradients/Pow_grad/Reshape_1
gradients/Pow_grad/tuple/group_deps
gradients/Pow_grad/tuple/control_dependency
gradients/Pow_grad/tuple/control_dependency_1
gradients/sub_grad/Shape
gradients/sub_grad/Shape_1
gradients/sub_grad/BroadcastGradientArgs
gradients/sub_grad/Sum
gradients/sub_grad/Reshape
gradients/sub_grad/Sum_1
gradients/sub_grad/Neg
gradients/sub_grad/Reshape_1
gradients/sub_grad/tuple/group_deps
gradients/sub_grad/tuple/control_dependency
gradients/sub_grad/tuple/control_dependency_1
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
Shape_1
strided_slice_1/stack
strided_slice_1/stack_1
strided_slice_1/stack_2
strided_slice_1
sub_1
Pow_1/y
Pow_1
Const_1
Sum_1
Cast_1
mul_1/x
mul_1
truediv_1

"""
# #####################################################






