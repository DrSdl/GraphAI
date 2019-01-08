import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import random as zufall

# ########################################################
# 
# Helper functions for GraphAI
# (c) 2019 DrSdl
# 
# ########################################################


# adapted from "https://github.com/JudasDie/deeplearning.ai/blob/master/Convolutional%20Neural%20Networks/week1/cnn_utils.py"
# 140418: created function "load_graph_dataset"
#         created function "load_chara_dataset"
# 12052018: added microbatch reading from harddisk instead from memory... needed for very large training sets

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# load figure image data from HDF5 file
def load_graph_dataset(filename):
    training_dataset = h5py.File(filename, "r")
    #train_set=np.empty()
    hx=0
    hy=0
    hd=0
    train_set=[]
    for myname in training_dataset:
        print(myname)
        dats=np.array([training_dataset[myname]],dtype=np.uint8)
        print(dats.shape," ",np.amin(dats)," ",np.amax(dats))
        if hx==0:
            hx=dats.shape[0]
            hy=dats.shape[1]
            hd=dats.shape[2]
            #train_set=np.empty((1,hx,hy,hd))
            train_set=dats
        else:
            train_set=np.concatenate((train_set,dats),axis=0)

    return train_set

# load random subset of figure image data from HDF5 file
# quick info: "http://docs.h5py.org/en/latest/quick.html"
def load_graph_dataset_batch(filename, seed, macro_length):
    training_dataset = h5py.File(filename, "r")
    liste=list(training_dataset.keys())
    anzahl=len(liste)
    zufall.seed(seed)
    liste_sample=zufall.sample(liste,macro_length)
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

# load figure image characterisation from HDF5 file
def load_chara_dataset(filename):
    training_dataset = h5py.File(filename, "r")
    train_set=[]
    sum=0
    for myname in training_dataset:
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

# load figure image function parameters from HDF5 file
def load_chara_dataset_param(filename):
    training_dataset = h5py.File(filename, "r")
    train_set=[]
    sum=0
    for myname in training_dataset:
        #print(myname)
        dats=np.array([training_dataset[myname]],dtype=np.float)
        #print(dats.shape," ",np.amin(dats)," ",np.amax(dats))
        if sum==0:
            train_set=dats
            sum +=1
        else:
            train_set=np.concatenate((train_set,dats),axis=0)
  
    #print(train_set)
    return train_set

# load random samnle of figure image characterisation from HDF5 file
# quick info: "http://docs.h5py.org/en/latest/quick.html"
def load_chara_dataset_batch(filename, seed, macro_length):
    training_dataset = h5py.File(filename, "r")
    liste=list(training_dataset.keys())
    anzahl=len(liste)
    zufall.seed(seed)
    liste_sample=zufall.sample(liste,macro_length)

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

# load figure image function parameters from HDF5 file - batchwise
def load_chara_dataset_param_batch(filename, seed, macro_length):
    training_dataset = h5py.File(filename, "r")
    liste=list(training_dataset.keys())
    anzahl=len(liste)
    zufall.seed(seed)
    liste_sample=zufall.sample(liste,macro_length)
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



# create mini-batches 
# first the arrays are shuffled randomly and then packaged into batches of "mini_batch_size"
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- make results repeatable
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# do a one-hot encoding of a category vector
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# extract the target parameters of the functions from HDF5 store
def convert_to_target(Y, C):
    Y = Y[1:C+1,:]
    return Y



def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

# quick info: https://stackoverflow.com/questions/35336648/list-of-tensor-names-in-graph-in-tensorflow
# function to get names of Tensors in a graph (defaults to default graph):
def get_names(graph=tf.get_default_graph()):
    return [t.name for op in graph.get_operations() for t in op.values()]

# Function to get Tensors in a graph (defaults to default graph):
def get_tensors(graph=tf.get_default_graph()):
    return [t for op in graph.get_operations() for t in op.values()]
