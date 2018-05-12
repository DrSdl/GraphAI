import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

# necessaty for pure ssh connection
# plt.switch_backend('agg')
# ----------------------------

import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils_06 import *
from sys import exit

np.random.seed(2018)

# adapted from "https://github.com/JudasDie/deeplearning.ai/blob/master/Convolutional%20Neural%20Networks/week1/Convolution%2Bmodel%2B-%2BApplication%2B-%2Bv1.ipynb"

# create variables that contain the image and the classification data
# 20042018: added batch processing for training and test accuracy evaluation
# 12052018: added microbatch reading from harddisk instead from memory... needed for very large training sets

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    
    ## Placeholder creation for tensorflow
    ## Inputs:
    ## n_H0 -- scalar, height of an input image
    ## n_W0 -- scalar, width of an input image
    ## n_C0 -- scalar, number of channels of the input
    ## n_y -- scalar, number of classes
        
    ## Outputs:
    ## X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    ## Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y

def initialize_parameters():

    ## Initialize weight parameters
    ## Outputs:
    ## parameters -- a dictionary of tensors containing W1, W2

    tf.set_random_seed(2018)                              
    
    ## W1 and W2 are the "filters": filter_height, filter_width, in_channels, out_channels

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    ## attemtion: "Id" has to be set to the number of classes we want to identify
    parameters = {"W1": W1,
                  "W2": W2,
                  "Id": 6}
    
    return parameters

# Implements a three-layer ConvNet in Tensorflow:
# CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
def forward_propagation_model3L_01(X, parameters):
    
    ## Default forward propagation model:
    ## CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    ## Inputs:
    ## X -- input dataset placeholder, of shape (input size, number of examples)
    ## parameters -- python dictionary containing your parameters "W1", "W2"
    ##               the shapes are given in initialize_parameters

    ## Outputs:
    ## Z3 -- the output of the last LINEAR unit
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    N1 = parameters['Id']
    
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')
    # FLATTEN
    P = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function 
    # 6=N1 neurons in output layer. 
    Z3 = tf.contrib.layers.fully_connected(P, N1, activation_fn=None)

    return Z3

def compute_cost(Z3, Y):
    
    ## Compute optimisation target
    ## Arguments:
    ## Z3 -- output of forward propagation 
    ## Y -- "true" labels vector placeholder, same shape as Z3
    
    ## Outputs:
    ## cost - Tensor of the cost function

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
   
    return cost

# get number of samples in hdf5 file
def GetTrainSamples(filename):
    training_dataset = h5py.File(filename, "r")
    liste=list(training_dataset.keys())
    anzahl=len(liste)
    return anzahl

# naive model loading all the training and testing data in to memory
def model3L(X_train_name, Y_train_name, X_test, Y_test, learning_rate=0.009,
          num_epochs=10, minibatch_size=4, print_cost=True):
    
    ## uses forward_propagation_model3L_01
    
    ## Arguments:
    ## X_train -- training set, of shape (None, 480, 640, 3)
    ## Y_train -- training set labels, of shape (None, n_y)
    ## X_test -- test set, of shape (None, 480, 640, 3)
    ## Y_test -- test set labels, of shape (None, n_y)
    ## learning_rate -- learning rate of the optimization
    ## num_epochs -- number of epochs of the optimization loop
    ## minibatch_size -- size of a minibatch
    ## print_cost -- True to print the cost every 100 epochs
    
    ## Outputs:
    ## train_accuracy -- real number, accuracy on the train set (X_train)
    ## test_accuracy -- real number, testing accuracy on the test set (X_test)
    ## parameters -- parameters learnt by the model. They can then be used to predict.
    
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(2018)                          # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    (mt,n_H0t,n_W0t,n_C0t) = X_test.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation_model3L_01(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation:AdamOptimizer minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # tensorflow graph starts calculation
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        #print("Test Accuracy:", test_accuracy)

        # Too big for memory...
        #train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        #print("Train Accuracy:", train_accuracy)

        train_accuracy = 0.0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            temp_accuracy = accuracy.eval({X: minibatch_X, Y: minibatch_Y})

            train_accuracy += temp_accuracy / num_minibatches

        print("Train Accuracy:", train_accuracy)
    
        minibatches_test = random_mini_batches(X_test, Y_test, minibatch_size, 2018)
        num_minibatches_test = int(mt / minibatch_size)

        test_accuracy = 0.0
        for minibatch in minibatches_test:
            (minibatch_X, minibatch_Y) = minibatch
            temp_accuracy = accuracy.eval({X: minibatch_X, Y: minibatch_Y})

            test_accuracy += temp_accuracy / num_minibatches_test

        print("Test Accuracy:", test_accuracy)


        
        return train_accuracy, test_accuracy, parameters

# model which loads a large training data set step by step into memory, i.e. for each minibatch
def model3L_batchFile(X_train_name, Y_train_name, X_test, Y_test, learning_rate=0.009,
          num_epochs=10, minibatch_size=4, print_cost=True):
    
    ## uses forward_propagation_model3L_01
    
    ## Arguments:
    ## X_train_name -- name of hdf5 file containing training set, of shape (None, 480, 640, 3)
    ## Y_train_name -- name of hdf5 file containing training set labels, of shape (None, n_y)
    ## is is assumed that a macro-batch (i.e. batch loaded into memory is exact 200=n_y units big)
    ## X_test -- testing set, of shape (None, 480, 640, 3)
    ## Y_test -- testing set labels, of shape (None, n_y)
    ## learning_rate -- learning rate of the optimization
    ## num_epochs -- number of epochs of the optimization loop
    ## minibatch_size -- size of a minibatch
    ## print_cost -- True to print the cost every 100 epochs
    
    ## Outputs:
    ## train_accuracy -- real number, accuracy on the train set (X_train)
    ## test_accuracy -- real number, testing accuracy on the test set (X_test)
    ## parameters -- parameters learnt by the model. They can then be used to predict.
    
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(2018)                          # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    # (m, n_H0, n_W0, n_C0) = X_train.shape
    m=GetTrainSamples(X_train_name) # get total number of training samples in hdf5 file
    print("Number of training cases: ",m)
    m_macro=100
    print("Number of macro training cases: ",m_macro)
    print("Number of micro training cases: ",minibatch_size)

    (mt,n_H0t,n_W0t,n_C0t) = X_test.shape             
    n_y = Y_test.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0t, n_W0t, n_C0t, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation_model3L_01(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation:AdamOptimizer minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # tensorflow graph starts calculation
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            seed = seed + 1

            X_train_orig = load_graph_dataset_batch(X_train_name, seed, m_macro)
            Y_train_orig = load_chara_dataset_batch(Y_train_name, seed, m_macro)
            Y_train_orig = Y_train_orig.T
            X_train = X_train_orig/255.
            Y_train = convert_to_one_hot(Y_train_orig, NumCategories).T

            minibatch_cost = 0.
            num_minibatches = int(m_macro / minibatch_size) # number of minibatches of size minibatch_size in the train set
   
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        #print("Test Accuracy:", test_accuracy)

        # Too big for memory...
        #train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        #print("Train Accuracy:", train_accuracy)

        train_accuracy = 0.0
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            temp_accuracy = accuracy.eval({X: minibatch_X, Y: minibatch_Y})

            train_accuracy += temp_accuracy / num_minibatches

        print("Train Accuracy:", train_accuracy)
    
        minibatches_test = random_mini_batches(X_test, Y_test, minibatch_size, 2018)
        num_minibatches_test = int(mt / minibatch_size)

        test_accuracy = 0.0
        for minibatch in minibatches_test:
            (minibatch_X, minibatch_Y) = minibatch
            temp_accuracy = accuracy.eval({X: minibatch_X, Y: minibatch_Y})

            test_accuracy += temp_accuracy / num_minibatches_test

        print("Test Accuracy:", test_accuracy)


        
        return train_accuracy, test_accuracy, parameters




#print("Loading training images")
#X_train_orig = load_graph_dataset('GraphTrainData.hdf5')
#print("Loading training labels")
#Y_train_orig = load_chara_dataset('GraphTrainIds.hdf5')
#Y_train_orig = Y_train_orig.T
#print("Train sample properties: ",index)
#print(type(X_train_orig)," ",X_train_orig.shape)
#print(type(X_train_orig[index])," ",X_train_orig[index].shape)
#print(type(Y_train_orig)," ",Y_train_orig.shape)
#X_train = X_train_orig/255.
#Y_train = convert_to_one_hot(Y_train_orig, NumCategories).T


# Loading the testing data 
print("Loading testing images")
X_test_orig = load_graph_dataset('GraphTestData.hdf5')
print("Loading testing labels")
Y_test_orig = load_chara_dataset('GraphTestIds.hdf5')
Y_test_orig = Y_test_orig.T

index=4
print(Y_test_orig)
print(type(X_test_orig)," ",X_test_orig.shape)
print(type(X_test_orig[index])," ",X_test_orig[index].shape)
print(X_test_orig[index])
print(type(Y_test_orig)," ",Y_test_orig.shape)
#exit(0)

# Number of categories to be trained
NumCategories=6

X_test = X_test_orig/255.
Y_test = convert_to_one_hot(Y_test_orig, NumCategories).T

conv_layers = {}
X, Y = create_placeholders(480, 640, 3, 6)

tf.reset_default_graph()

# start network optimisation --------------------------------------------------
_, _, parameters = model3L_batchFile('GraphTrainData.hdf5', 'GraphTrainIds.hdf5', X_test, Y_test)


