# GraphAI
A graph database for machine learning

The main challenge of training neural networks is to have enough data! There are many codes which implement deep networks, (for example: tensorflow, keras, Caffe2, Apache MXNet) but not so many codes which generate data... 

Tensorflow: https://www.tensorflow.org/
Keras: https://keras.io/
Caffe2: https://caffe2.ai/
Apache MXNet: https://mxnet.apache.org/

The objective of this project is to automatically analyze printed graphs in scientific journals: is it a linear graph or does it show an exponential function (i.e. classification), does it contain single datapoints and estimate the parameters of these functions.

But first steps first. We have to have a repository of graphs for training: mostly 2D graphs found in scientific publications and then extract from them a number of features: length of x- and y-axis, dimension of x- and y-axis, degree of polynomial or spline interpolation to reproduce the data points shown. The problem: there would be some copyright issues if I would simply copy and paste existing graphs from scientific journals and distribute them as a training dataset. Moreover much work would be needed to label these graphs...

Hence this repository firstly recreates graphs synthetically using Matplotlib and other science tools.

In the second step we will try out a number of deep learning architectures to do the above mentioned classification and parameter extraction.



