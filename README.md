# nnUnderTheBonnet
simple 3 layer neural network implementation in python based on the book "make your own neural network" from tariq rashid with some extended functionality and performance tests

mnist:

original dataset: 
http://yann.lecun.com/exdb/mnist/

short description: 
The MNIST database of handwritten digits, available from this page, 
has a training set of 60,000 examples, and a test set of 10,000 examples. 
It is a subset of a larger set available from NIST. The digits have been 
size-normalized and centered in a fixed-size image.
It is a good database for people who want to try learning 
techniques and pattern recognition methods on real-world data while 
spending minimal efforts on preprocessing and formatting.

performance: ~93% accuracy



cifar10:

original dataset:
https://www.cs.toronto.edu/~kriz/cifar.html

short description:

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. 
The test batch contains exactly 1000 randomly-selected images from each class. 
The training batches contain the remaining images in random order, but some training batches may 
contain more images from one class than another. 
Between them, the training batches contain exactly 5000 images from each class. 

performance: ~25%
at least it is double as good as randomly choosing a category ^^
this very simple neural net is already struggling with this problem.
ill use this dataset to get my first steps in tensorflow/keras done, where i hopefully will
achieve a much better performance



bottles:

original dataset: N/A

short description:

a friend of mine is blogging about whiskey market analysis and is implementing a feature with image recognition.
read more about it here: https://www.whiskystats.net/

performance: ~15%
the pictures are 200*300 pixels and are much more complex (100k input nodes) then the cifar10 dataset (3072 input nodes).
it was my first try after doing tests on the mnist dataset, 
but i had no clue how much harder this is going to be for this simple nural net.
after working with cifar10, i already knew that this tests will not be a huge success. but as everything was prepared, 
i ran the performance tests to get simply an idea of processing time and so on.
