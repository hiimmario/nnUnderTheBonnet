import numpy
import nn

# to plot the output of the neural net
import matplotlib.pyplot

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 196
output_nodes = 10

# learning rate
learning_rate = 0.2

# create instance of neural network
n = nn.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/mnist/mnist_dataset/mnist_train_100.csv",
                          'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 3

for index_epochs, e in enumerate(range(epochs)):
    # go through all records in the training data set
    for index_records, record in enumerate(training_data_list):
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        if index_records == 50000:
            break

        pass
    pass

# run the network backwards, given a label, see what image it produces

# label to test
label = 0
# create the output signals for this label
targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] is the target label for this record
targets[label] = 0.99

# get image data
image_data = n.backquery(targets)

# plot image data
matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

# label to test
label = 3
# create the output signals for this label
targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] is the target label for this record
targets[label] = 0.99

# get image data
image_data = n.backquery(targets)

# plot image data
matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

# label to test
label = 7
# create the output signals for this label
targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] is the target label for this record
targets[label] = 0.99

# get image data
image_data = n.backquery(targets)

# plot image data
matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
