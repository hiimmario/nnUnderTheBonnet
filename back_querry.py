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

# load model from mnist max performance
n.load_model("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/mnist/who_1.json",
             "C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/mnist/wih_1.json")

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

