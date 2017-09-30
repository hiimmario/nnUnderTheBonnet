import nn
import numpy

# scipy.ndimage for rotating image arrays
import scipy.ndimage

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 196
output_nodes = 10

# learning rate
learning_rate = 0.2

# create instance of neural network
n = nn.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/image_data/mnist_dataset/mnist_train.csv",
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

        ## create rotated variations
        # rotated anticlockwise by x degrees
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1,
                                                              reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        # rotated clockwise by x degrees
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1,
                                                               reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)

        if index_records == 50000:
            break

        pass

    # store the model
    n.store_model("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/mnist/who_" + str(index_epochs) + ".json",
                  "C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/mnist/wih_" + str(index_epochs) + ".json")

    pass

# load the mnist test data CSV file into a list
test_data_file = open("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/image_data/mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural net with all stored models
for index_epochs, e in enumerate(range(epochs)):

    # scorecard for how well the network performs
    scorecard = []

    # load model from json file
    n.load_model("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/mnist/who_" + str(index_epochs) + ".json",
                 "C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/mnist/wih_" + str(index_epochs) + ".json")

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass

        pass

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print("performance model " + str(index_epochs) + ": " + str(scorecard_array.sum() / scorecard_array.size))

    pass
