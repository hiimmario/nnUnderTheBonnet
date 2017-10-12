import numpy as np
import pickle
import nn

import matplotlib.pyplot

def unpickle(file):
    # Load byte data from file
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        return data


def load_cifar10_data(data_dir):
    # Return train_data, train_labels, test_data, test_labels
    # The shape of data is 32 x 32 x3'''
    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


data_dir = "C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/cifar10/batches/"
train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)

# plot a test picture from te cifar 10 dataset after unpacking the batches
# matplotlib.pyplot.imshow(train_data[0])
# matplotlib.pyplot.show()

# number of input, hidden and output nodes
input_nodes = 3072
hidden_nodes = 1021
output_nodes = 10

# learning rate
learning_rate = 0.15

# create instance of neural network
n = nn.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


inputs = []

# n.load_model("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/cifar10/who_1.json",
#              "C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/cifar10/wih_1.json")


epochs = 5

# preparing the rgb picture 32*32*3 for the input - 1d vector with 0.01-1 floating point number
for index_epochs, e in enumerate(range(epochs)):
    for index, p in enumerate(train_data):
        for r in p:
            for i in r:
                for rec in i:
                    rec = (rec / 255.0 * 0.99) + 0.01
                    inputs.append(rec)

        targets = np.zeros(output_nodes) + 0.01
        targets[train_labels[index]] = 0.99
        n.train(inputs, targets)
        print(str(index) + " ... label: " + str(train_labels[index]) + " trained!")
        inputs = []

        if index == 1050:
            break

    # store the model
    n.store_model("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/cifar10/who_" + str(index_epochs) + ".json",
                  "C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/cifar10/wih_" + str(index_epochs) + ".json")


scorecard = []
for index_epochs, e in enumerate(range(epochs)):
    # load model from json file
    n.load_model("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/cifar10/who_" + str(index_epochs) + ".json",
                 "C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/cifar10/wih_" + str(index_epochs) + ".json")

    for index, p in enumerate(test_data):
        for r in p:
            for i in r:
                for e in i:
                    e = (e / 255.0 * 0.99) + 0.01
                    inputs.append(e)

        correct_label = test_labels[index]
        result = n.query(inputs)
        label = np.argmax(result)

        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
            print(index, "match!")
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            print(index)


        inputs = []

        if index == 100:
            break

    scorecard_array = np.asarray(scorecard)
    print("performance model " + str(index_epochs) + ": " + str(scorecard_array.sum() / scorecard_array.size))
