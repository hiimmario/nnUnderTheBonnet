import nn
import numpy
# scipy.misc for reading the image
import scipy.misc
# glob to pick up pictures from folder
import glob
# datetime to measure the time
from datetime import datetime

# number of input, hidden and output nodes
input_nodes = 100000
hidden_nodes = 2000
output_nodes = 10

# learning rate
learning_rate = 0.2

# create instance of neural network
startTime = datetime.now()
n = nn.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
print("init neural net: ", datetime.now() - startTime)

# epochs is the number of times the training data set is used for training
epochs = 5

for index_epochs, e in enumerate(range(epochs)):
    print("lol")
    # load the image data as test data set and train the neural net
    for index, image_file_name in enumerate(glob.glob("whiskey_bottles/bottles_training_sub/*")):
        print("qwer")
        label = int(image_file_name[48:49])
        img_array = scipy.misc.imread(image_file_name, flatten=True)
        img_data = 255.0 - img_array.reshape(100000)
        img_data = (img_data / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[label] = 0.99
        n.train(img_data, targets)

        print("[", index, "]", image_file_name, " ... min: ", numpy.min(img_data), " ... max: ", numpy.max(img_data))
        pass

    # store the model
    n.store_model("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/whiskey_bottles/who_" + str(index_epochs) + ".json",
                  "C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/whiskey_bottles/wih_" + str(index_epochs) + ".json")
    pass
    print(index_epochs)

# scorecard for how well the network performs
scorecard = []

# test the neural net with all stored models
for index_epochs, e in enumerate(range(epochs)):

    # load model from json file
    n.load_model("C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/whiskey_bottles/who_" + str(index_epochs) + ".json",
                 "C:/Users/Mario/PycharmProjects/nnUnderTheBonnet/weights_json/whiskey_bottles/wih_" + str(index_epochs) + ".json")

    for index, image_file_name in enumerate(glob.glob("whiskey_bottles/bottles_testset_sub_/*")):

        label = int(image_file_name[48:49])
        img_array = scipy.misc.imread(image_file_name, flatten=True)
        img_data = 255.0 - img_array.reshape(100000)
        img_data = (img_data / 255.0 * 0.99) + 0.01

        outputs = n.query(img_data)
        guessed_label = numpy.argmax(outputs)

        print(outputs)
        print(guessed_label)

        # append correct or incorrect to list
        if (label == guessed_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
            print("image: ", image_file_name, "... MATCH")
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            print("image: ", image_file_name, "... NOK")
            pass

        pass
    pass

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print("performance model " + str(index_epochs) + ": " + str(scorecard_array.sum() / scorecard_array.size))
