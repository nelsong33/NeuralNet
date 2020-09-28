import numpy
from neuralNet import neuralNet


# set variables and create neural network object
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = .3

net = neuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

# train neural network
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')

for line in training_data_file:
    all_values = line.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    net.train(inputs, targets)
    pass

training_data_file.close()

# test neural network
testing_data_file = open("mnist_dataset/mnist_test.csv", 'r')

test_results = []
for line in testing_data_file:
    all_values = line.split(',')
    test_results.append([int(all_values[0]), net.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)])
    pass

w = 0
for pair in test_results:
    if pair[0] != pair[1]:
        w += 1
    pass

percent_correct = ((len(test_results) - w) / len(test_results)) * 100

print(percent_correct)
