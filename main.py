from imports import np
import network as nn
import utils

training_data, validation_data, test_data = utils.load_data_together()
 
net = nn.Network([784, 30, 10])
net.SGD(training_data, 10, 10, 3.0, test_data=test_data)