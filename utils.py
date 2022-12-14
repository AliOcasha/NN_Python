from imports import gzip,pickle
from imports import np
# activation function 
def sigmoid(n):
    return 1.0/(1.0+np.exp(-n))

# calculate delta or gradient for backpropagation
def sigmoid_derivative(n):
    return sigmoid(n)*(1-sigmoid(n))

def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)
 
def one_hot_encode(y):
    encoded = np.zeros((10, 1))
    encoded[y] = 1.0
    return encoded
 
def load_data_together():
    train, validate, test = load_data()
    train_x = [np.reshape(x, (784, 1)) for x in train[0]]
    train_y = [one_hot_encode(y) for y in train[1]]
    training_data = zip(train_x, train_y)
    validate_x = [np.reshape(x, (784, 1)) for x in validate[0]]
    validate_y = [one_hot_encode(y) for y in validate[1]]
    validation_data = zip(validate_x, validate_y)
    test_x = [np.reshape(x, (784, 1)) for x in test[0]]
    test_y = [one_hot_encode(y) for y in test[1]]
    testing_data = zip(test_x, test_y)
    return (training_data, validation_data, testing_data)