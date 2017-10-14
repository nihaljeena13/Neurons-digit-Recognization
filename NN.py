import os
import random 
import numpy as np
import sklearn
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

input_size = 785
output_size = 10
training_set = 100
testing_set = 50
n = [20, 50, 100]
momentum = [0, .25, .50]
eta = 0.1

#####################################################################################
# Functions for forward and back propagation
#####################################################################################

def sig(z):
    return 1/(1 + np.exp(-z))

def d_sig(z):
    return z*(1-z)

def forwardp(data, wi2h, wh2o): 
    ai = np.reshape(data, (1, input_size))
    ah = sig(np.dot(ai, wi2h))
    ah[0][0] = 1
    ao = sig(np.dot(ah, wh2o))
    return ai, ah, ao

def backp(error, ai, ah, ao, wh2o, wi2h, d_wh2o_prev, d_wi2h_prev, mv):
    delta_k = d_sig(ao)*error
    delta_j = d_sig(ah)*np.dot(delta_k, np.transpose(wh2o))
    d_wh2o_curr = (eta*np.dot(np.transpose(ah), delta_k)) + (mv*d_wh2o_prev)
    d_wi2h_curr = (eta*np.dot(np.transpose(ai), delta_j)) + (mv*d_wi2h_prev)
    wh2o = wh2o + d_wh2o_curr
    wi2h = wi2h + d_wi2h_curr
    return wh2o, wi2h, d_wh2o_curr, d_wi2h_curr

####################################################################################
# Functions for training and testing 
####################################################################################

def train_NN(wh2o, wi2h, d_wh2o_prev, d_wi2h_prev, mv):
    for i in range(0, training_set):
        ai, ah, ao = forwardp(training_inputs[i, :], wi2h, wh2o)
        t_k = np.insert((np.zeros((1, output_size - 1)) + 0.0001), int(training_outputs[i]), 0.9999)
        wh2o, wi2h, d_wh2o_prev, d_wi2h_prev = backp(t_k - ao, ai, ah, ao, wh2o, wi2h, d_wh2o_prev, d_wi2h_prev, mv)
    return wi2h, wh2o

def test_NN(dataset, data_labels, set_size, wi2h, wh2o):
    pred = []
    for i in range(0, set_size):
        ai, ah, ao = forwardp(dataset[i, :], wi2h, wh2o)
        pred.append(np.argmax(ao))
    return accuracy_score(data_labels, pred), pred

####################################################################################
# Nural network
####################################################################################

def NN(h_size, mv):
    print("entering")
    wi2h = (np.random.rand(input_size, h_size) - 0.5)*0.1       # Randomising weights
    wh2o = (np.random.rand(h_size, output_size) - 0.5)*0.1
    
    d_wh2o_prev = np.zeros(wh2o.shape)
    d_wi2h_prev = np.zeros(wi2h.shape)

    arr_test_acc = []
    arr_train_acc = []
    for epoch in range(0, 50):
        training_accuracy, pred = test_NN(training_inputs, training_outputs, training_set, wi2h, wh2o)
        testing_accuracy, pred = test_NN(testing_inputs, testing_outputs, testing_set, wi2h, wh2o)
        print("Epoch " + str(epoch) + " :\tTraining Set Accuracy = " + str(training_accuracy) + "\n\t\tTest Set Accuracy = " + str(testing_accuracy))
        wi2h, wh2o = train_NN(wh2o, wi2h, d_wh2o_prev, d_wi2h_prev,mv)
        arr_train_acc.append(training_accuracy)
        arr_test_acc.append(testing_accuracy)
    epoch = epoch + 1
    training_accuracy, pred = test_NN(training_inputs, training_outputs, training_set, wi2h, wh2o)
    testing_accuracy, pred = test_NN(testing_inputs, testing_outputs, testing_set, wi2h, wh2o)
    print("Epoch " + str(epoch) + " :\tTraining Set Accuracy = " + str(training_accuracy) + "\n\t\tTest Set Accuracy = " + str(testing_accuracy) + "\n\nHidden Layer Size = " + str(h_size) + "\tMomentum = " + str(mv) + "\tTraining Samples = " + str(training_set) + "\n\nConfusion Matrix :\n")
    print(confusion_matrix(testing_outputs, pred))
    print("\n")

# Graph Plot
    plt.title("Learning Rate %r" %eta)
    plt.plot(arr_train_acc)
    plt.plot(arr_test_acc)
    plt.ylabel("Acuracy %")
    plt.xlabel("Epochs")
    plt.show()    
    return

####################################################################################
# Loading the MNIST data set
####################################################################################

def data_ld_scale(file_name):
    data_file = np.loadtxt(file_name, delimiter=',')
    dataset = np.insert(data_file[:, np.arange(1, input_size)]/255, 0, 1, axis=1)
    data_labels = data_file[:, 0]
    return dataset, data_labels

####################################################################################
# Initiating
####################################################################################

print("\nLoading Training Set")
training_inputs, training_outputs = data_ld_scale('mnist_train.csv')
print("\nLoading Test Set\n")
testing_inputs, testing_outputs = data_ld_scale('mnist_test.csv')

####################################################################################
# Experiment 1
####################################################################################

print("\n Experiment 1 \n")
for h_size in n:
	print("Starting loop")
    NN(h_size, 0.9)

####################################################################################
# Experiment 2
####################################################################################

print("\n Experiment 2 \n")
for alpha in momentum:
    NN(100, alpha)

####################################################################################
# Experiment 3
####################################################################################

print("\n Experiment 3 \n")
for i in range(0, 2):
    training_inputs,X, training_outputs, Y = train_test_split(training_inputs, training_outputs, test_size = 0.50)
    training_set = int(training_set/2)
    NN(100, 0.9)