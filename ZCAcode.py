# Backprop on the Vowel Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import numpy as np
import csv
import copy
import seaborn as sns

def loadNew(file):    
        trainSet = []        
        lines = csv.reader(open(file, 'r'))
        dataset = list(lines)
        #print("training set {}".format(dataset))
        for i in range(len(dataset[0])-1):
                for row in dataset:
                        try:
                                row[i] = float(row[i])
                        except ValueError:
                                print("Error with row",column,":",row[i])
                                pass
        trainSet = dataset
        return trainSet


# Load a CSV file
def loadCsv(filename, file1):
        trainSet = []
        testSet = []
        lines = csv.reader(open(filename, 'r'))
        dataset = list(lines)
        #print("training set {}".format(dataset))
        for i in range(len(dataset[0])):
                for row in dataset:
                        try:
                                row[i] = float(row[i])
                        except ValueError:
                                print("Error with row",row[i])
                                pass
        trainSet = dataset
        lines = csv.reader(open(file1, 'r'))
        dataset = list(lines)
        for i in range(len(dataset[0])):
                for row in dataset:
                        try:
                                row[i] = float(row[i])
                        except ValueError:
                                print("Error with row ",row[i])
                                pass
        testSet = dataset
        #print("training set {}".format(trainSet))
        return trainSet, testSet
 
# Convert string column to float
def str_column_to_float(dataset, column):
        for row in dataset:
                try:
                        row[column] = float(row[column])
                except ValueError:
                        print("Error with row",column,":",row[column])
                        pass
 
# Convert string column to integer
def str_column_to_int(dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
                lookup[value] = i
        for row in dataset:
                row[column] = lookup[row[column]]
        return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
        minmax = list()
        stats = [[min(column), max(column)] for column in zip(*dataset)]
        return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
        for row in dataset:
                for i in range(len(row)-1):
                        row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 

 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
                if actual[i] == predicted[i]:
                        correct += 1
        return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(train_set, test_set, algorithm, *args):               
                test_set_for_class = copy.deepcopy(test_set)
                test_set_again = copy.deepcopy(test_set)
                for row in test_set:
                        del row[-1]
                #print(test_set_for_class)
                network_one, new_train, new_test = algorithm(train_set, test_set, test_set_for_class, *args)

                
                test_set_for_class = copy.deepcopy(new_test)

                print(" {} ", len(new_test))
                
                for row in new_test:
                        del row[-1]

                network_two, second_train, second_test = backpropagation_two(new_train,new_test, test_set_for_class, *args)


                
                network_three, predicted=back_prop_for_classification(second_train, second_test, *args)
                new_network = list()
                for layer in network_one:
                        new_network.append(layer)
                for layer in network_two:
                        new_network.append(layer)
                for layer in network_three:
                        new_network.append(layer)

##                for layer in new_network:
##                        print("\n\n{}\n\n".format(layer))
                n_inputs = len(test_set_again[0]) - 1
                n_outputs = len(set([row[-1] for row in test_set_again]))
                
                #print("For Classification {}".format(network))
                predictions = list()
                for row in test_set_again:
                        prediction = predict(new_network, row)
                        predictions.append(prediction)
                #predicted = back_prop_for_classification(train_set,test_set_for_class, network, *args)
                actual = [int(row[-1]) for row in test_set_again]
                #print(" {} \n\n {} \n\n".format(actual,predicted))
                accuracy = accuracy_metric(actual, predictions)
                cm = confusion_matrix(actual, predictions)
                print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))
                #confusionmatrix = np.matrix(cm)
                FP = cm.sum(axis=0) - np.diag(cm)
                FN = cm.sum(axis=1) - np.diag(cm)
                TP = np.diag(cm)
                TN = cm.sum() - (FP + FN + TP)
                print('False Positives\n {}'.format(FP))
                print('False Negetives\n {}'.format(FN))
                print('True Positives\n {}'.format(TP))
                print('True Negetives\n {}'.format(TN))
                TPR = TP/(TP+FN)
                print('Sensitivity \n {}'.format(TPR))
                TNR = TN/(TN+FP)
                print('Specificity \n {}'.format(TNR))
                Precision = TP/(TP+FP)
                print('Precision \n {}'.format(Precision))
                Recall = TP/(TP+FN)
                print('Recall \n {}'.format(Recall))
                Acc = (TP+TN)/(TP+TN+FP+FN)
                print('Áccuracy \n{}'.format(Acc))
                Fscore = 2*(Precision*Recall)/(Precision+Recall)
                print('FScore \n{}'.format(Fscore))
                flag = 1
                re_train_network(new_network, train_set, flag, l_rate, n_epoch, n_outputs)
                #print("For Classification {}".format(network))
                predictions = list()
                for row in test_set_again:
                        prediction = predict(new_network, row)
                        predictions.append(prediction)
                actual = [int(row[-1]) for row in test_set_again]
                #print(" {} \n\n {} \n\n".format(actual,predicted))
                accuracy = accuracy_metric(actual, predictions)
                cm = confusion_matrix(actual, predictions)
                print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))
                #confusionmatrix = np.matrix(cm)
                FP = cm.sum(axis=0) - np.diag(cm)
                FN = cm.sum(axis=1) - np.diag(cm)
                TP = np.diag(cm)
                TN = cm.sum() - (FP + FN + TP)
                print('False Positives\n {}'.format(FP))
                print('False Negetives\n {}'.format(FN))
                print('True Positives\n {}'.format(TP))
                print('True Negetives\n {}'.format(TN))
                TPR = TP/(TP+FN)
                print('Sensitivity \n {}'.format(TPR))
                TNR = TN/(TN+FP)
                print('Specificity \n {}'.format(TNR))
                Precision = TP/(TP+FP)
                print('Precision \n {}'.format(Precision))
                Recall = TP/(TP+FN)
                print('Recall \n {}'.format(Recall))
                Acc = (TP+TN)/(TP+TN+FP+FN)
                print('Áccuracy \n{}'.format(Acc))
                Fscore = 2*(Precision*Recall)/(Precision+Recall)
                print('FScore \n{}'.format(Fscore))

                                
        #return scores
 
# Calculate neuron activation for an input
def activate(weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
                activation += weights[i] * inputs[i]
        return activation
 
# Transfer neuron activation
def transfer(activation):
        return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
        inputs = row
        for layer in network:
                new_inputs = []
                for neuron in layer:
                        activation = activate(neuron['weights'], inputs)
                        neuron['output'] = transfer(activation)
                        new_inputs.append(neuron['output'])
                inputs = new_inputs
        return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
        return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
        for i in reversed(range(len(network))):
                layer = network[i]
                errors = list()
                if i != len(network)-1:
                        for j in range(len(layer)):
                                error = 0.0
                                for neuron in network[i + 1]:
                                        error += (neuron['weights'][j] * neuron['delta'])
                                errors.append(error)
                else:
                        for j in range(len(layer)):
                                neuron = layer[j]
                                errors.append(expected[j] - neuron['output'])
                for j in range(len(layer)):
                        neuron = layer[j]
                        neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate):
        for i in range(len(network)):
                inputs = row[:-1]
                
                if i != 0:
                        inputs = [neuron['output'] for neuron in network[i - 1]]
                for neuron in network[i]:
                        for j in range(len(inputs)):
                                temp = l_rate * neuron['delta'] * inputs[j] + mu * neuron['prev'][j]                                
                                neuron['weights'][j] += temp
                                #print("neuron weight{} \n".format(neuron['weights'][j]))
                                neuron['prev'][j] = temp
                        temp = l_rate * neuron['delta'] + mu * neuron['prev'][-1]
                        neuron['weights'][-1] += temp
                        neuron['prev'][-1] = temp

def decorrelate(X):
            X_norm = (X - 0) / 1.
            #print ('X.min()', X_norm.min())
            #print ('X.max()', X_norm.max())
            X_norm = center(X_norm)
            cov = np.cov(X_norm, rowvar=True)
            # Calculate the eigenvalues and eigenvectors of the covariance matrix
            U,S,V = np.linalg.svd(cov)
            #print(U, S)
            # Apply the eigenvectors to X
            epsilon = 1.0
            X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X_norm)
            X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())
            #print ('N min:', X_ZCA_rescaled.min())
            #print ('N max:', X_ZCA_rescaled.max())
            return X_ZCA_rescaled


def center(X_norm):
    
    newX = X_norm - np.mean(X_norm, axis = 0)
    #np.savetxt("meanX1.csv", newX.T, delimiter=",")
    return newX

                                
 
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
        plotting = list()
        rang = list(range(1,101))
        for epoch in range(n_epoch):
                i = 1                
                msq = 0
                error = 0
                for row in train:
                        outputs = forward_propagate(network, row)
                        expected=row
                        row_new = copy.deepcopy(row)
                        
                        row_new=row_new[0:-1]
                        #print("{}\n\n".format(row))
                        r = np.array(row_new)
                        o = np.array(outputs)
                        r = r.astype(float)
                        o = o.astype(float)
                        #print("{}      {}".format(row_new,o))
                        err = np.subtract(r,o)
                        #print(err)
                        err = np.sum(err**2)/len(err)
                        #print(err)
                        error = error+err
                        backward_propagate_error(network, expected)
                        update_weights(network, row, l_rate)
                        i = i+1
                mse = error/i
                plotting.append(mse)
        plt.plot(rang, plotting, 'ro')
        plt.xlabel('no of iterations 1000')
        plt.ylabel('Mean square error')
        #plt.show()
        print("mean square error{}\n".format(mse))
                

def re_train_network(network_two, train_set, flag, l_rate, n_epoch, n_outputs):
        print(flag)
        if flag == 0:
                l_r = 0.06
                n_epoch = 100
        else:
                l_r = 0.06
                n_epoch = 900
        print("learning rate  ",l_r)
        plotting = list()
        r = n_epoch+1
        rang = list(range(1,r))
        for epoch in range(n_epoch):
                i = 1                
                msq = 0
                error = 0                
                for row in train_set:
                        outputs = forward_propagate(network_two, row)
                        
                        expected = [0 for i in range(n_outputs)]
                        #print(row)
                        expected[int(row[-1])-1] = 1
                        #print("expected row{}\n".format(expected))

                       
                        #print("{}\n\n".format(row))
                        r = np.array(expected)
                        o = np.array(outputs)
                        r = r.astype(float)
                        o = o.astype(float)
                        #print("{}      {}".format(row_new,o))
                        err = np.subtract(r,o)
                        #print(err)
                        err = np.sum(err**2)/len(err)
                        #print(err)
                        error = error+err
                        backward_propagate_error(network_two, expected)
                        update_weights(network_two, row, l_r)
                        i = i+1
                        
                mse = error/i
                plotting.append(mse)
        plt.plot(rang, plotting, 'ro')
        plt.xlabel('no of iterations 1000')
        plt.ylabel('Mean square error')
        plt.show()
        print("First Classification mean square error{}\n".format(mse))

def prepare_dataset(network, test_with_class):                
        new_data_train = list()        
        for row in trainingSet:
                outputs = forward_propagate(network, row)
                outputs.append(int(row[-1]))
                #print(" {} \n".format(row))
                new_data_train.append(outputs)      
        

        new_data_test = list()        
        for row in test_with_class:
                outputs = forward_propagate(network, row)
                outputs.append(int(row[-1]))
                #print(" {} \n".format(row))
                new_data_test.append(outputs)

        
        len1 = len(new_data_train)
        for i in range(len(new_data_test)):
                new_data_train.append(new_data_test[i])
        
        new_data_train = np.asarray(new_data_train)
        last = new_data_train[:,-1]
        a_train=np.delete(new_data_train,-1,axis=1)
        #plotImage(a_train[45,:])
        a_train = decorrelate(a_train)
        #plotImage(a_train[45,:])
        last = (np.array([last])).T
        #print(last)
        new_data_train = np.append(a_train,last,axis =1)
        new_data_train1 = new_data_train[:len1]
        new_data_test1 = new_data_train[len1:]
        new_data_train1=new_data_train1.tolist()
        new_data_test1=new_data_test1.tolist()        
        return new_data_train1, new_data_test1

def prepare_dataset_two(network, train, test, test_with_class):
        
        second_data_train = list()        
        for row in train:
                outputs = forward_propagate(network, row)
                outputs.append(row[-1])
                #print(" {} \n".format(row))
                second_data_train.append(outputs)      
        
##        csvfile = "newdata_train_two.csv"        
##        with open(csvfile, "w") as output:
##                writer = csv.writer(output, lineterminator='\n')
##                writer.writerows(second_data_train)

        second_data_test = list()        
        for row in test_with_class:
                outputs = forward_propagate(network, row)
                outputs.append(row[-1])
                #print(" {} \n".format(row))
                second_data_test.append(outputs)      
        
##        csvfile = "newdata_test_two.csv"        
##        with open(csvfile, "w") as output:
##                writer = csv.writer(output, lineterminator='\n')
##                writer.writerows(second_data_test)
        len1 = len(second_data_train)
        for i in range(len(second_data_test)):
                second_data_train.append(second_data_test[i])
        
        second_data_train = np.asarray(second_data_train)
        last = second_data_train[:,-1]
        a_train=np.delete(second_data_train,-1,axis=1)
        #plotImage(a_train[45,:])
        a_train = decorrelate(a_train)
        #plotImage(a_train[45,:])
        last = (np.array([last])).T
        #print(last)
        second_data_train = np.append(a_train,last,axis =1)
        second_data_train1 = second_data_train[:len1]
        second_data_test1 = second_data_train[len1:]
        second_data_train1=second_data_train1.tolist()
        second_data_test1=second_data_test1.tolist()
        return second_data_train1, second_data_test1
        
 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [{'weights':[np.random.uniform(-0.9, 0.9) for i in range(n_inputs + 1)], 'prev':[0 for i in range(n_inputs+1)]} for i in range(n_hidden)]        
        network.append(hidden_layer)
##        hidden_layer = [{'weights':[random() for i in range(n_hidden + 1)], 'prev':[0 for i in range(n_hidden+1)]} for i in range(n_hidden)]
##        network.append(hidden_layer)
        output_layer = [{'weights':[np.random.uniform(-0.9, 0.9) for i in range(n_hidden + 1)],'prev':[0 for i in range(n_hidden+1)]} for i in range(n_outputs)]
        network.append(output_layer)
        #print("FIRST {} \n\n".format(network))
        return network


def reinitialize_to_classify(n_inputs, n_outputs):
        network_two = list()
        output_layer = [{'weights':[np.random.uniform(-0.9, 0.9) for i in range(n_inputs + 1)],'prev':[0 for i in range(n_inputs + 1)]} for i in range(n_outputs)]
        network_two.append(output_layer)
        #print(network)
        return network_two
        
 
# Make a prediction with a network
def predict(network, row):
        outputs = forward_propagate(network, row)
        return outputs.index(max(outputs)) + 1


 
# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, test_with_class, l_rate, n_epoch, n_hidden):
        n_inputs = len(train[0]) - 1
        n_outputs = n_inputs
        network = initialize_network(n_inputs, n_hidden, n_outputs)
        train_network(network, train, l_rate, n_epoch, n_outputs)
        #print("network {}\n".format(network))
        avg_msq=0

        for row in test:
                output = forward_propagate(network, row)
                #print(" row {} ---- output {}".format(row,output))
                r = np.array(row)
                o = np.array(output)
                err = r - o
                mssq = np.sum(err**2)/len(err)
                #print("MSE = {}".format(mssq))
                avg_msq=avg_msq+ mssq
        avg_msq=avg_msq/len(test)
        print("  MSE First  {}    ".format(avg_msq))
        #print("FIRST {} \n\n".format(network))
        #network = transform_auto(network,train, test, l_rate, n_epoch, n_hidden, n_outputs)
        network = network[:-1]
        new_train, new_test = prepare_dataset(network,test_with_class)
        #print(network)
        return network, new_train, new_test

def backpropagation_two(train ,test, test_with_class, *args):
        
        n_inputs = len(train[0]) - 1
        n_outputs = n_inputs
        n_hidden_two = 11
        network_two = initialize_network(n_inputs, n_hidden_two, n_outputs)
        train_network(network_two, train, l_rate, n_epoch, n_outputs)
        #print("network {}\n".format(network))
        avg_msq=0
        

        for row in test:
                output = forward_propagate(network_two, row)
                #print(" row {} ---- output {}".format(row,output))
                r = np.array(row)
                o = np.array(output)
                err = r - o
                mssq = np.sum(err**2)/len(err)
                #print("MSE = {}".format(mssq))
                avg_msq=avg_msq+ mssq
        avg_msq=avg_msq/len(test)
        print("  MSE Second  {}    ".format(avg_msq))       
        network_two = network_two[:-1]
        second_train, second_test = prepare_dataset_two(network_two,train, test, test_with_class)
        #print(network)
        return network_two, second_train, second_test
    
    

def back_prop_for_classification(train_set,test_set_for_class, *args):        
        n_inputs = len(train_set[0]) - 1
        n_outputs = len(set([row[-1] for row in train_set]))
        flag =0
        #print(n_outputs)
        network_three = reinitialize_to_classify(n_inputs, n_outputs)
        re_train_network(network_three, train_set, flag, l_rate, n_epoch, n_outputs)
        #print("For Classification {}".format(network))
        predictions = list()
        for row in test_set_for_class:
                prediction = predict(network_three, row)
                predictions.append(prediction)
        #print(predictions)
        return network_three, predictions

# Test Backprop on Seeds dataset
seed(1)
# load and prepare data, epsilon = 1.0
filename = 'zca_row_mean_ep_10.csv'
file1 = 'zca_row_test_ep_10.csv'
        #sRatio = 0.80
trainingSet, testSet = loadCsv(filename, file1)


l_rate = 0.3
mu=0.001
n_epoch = 100
n_hidden = 19
scores = evaluate_algorithm(trainingSet,testSet, back_propagation, l_rate, n_epoch, n_hidden)

#print('Scores: %s' % scores)
#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))





