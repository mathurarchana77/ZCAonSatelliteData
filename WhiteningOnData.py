from random import seed
from random import randrange
from random import random

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import numpy as np
import csv
import copy
import seaborn as sns

def loadCsv(filename):
        trainSet = []
        
        lines = csv.reader(open(filename, 'r'))
        dataset = list(lines)
        #print("training set {}".format(dataset))
        for i in range(len(dataset[0])):
                for row in dataset:
                        try:
                                row[i] = float(row[i])
                        except ValueError:
                                print("Error with row",column,":",row[i])
                                pass
        trainSet = dataset        
        return trainSet
 
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

def decorrelate(X):
            X_norm = (X - 27) / 130.
            print ('X.min()', X_norm.min())
            print ('X.max()', X_norm.max())
            X_norm = center(X_norm)
            cov = np.cov(X_norm, rowvar=True)
            # Calculate the eigenvalues and eigenvectors of the covariance matrix
            U,S,V = np.linalg.svd(cov)
            print(U, S)
            # Apply the eigenvectors to X
            epsilon = 3.0
            X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X_norm)
            X_ZCA_rescaled = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())
            print ('N min:', X_ZCA_rescaled.min())
            print ('N max:', X_ZCA_rescaled.max())
            return X_ZCA_rescaled


def center(X_norm):
    
    newX = X_norm - np.mean(X_norm, axis = 0)
    #np.savetxt("meanX1.csv", newX.T, delimiter=",")
    return newX

def standardize(X):
    newX = center(X)/np.std(X, axis = 0)
    return newX

def plotImage(X):
    plt.figure(figsize=(1.5, 1.5))
    plt.imshow(X.reshape(3,3,4))
    plt.show()
    plt.close()

seed(1)
# load and prepare data
filename = 'Sat_train_comb.csv'

        #sRatio = 0.80
trainingSet = loadCsv(filename)



# normalize input variables
##minmax = dataset_minmax(trainingSet)
##normalize_dataset(trainingSet, minmax)
##
### normalize input variables
##minmax = dataset_minmax(testSet)
##normalize_dataset(testSet, minmax)

#process training set
trainingSet = np.asarray(trainingSet)
last = trainingSet[:,-1]
a_train=np.delete(trainingSet,-1,axis=1)
#plotImage(a_train[45,:])
a_train = decorrelate(a_train)
#plotImage(a_train[45,:])
last = (np.array([last])).T
#print(last)
a_train = np.append(a_train,last,axis =1)

##minmax = dataset_minmax(a_train)
##normalize_dataset(a_train, minmax)

np.savetxt("zca_row_mean_ep_30.csv", a_train, delimiter=",")

#process test set


ACov = np.cov(a_train, rowvar=False, bias=True)
#print ('Covariance matrix:\n', ACov)

##sns.distplot(trainingSet[:,0], color="#53BB04")
##sns.distplot(trainingSet[:,1], color="#0A98BE")



fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(10, 10)

ax0 = plt.subplot(2, 2, 1)

# Choosing the colors
cmap = sns.color_palette("GnBu", 10)
sns.heatmap(ACov, cmap=cmap, vmin=0)

ax1 = plt.subplot(2, 2, 2)

# data can include the colors
if trainingSet.shape[1]==3:
        c=data[:,2]
else:
        c="#0A98BE"
        
#ax1.scatter(CDecorrelated[:,0], CDecorrelated[:,1], c=c, s=40)
ax1.scatter(trainingSet[:,0], trainingSet[:,4], c=c, s=40)

# Remove the top and right axes from the data plot
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

#plt.show()
plt.close()
trainingSet.tolist()
