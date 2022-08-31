from typing import final
import numpy as np
import scipy.special as scp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import os
path = os.path.dirname(os.path.realpath(__file__))

class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate) -> None:
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        self.wih = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
        self.who = np.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))
        self.lr = learningRate
        self.actFunc = lambda x: scp.expit(x)
        self.inverse_activation_function = lambda x: scp.logit(x)

    def query(self, inputs):
        inputs = np.array(inputs, ndmin = 2).T
        inputOutput = np.dot(self.wih, inputs)

        hiddenInput = self.actFunc(inputOutput)
        hiddenOutput = np.dot(self.who, hiddenInput)

        outputs = self.actFunc(hiddenOutput)
        return outputs

    def train(self, xTrain, yTrain):
        x = np.array(xTrain, ndmin = 2).T
        y = np.array(yTrain, ndmin = 2).T
        
        inputOutput = np.dot(self.wih, x)
        hiddenInput = self.actFunc(inputOutput)

        hiddenOutput = np.dot(self.who, hiddenInput)
        outputs = self.actFunc(hiddenOutput)

        loss = y - outputs
        hiddenErrors = np.dot(self.who.T, loss)

        self.who += self.lr * np.dot((loss * outputs * (1.0 - outputs)), hiddenInput.T )
        self.wih += self.lr * np.dot((hiddenErrors * hiddenInput * ( 1.0 - hiddenInput)), x.T)

    def backQuery(self, target):
        final_outputs = np.zeros([10,1]) + 0.01
        final_outputs[target] = 0.99

        final_inputs = self.inverse_activation_function(final_outputs)
        hidden_outputs = np.dot(self.who.T, final_inputs)
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        inputs = np.dot(self.wih.T, hidden_inputs)
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs
      

iris = load_iris()
sample_data = pd.DataFrame(iris['data'], columns = iris['feature_names'], index = iris['target'])

targetNames = iris['target_names']

target = iris['target']
X_train, X_test, Y_train,  Y_test = train_test_split(sample_data, target, test_size = 0.3, random_state = 1)

Y_train2 = np.zeros([len(Y_train), 3]) + 0.01
for j,i in enumerate(Y_train2): i[Y_train[j]] = 0.99

Y_test2 = np.zeros([len(Y_test), 3]) + 0.01
for j,i in enumerate(Y_test2): i[Y_test[j]] = 0.99


inputNodes = 4
hiddenNodes = 100
outputNodes = 3
learningRate = 0.8

n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

X_train = np.array(X_train)
X_test = np.array(X_test)

for i in range(4):
    X_train[:, i] = ((X_train[:, i] / np.amax(X_train[:,i],axis=0)) * 0.99) + 0.01
    X_test[:, i] = ((X_test[:, i] / np.amax(X_test[:,i],axis=0)) * 0.99) + 0.01

accuracyPerEpochs = []

for i in range(15):
    for i in range(len(X_train)):
        inputs = X_train[i]
        targets = Y_train2[i]

        n.train(inputs,targets)

    nn = 0
    for i in range(len(X_test)):
        inputs = X_test[i]

        outputs = n.query(inputs)

        pred, real = targetNames[np.argmax(outputs)],targetNames[np.argmax(Y_test2[i])]

        nn += 1 if pred == real else 0

    # print('labels : {}\treal : {}'.format(pred, real))
    # print(nn / len(X_test))

    accuracyPerEpochs.append(nn / len(X_test))
    
print(accuracyPerEpochs[-1])
plt.plot(accuracyPerEpochs)
plt.show()


