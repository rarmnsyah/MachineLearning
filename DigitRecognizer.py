import numpy as np
import matplotlib.pyplot as plt 
import scipy.special as scp

class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate, initWeight = False) -> None:
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        if initWeight:
            self.wih = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
            self.who = np.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))
        else:
            dataweight = open('C:/Users/LENOVO/Programming/Python/Machine Learning/model/modelNNReadLetter.txt', 'r')
            weights = dataweight.readlines()
            self.wih = np.reshape(np.asfarray(weights[0].split(',')[0:-1]), (200, 784))
            self.who = np.reshape(np.asfarray(weights[1].split(',')[0:-1]), (10, 200))
            dataweight.close()
            
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

    def updateWeights(self):
        file1 = open('C:/Users/LENOVO/Programming/Python/Machine Learning/model/modelNNReadLetter.txt', 'w')
        for i in self.wih:
            file1.write(str(i) + ',')

        file1.write('\n')
        for i in self.who:
            file1.write(str(i) + ',')

        file1.close()
      

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1


n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
data = open('C:/Users/LENOVO/Programming/Python/Machine Learning/data/mnist_test.csv', 'r')
datasets = data.readlines()
data.close()
data =[]
for dataset in datasets:
    data.append(dataset.split(','))

def trainModel(data):
    for i in data:
        inputs = np.asfarray(i[1:])/255.0 * 0.99 + 0.1
        targets = np.zeros(10)
        targets[int(i[0])] = 0.99

        n.train(inputs, targets)

    n.updateWeights()

def checkAccuracyModel(data):
    outputAcc = 0
    for i in data:
        inputs = np.asfarray(i[1:])/255.0 * 0.99 + 0.1

        output = n.query(inputs)
        if int(i[0]) == np.argmax(output): outputAcc += 1

    acc = outputAcc / len(data)
    print(acc)

def queryTest():
    inputss = np.asfarray(data[np.random.randint(0,10000)][1:])/255.0 * 0.99 + 0.1
    inputs = np.reshape(inputss, (28,28))
    plt.imshow(inputs, cmap = 'Greys')
    output = n.query(inputss)
    print('labels: {}, shape: {}'.format(np.argmax(output), output.shape))
    plt.show()

def backQueryTest():
    target = np.random.randint(0,10)
    inputs = n.backQuery(target)
    plt.imshow(np.reshape(inputs, (28,28)), cmap = 'Greys')
    print('labels: {}'.format(target))
    plt.show()

queryTest()
