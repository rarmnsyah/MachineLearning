import numpy as np

path = 'C:/Users/LENOVO/Programming/Python/Machine Learning'

file1 = open(f'C:/Users/LENOVO/Programming/Python/MachineLearning/model/modelNNReadIris.txt', 'r')
database = file1.readlines()
sample_data = []

for i in database:
    sample_data.append(i.split(','))

for i in sample_data:
    print(len(i))