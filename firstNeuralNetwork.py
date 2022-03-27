from typing import final
import numpy as np
import scipy.special as scp
import matplotlib.pyplot as plt
import pandas as pd

path = 'C:/Users/LENOVO/Programming/Python/Machine Learning'

class neuralNetwork:
    def __init__(self) -> None:
        pass

data = pd.read_csv(f'{path}/data/cancer.csv')
print(data.describe())