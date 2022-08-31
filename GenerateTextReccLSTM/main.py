import numpy as np 
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM   
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

cwd = os.path.dirname(os.path.realpath(__file__))
filename = 'AliceWonderland.txt'
raw_text = open(filename,'r', encoding='utf-8').read()
raw_text = raw_text.lower()

print(raw_text)
