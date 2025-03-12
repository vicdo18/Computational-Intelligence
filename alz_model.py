import os 
import sys 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# ----------------- Data Preprocessing ----------------- #

# import csv data
data = pd.read_csv('alzheimers_disease_data.csv')
# print(data.head())   # testing csv import 

# data.info() 

# print(data.isnull().sum())  #no missing values

# drop doctorincharge column 
data = data.drop(['DoctorInCharge'], axis=1)
# print(data.head())

print(data.describe().T)

#perform centering 
data_centered = data - data.mean()
print(data_centered.describe().T)