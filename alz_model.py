import os 
import sys 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ----------------- Data Preprocessing ----------------- #

# import csv data
data = pd.read_csv('alzheimers_disease_data.csv')
# print(data.head())   # testing csv import 
# data.info() 

# print(data.isnull().sum())  #no missing values

data = data.drop(columns=['PatientID', 'DoctorInCharge'])
# print(data.describe().T)

# Identify column types
continuous_cols = [
    'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 
    'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 
    'MMSE', 'FunctionalAssessment', 'ADL'
]

categorical_cols = ['Ethnicity', 'EducationLevel']
binary_cols = [
    'Gender', 'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 
    'Depression', 'HeadInjury', 'Hypertension', 'MemoryComplaints', 'BehavioralProblems', 
    'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
]

# Separate target variable
target_col = 'Diagnosis'
X = data.drop(columns=[target_col])
y = data[target_col]

#MinMaxScaler
scaler = MinMaxScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# One-Hot Encode categorical features
X = pd.get_dummies(X, columns=categorical_cols)

# Ensure binary variables are 0/1 integers
X[binary_cols] = X[binary_cols].astype(int)

# Display processed data
print(X.head())

# Save preprocessed dataset
X.to_csv('preprocessed_data.csv', index=False)
y.to_csv('target.csv', index=False)

# ----------------- 5-Fold Cross Validation ----------------- #