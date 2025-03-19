import os 
import sys 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTENC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

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

# # Save preprocessed dataset
# X.to_csv('preprocessed_data.csv', index=False)
# y.to_csv('target.csv', index=False)

# ----------------- 5-Fold Cross Validation ----------------- #

# Define 5-Fold Stratified Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over folds
for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    print(f"Fold {fold}:")
    print(f"Train set class distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test set class distribution:\n{y_test.value_counts(normalize=True)}\n")

# Check the class distribution in the entire dataset
print(data['Diagnosis'].value_counts())
# Check the percentage distribution
print(data['Diagnosis'].value_counts(normalize=True) * 100)
 
# Identify categorical feature indices AFTER one-hot encoding
categorical_feature_names = [col for col in X.columns if any(base_col in col for base_col in categorical_cols)]
categorical_indices = [X.columns.get_loc(col) for col in categorical_feature_names]

# Perform SMOTENC with corrected categorical indices
smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)

# Display class distribution after SMOTENC
print("Class distribution after SMOTENC:")
print(y_resampled.value_counts(normalize=True) * 100)


# ----------------- Model Training A2 ----------------- #

# Define model architecture

# model = Sequential([
#     Dense(32, activation='relu', input_shape=(X.shape[1],)), # input # 32 16 1
#     Dense(16, activation='relu'), # hidden layer
#     Dense(1, activation='sigmoid') # output
# ])

model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),  
    Dropout(0.2),  # Dropout with 30% probability

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),  #erwthma g A2
    Dropout(0.2),

    # Dense(32, activation='relu', kernel_regularizer=l2(0.001)),  
    # Dropout(0.2),

    Dense(1, activation='sigmoid')  # Output layer for binary classification                 
])

# Set the optimizer with learning rate η = 0.001
optimizer = Adam(learning_rate=0.001)  # η = 0.001

# Compile model with Cross-Entropy loss
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy','mse'])

# Train model
# history = model.fit(X_smote, y_smote, epochs=100, batch_size=32, validation_split=0.2)
history = model.fit(X_resampled, y_resampled, epochs=100, batch_size=16, validation_split=0.2)

