import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
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

# print(data.isnull().sum())  # no missing values

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

# MinMaxScaler
scaler = MinMaxScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])  # scale continuous features

# One-Hot Encode categorical features
X = pd.get_dummies(X, columns=categorical_cols)

# Ensure binary variables are 0/1 integers
X[binary_cols] = X[binary_cols].astype(int) 

# Display processed data
print(X.head())

# ----------------- 5-Fold Cross Validation ----------------- #

def create_model(input_shape):  # Pass the input shape dynamically
    model = Sequential([
        Dense(32,activation='relu', input_shape=(input_shape,)),  # 32 16 
        Dense(16,activation='relu'),
        Dense(1, activation='sigmoid')  # binary classification output
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy','mse'])
    return model

# Define 5-Fold Stratified Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

# Iterate over folds
for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Perform SMOTENC on categorical features for each fold
    categorical_feature_names = [col for col in X.columns if any(base_col in col for base_col in categorical_cols)]
    categorical_indices = [X.columns.get_loc(col) for col in categorical_feature_names]

    smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
    
    X_resampled, y_resampled = smote_nc.fit_resample(X_train_fold, y_train_fold)

    print(f"Fold {fold}:")
    # print(f"Train set class distribution after SMOTENC:\n{y_resampled.value_counts(normalize=True)}")
    # print(f"Test set class distribution:\n{y_test_fold.value_counts(normalize=True)}\n")

    # Create and train model for each fold
    model = create_model(X_train_fold.shape[1])
    
    model.fit(X_resampled, y_resampled, epochs=100, batch_size=32, validation_data=(X_test_fold, y_test_fold))

    # Evaluate the model on the test set
    loss, accuracy, mse = model.evaluate(X_test_fold, y_test_fold)
    print(f"Test set accuracy for Fold {fold}: {accuracy * 100:.2f}%\n")  # create a table to store the accuracy of each fold

    # print(f"Average test set accuracy: {np.mean(accuracy) * 100:.2f}%\n") 

    # Optional: Save the model after each fold
    # model.save(f"model_fold_{fold}.h5")

