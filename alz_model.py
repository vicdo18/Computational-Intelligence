import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
import warnings

# warnings.filterwarnings("ignore", message="Do not pass an `input_shape`/`input_dim` argument to a layer.")

# ----------------- Data Preprocessing ----------------- #

# import csv data
data = pd.read_csv('alzheimers_disease_data.csv')
# print(data.isnull().sum())  # no missing values

data = data.drop(columns=['PatientID', 'DoctorInCharge'])
print(data.describe().T)

# Identify column types
continuous_cols = [
    'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 
    'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 
    'MMSE', 'FunctionalAssessment', 'ADL'
]

categorical_cols = ['Ethnicity', 'EducationLevel']  # not exactly categorical,
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

#z score   -- drop , leave minmax
# X[continuous_cols] = (X[continuous_cols] - X[continuous_cols].mean()) / X[continuous_cols].std()  # z-score normalization

# X.to_csv('processed_data_z.csv', index=False)  # Save the processed data to a new CSV file

# Display processed data
# print(X.head())

# ----------------- Train-Test Split ----------------- #

# Split into 80% train (for CV) and 20% test (final evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, #target
    test_size=0.2, 
    random_state=42,  # reproducibility
    stratify=y  # ensures balanced class distribution
)

print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Class distribution in test set:", y_test.value_counts(normalize=True))

# ----------------- 5-Fold Cross Validation ----------------- #

def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),  # Explicit Input layer
        Dense(32, activation='relu'),  # 1 hidden layer
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.001) #η = 0.001
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    return model

# ----------------- Stratified K-Fold Cross Validation ----------------- #

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics for each fold
fold_metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'roc_auc': []
    'mse': []
}

# Convert data to numpy arrays for compatibility with Keras
X_train_np = X_train.values
y_train_np = y_train.values

# Perform 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
    print(f"\n=== Fold {fold + 1} ===")
    
    # Split data into training and validation sets for this fold
    X_fold_train, X_fold_val = X_train_np[train_idx], X_train_np[val_idx]
    y_fold_train, y_fold_val = y_train_np[train_idx], y_train_np[val_idx]
    
    # Create and train model
    model = create_model(input_shape=X_fold_train.shape[1])
    history = model.fit(
        X_fold_train, 
        y_fold_train,
        validation_data=(X_fold_val, y_fold_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Predict on validation set
    y_pred_proba = model.predict(X_fold_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    fold_metrics['accuracy'].append(accuracy_score(y_fold_val, y_pred))
    fold_metrics['precision'].append(precision_score(y_fold_val, y_pred))
    fold_metrics['recall'].append(recall_score(y_fold_val, y_pred))
    fold_metrics['f1'].append(f1_score(y_fold_val, y_pred))
    fold_metrics['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
    fold_metrics['mse'].append(np.mean((y_fold_val - y_pred.flatten())**2))  # Mean Squared Error
    
    # Print fold results
    print(f"Fold {fold + 1} Metrics:")
    print(f"Accuracy: {fold_metrics['accuracy'][-1]:.4f}")
    print(f"Precision: {fold_metrics['precision'][-1]:.4f}")
    print(f"Recall: {fold_metrics['recall'][-1]:.4f}")
    print(f"F1-Score: {fold_metrics['f1'][-1]:.4f}")
    print(f"ROC AUC: {fold_metrics['roc_auc'][-1]:.4f}")
    print(f"MSE: {fold_metrics['mse'][-1]:.4f}")

# Calculate and print average metrics across all folds
print("\n=== Average Cross-Validation Metrics ===")
for metric in fold_metrics:
    print(f"Mean {metric}: {np.mean(fold_metrics[metric]):.4f} (±{np.std(fold_metrics[metric]):.4f})")


