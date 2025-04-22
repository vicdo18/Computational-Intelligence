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
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from collections import Counter

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

# def create_model(input_shape):
#     model = Sequential([
#         Input(shape=(input_shape,)),  # Explicit Input layer
#         Dense(32, activation='relu'),  # 1 hidden layer
#         Dense(1, activation='sigmoid')
#     ])
#     optimizer = Adam(learning_rate=0.001) #η = 0.001
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mse'])
#     return model

def create_model(input_shape, hidden_units=32):  # Default 32 neurons if not specified
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(hidden_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'mse', tf.keras.metrics.AUC(name='auc')]  # Track MSE explicitly
    )
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
    'roc_auc': [],
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
    model = create_model(input_shape=X_fold_train.shape[1])  # No hidden_units specified
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
    fold_metrics['mse'].append(history.history['val_mse'][-1])
    
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


#print input shape
# print("Input shape:", X_train.shape[1])

#------------ TEST I ------------------------------#

# Define neuron counts to test (I = number of input features)
I = X_train_np.shape[1]
hidden_units_list = [I//2, 2*I//3, I, 2*I]  # [I/2, 2I/3, I, 2I]

# Store results
results = {
    'Hidden Units': [],
    'Val Accuracy': [],
    'MSE': [],
    'CE Loss': [],
    'Val F1': [],
    'Val ROC AUC': []
    # 'Final Loss': []
}

# Plotting setup
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red', 'orange']  # Colors for different hidden units

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',     # Metric to monitor
    patience=5,            # Number of epochs with no improvement before stopping
    restore_best_weights=True,  # Keep the best weights when stopping
    verbose=1
)

mean_val_accuracy = [] # solution 2 

for H, color in zip(hidden_units_list, colors):
    print(f"\n=== Experiment: Hidden Units = {H} ===")
    
    # Track metrics across folds
    fold_histories = {'val_loss': [], 'val_accuracy': [], 'val_auc': [],'val_mse': []}
    all_val_accuracies = []  # Store all accuracy histories for plotting
    # n_epochs = len(history.history['val_accuracy'])
    # mean_val_accuracy = np.zeros(n_epochs)
    # mean_val_accuracy = np.zeros(50)  # For convergence plot
    
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_np, y_train_np)):
        model = create_model(input_shape=X_train_np.shape[1], hidden_units=H)
        history = model.fit(
            X_train_np[train_idx], y_train_np[train_idx],
            validation_data=(X_train_np[val_idx], y_train_np[val_idx]),
            epochs=50,  
            batch_size=32,
            callbacks=[early_stopping],  
            verbose=0
        )
        
        # Store metrics
        for key in fold_histories:
            fold_histories[key].append(history.history[key])
        
        # Store accuracy history for this fold
        all_val_accuracies.append(history.history['val_accuracy'])

        # For convergence plot
        # mean_val_accuracy.append(history.history['val_accuracy'])
        # mean_val_accuracy += np.array(history.history['val_accuracy'])
    
    # Calculate mean accuracy (handles variable epoch counts)
    max_epochs = max(len(acc) for acc in all_val_accuracies)
    mean_val_accuracy = np.zeros(max_epochs)
    counts = np.zeros(max_epochs)

    for acc in all_val_accuracies:
        current_epochs = len(acc)
        mean_val_accuracy[:current_epochs] += np.array(acc)
        counts[:current_epochs] += 1
        
    mean_val_accuracy = mean_val_accuracy / counts  # Avoid division by zero
    
    # Plot the mean accuracy
    plt.plot(mean_val_accuracy, color=color, label=f'H={H}', linestyle='-')

    # # Calculate mean metrics
    # mean_val_accuracy /= 5  # Average across folds
    # plt.plot(mean_val_accuracy, color=color, label=f'H={H}', linestyle='-')
    
    # Record results
    results['Hidden Units'].append(H)
    results['Val Accuracy'].append(
        f"{np.mean([h[-1] for h in fold_histories['val_accuracy']]):.3f} ± "
        f"{np.std([h[-1] for h in fold_histories['val_accuracy']]):.3f}"
)
    results['Val F1'].append(
        f"{np.mean([f1_score(y_train_np[val_idx], (model.predict(X_train_np[val_idx]) > 0.5).astype(int)) for _, val_idx in StratifiedKFold(n_splits=5).split(X_train_np, y_train_np)]):.3f}")
    results['Val ROC AUC'].append(
        f"{np.mean([h[-1] for h in fold_histories['val_auc']]):.3f} ± "
        f"{np.std([h[-1] for h in fold_histories['val_auc']]):.3f}"
    )
    results['CE Loss'].append(np.mean([h[-1] for h in fold_histories['val_loss']]))
    results['MSE'].append(np.mean([h[-1] for h in fold_histories['val_mse']]))  # From metrics
    

# Plot formatting
plt.title('Validation Accuracy Convergence (50 Epochs)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Display results
results_df = pd.DataFrame(results)
print("\n=== Performance Summary (50 Epochs) ===")
print(results_df.to_markdown(index=False))

#save results to csv
# results_df.to_csv('results.csv', index=False)
