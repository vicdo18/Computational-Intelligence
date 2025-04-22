import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from collections import Counter

# ----------------- Data Preprocessing ----------------- #
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

def create_model(input_shape, hidden_units=32):  # Default 32 neurons if not specified
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(hidden_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'mse', tf.keras.metrics.AUC(name='auc')] 
    )
    return model

# ----------------- Stratified K-Fold Cross Validation ----------------- #

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

#------------ TEST I ------------------------------#

# Define neuron counts to test (I = number of input features)
I = X_train_np.shape[1]
hidden_units_list = [I//2, 2*I//3, I, 2*I]  # [I/2, 2I/3, I, 2I]

# Hyperparameter optimization for learning rate and momentum
learning_rates = [0.001, 0.001, 0.05, 0.1]
momentums = [0.2, 0.6, 0.6, 0.6]

# Store results
results = {
    'Hidden Units': [],
    'Val Accuracy': [],
    'MSE': [],
    'CE Loss': [],
    'Val F1': [],
    'Val ROC AUC': []
}

hyper_results = {
    'η': [],
    'm': [],
    'CE Loss': [],
    'MSE': [],
    'Accuracy': []
}

# plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red', 'orange']  # Colors for different hidden units

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',     # Metric to monitor
    patience=5,            # Number of epochs with no improvement before stopping
    restore_best_weights=True,  # Keep the best weights when stopping
    verbose=1
)

mean_val_accuracy = [] 

#------------------------------------------ A2 -----------------------------#

# for H, color in zip(hidden_units_list, colors):
#     print(f"\n=== Experiment: Hidden Units = {H} ===")
    
#     # Track metrics across folds
#     fold_histories = {'val_loss': [], 'val_accuracy': [], 'val_auc': [],'val_mse': []}
#     all_val_accuracies = []  # Store all accuracy histories for plotting
    
#     for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train_np, y_train_np)):
#         model = create_model(input_shape=X_train_np.shape[1], hidden_units=H)
#         history = model.fit(
#             X_train_np[train_idx], y_train_np[train_idx],
#             validation_data=(X_train_np[val_idx], y_train_np[val_idx]),
#             epochs=50,  
#             batch_size=32,
#             callbacks=[early_stopping],  
#             verbose=0
#         )
        
#         # Store metrics
#         for key in fold_histories:
#             fold_histories[key].append(history.history[key])
        
#         # Store accuracy history for this fold
#         all_val_accuracies.append(history.history['val_accuracy'])

    
#     # Calculate mean accuracy (handles variable epoch counts)
#     max_epochs = max(len(acc) for acc in all_val_accuracies)
#     mean_val_accuracy = np.zeros(max_epochs)
#     counts = np.zeros(max_epochs)

#     for acc in all_val_accuracies:
#         current_epochs = len(acc)
#         mean_val_accuracy[:current_epochs] += np.array(acc)
#         counts[:current_epochs] += 1
        
#     mean_val_accuracy = mean_val_accuracy / counts  # Avoid division by zero
    
#     # Plot the mean accuracy
#     plt.plot(mean_val_accuracy, color=color, label=f'H={H}', linestyle='-')
    
#     # Record results
#     results['Hidden Units'].append(H)
#     results['Val Accuracy'].append(
#         f"{np.mean([h[-1] for h in fold_histories['val_accuracy']]):.3f} ± "
#         f"{np.std([h[-1] for h in fold_histories['val_accuracy']]):.3f}"
# )
#     results['Val F1'].append(
#         f"{np.mean([f1_score(y_train_np[val_idx], (model.predict(X_train_np[val_idx]) > 0.5).astype(int)) for _, val_idx in StratifiedKFold(n_splits=5).split(X_train_np, y_train_np)]):.3f}")
#     results['Val ROC AUC'].append(
#         f"{np.mean([h[-1] for h in fold_histories['val_auc']]):.3f} ± "
#         f"{np.std([h[-1] for h in fold_histories['val_auc']]):.3f}"
#     )
#     results['CE Loss'].append(np.mean([h[-1] for h in fold_histories['val_loss']]))
#     results['MSE'].append(np.mean([h[-1] for h in fold_histories['val_mse']]))  # From metrics
    

# # Plot formatting
# plt.title('Validation Accuracy Convergence (50 Epochs)')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

#-------------------------- A3 ------------------------------------------#

# all_convergence_curves = []
# labels = []

# for lr, momentum in zip(learning_rates, momentums):
#     print(f"\n=== Testing η={lr}, m={momentum} ===")
    
#     fold_metrics = {'val_loss': [], 'val_accuracy': [], 'val_mse': []}
#     all_val_accuracies = []
    
#     for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=5).split(X_train_np, y_train_np)):
#         model = Sequential([
#             Input(shape=(X_train_np.shape[1],)),
#             Dense(16, activation='relu'),  # Using optimal 16 neurons
#             Dense(1, activation='sigmoid')
#         ])
        
#         optimizer = Adam(learning_rate=lr, beta_1=momentum)
#         model.compile(optimizer=optimizer,
#                     loss='binary_crossentropy',
#                     metrics=['accuracy', 'mse'])
        
#         history = model.fit(
#             X_train_np[train_idx], y_train_np[train_idx],
#             validation_data=(X_train_np[val_idx], y_train_np[val_idx]),
#             epochs=50,
#             batch_size=32,
#             callbacks=[early_stopping],
#             verbose=0
#         )
        
#         fold_metrics['val_loss'].append(history.history['val_loss'][-1])
#         fold_metrics['val_accuracy'].append(history.history['val_accuracy'][-1])
#         fold_metrics['val_mse'].append(history.history['val_mse'][-1])
#         all_val_accuracies.append(history.history['val_accuracy'])
    
#     # Calculate mean validation accuracy curve
#     max_epochs = max(len(acc) for acc in all_val_accuracies)
#     mean_val_accuracy = np.zeros(max_epochs)
#     counts = np.zeros(max_epochs)
    
#     for acc in all_val_accuracies:
#         current_epochs = len(acc)
#         mean_val_accuracy[:current_epochs] += np.array(acc)
#         counts[:current_epochs] += 1
    
#     mean_val_accuracy = mean_val_accuracy / counts
#     all_convergence_curves.append(mean_val_accuracy)
#     labels.append(f"η={lr}, m={momentum}")
#     # plt.plot(mean_val_accuracy, label=f'η={lr}, m={momentum}')
    
#     # Record results
#     hyper_results['η'].append(lr)
#     hyper_results['m'].append(momentum)
#     hyper_results['CE Loss'].append(f"{np.mean(fold_metrics['val_loss']):.4f}")
#     hyper_results['MSE'].append(f"{np.mean(fold_metrics['val_mse']):.4f}")
#     hyper_results['Accuracy'].append(f"{np.mean(fold_metrics['val_accuracy']):.4f}")

# plt.figure(figsize=(10, 6))
# for curve, label in zip(all_convergence_curves, labels):
#     plt.plot(curve, label=label)
# plt.xlabel("Epoch")
# plt.ylabel("Mean Validation Accuracy")
# plt.title("Convergence Curves for Different Hyperparameters")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Display results
# hyper_df = pd.DataFrame(hyper_results)
# print("\n=== Hyperparameter Optimization Results ===")
# print(hyper_df.to_markdown(index=False))

#------------------------------------------------------------------------#

# # Display results
# results_df = pd.DataFrame(results)
# print("\n=== Performance Summary (50 Epochs) ===")
# print(results_df.to_markdown(index=False))

#save results to csv
# results_df.to_csv('results.csv', index=False)

# ----------------- Regularization Experiment (A4) ----------------- #

# Define the model creation function with L2 regularization
def create_regularized_model(input_shape, hidden_units=16, l2_lambda=0.001):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(hidden_units, activation='relu', 
              kernel_regularizer=l2(l2_lambda)),
        Dense(1, activation='sigmoid', 
              kernel_regularizer=l2(l2_lambda))
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.6),  # Using best η/m from A3
        loss='binary_crossentropy',
        metrics=['accuracy', 'mse']
    )
    return model

# Regularization coefficients to test
regularization_rates = [0.0001, 0.001, 0.01]

# Store results
regularization_results = {
    'r': [],
    'CE Loss': [],
    'MSE': [],
    'Accuracy': [],
    'Convergence_Curve': []
}

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=0
)

# Initialize plot
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red']

# Test each regularization rate
for r, color in zip(regularization_rates, colors):
    print(f"\n=== Testing L2 Regularization with r={r} ===")
    
    fold_metrics = {'val_loss': [], 'val_accuracy': [], 'val_mse': []}
    all_val_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=5).split(X_train_np, y_train_np)):
        model = create_regularized_model(
            input_shape=X_train_np.shape[1],
            l2_lambda=r
        )
        
        history = model.fit(
            X_train_np[train_idx], y_train_np[train_idx],
            validation_data=(X_train_np[val_idx], y_train_np[val_idx]),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Store metrics
        fold_metrics['val_loss'].append(history.history['val_loss'][-1])
        fold_metrics['val_accuracy'].append(history.history['val_accuracy'][-1])
        fold_metrics['val_mse'].append(history.history['val_mse'][-1])
        all_val_accuracies.append(history.history['val_accuracy'])
    
    # Calculate mean validation accuracy curve
    max_epochs = max(len(acc) for acc in all_val_accuracies)
    mean_val_accuracy = np.zeros(max_epochs)
    counts = np.zeros(max_epochs)
    
    for acc in all_val_accuracies:
        current_epochs = len(acc)
        mean_val_accuracy[:current_epochs] += np.array(acc)
        counts[:current_epochs] += 1
    
    mean_val_accuracy = mean_val_accuracy / counts
    
    # Store results
    regularization_results['r'].append(r)
    regularization_results['CE Loss'].append(f"{np.mean(fold_metrics['val_loss']):.4f}")
    regularization_results['MSE'].append(f"{np.mean(fold_metrics['val_mse']):.4f}")
    regularization_results['Accuracy'].append(f"{np.mean(fold_metrics['val_accuracy']):.4f}")
    regularization_results['Convergence_Curve'].append(mean_val_accuracy)
    
    # Plot the convergence curve
    plt.plot(mean_val_accuracy, color=color, label=f'r={r}', linewidth=2)

# Finalize plot
plt.title('Validation Accuracy Convergence with L2 Regularization')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.ylim(0.5, 0.9)  
plt.tight_layout()
plt.show()

# Display results in a table
reg_df = pd.DataFrame({
    'r': regularization_results['r'],
    'CE Loss': regularization_results['CE Loss'],
    'MSE': regularization_results['MSE'],
    'Accuracy': regularization_results['Accuracy']
})
print("\n=== Regularization Results ===")
print(reg_df.to_markdown(index=False))
