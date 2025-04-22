import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ----------------- Data Loading and Preprocessing ----------------- #
data = pd.read_csv('alzheimers_disease_data.csv')
data = data.drop(columns=['PatientID', 'DoctorInCharge'])

# Define feature columns and target
continuous_cols = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 
                  'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
                  'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 
                  'MMSE', 'FunctionalAssessment', 'ADL']
target_col = 'Diagnosis'
X = data.drop(columns=[target_col])
y = data[target_col]

# Normalize and split data
scaler = MinMaxScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_np = X_train.values
y_train_np = y_train.values

# ----------------- Experiment Setup ----------------- #

def create_multi_layer_model(input_shape, architecture):
    model = Sequential([Input(shape=(input_shape,))])
    
    for units in architecture:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(0.001)))
    
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))
    
    model.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.6),
                loss='binary_crossentropy',
                metrics=['accuracy', 'mse']) 
    return model

architectures = {
    'Single Layer (16)': [16],
    'Double Layer (32-16)': [32, 16],       # Decreasing
    'Double Layer (16-32)': [16, 32],       # Increasing 
    'Triple Layer (32-16-8)': [32, 16, 8],  # Steadily decreasing
    'Triple Layer (16-32-16)': [16, 32, 16] # Bottleneck
}

# ----------------- Training and Evaluation ----------------- #
results = []
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, arch in architectures.items():
    print(f"\n=== Testing Architecture: {name} ===")
    
    fold_metrics = {'ce_loss': [], 'mse': [], 'accuracy': []}
    all_val_acc = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        model = create_multi_layer_model(X_train_np.shape[1], arch)
        
        history = model.fit(
            X_train_np[train_idx], y_train_np[train_idx],
            validation_data=(X_train_np[val_idx], y_train_np[val_idx]),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Store metrics
        fold_metrics['ce_loss'].append(history.history['val_loss'][-1])
        fold_metrics['mse'].append(history.history['val_mse'][-1])
        fold_metrics['accuracy'].append(history.history['val_accuracy'][-1])
        all_val_acc.append(history.history['val_accuracy'])
    
    # Calculate mean metrics
    results.append({
        'Architecture': name,
        'CE Loss': f"{np.mean(fold_metrics['ce_loss']):.4f} (±{np.std(fold_metrics['ce_loss']):.4f})",
        'MSE': f"{np.mean(fold_metrics['mse']):.4f} (±{np.std(fold_metrics['mse']):.4f})",
        'Accuracy': f"{np.mean(fold_metrics['accuracy']):.4f} (±{np.std(fold_metrics['accuracy']):.4f})"
    })

# Display results
results_df = pd.DataFrame(results)
print("\n=== Architecture Comparison Results ===")
print(results_df.to_markdown(index=False))

# ----------------- Plotting ----------------- #
plt.figure(figsize=(12, 6))
for name, arch in architectures.items():
    model = create_multi_layer_model(X_train_np.shape[1], arch)
    history = model.fit(X_train_np, y_train_np, epochs=50, batch_size=32, 
                       validation_split=0.2, verbose=0)
    plt.plot(history.history['val_accuracy'], label=name)

plt.title('Validation Accuracy by Network Architecture')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()