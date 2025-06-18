import pygad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Load and preprocess data
data = pd.read_csv("alzheimers_disease_data.csv")
data = data.drop(columns=["PatientID", "DoctorInCharge"])

target_col = "Diagnosis"
X = data.drop(columns=[target_col])
y = data[target_col]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Settings
num_features = X.shape[1]  # Should be 34
λ = 0.05  # penalty coefficient

# Fixed NN model architecture (best from Part A)
def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

# Fitness function
def fitness_func(solution, sol_idx):
    selected_features = np.where(solution == 1)[0]
    
    if len(selected_features) < 2:
        return 0  # discard trivial solutions

    X_train_sub = X_train[:, selected_features]
    X_val_sub = X_val[:, selected_features]

    model = create_model(X_train_sub.shape[1])
    model.fit(X_train_sub, y_train, epochs=20, batch_size=32, verbose=0)

    loss, acc = model.evaluate(X_val_sub, y_val, verbose=0)
    
    # Fitness: high accuracy + penalty for many features
    penalty = λ * (len(selected_features) / num_features)
    return acc - penalty

# Initial population config
gene_space = [0, 1]

ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=num_features,
    gene_space=gene_space,
    parent_selection_type="tournament",
    keep_parents=2,
    crossover_type="uniform",
    mutation_type="random",
    mutation_percent_genes=10,
    stop_criteria=["saturate_10"]
)

ga_instance.run()

# Summary
solution, solution_fitness, _ = ga_instance.best_solution()
selected = np.where(solution == 1)[0]
print(f"Best Feature Subset: {selected}")
print(f"Fitness (accuracy - penalty): {solution_fitness:.4f}")

# Optional plot
ga_instance.plot_fitness()
# Optional: Save the best solution
# ga_instance.save("best_solution.pickle")