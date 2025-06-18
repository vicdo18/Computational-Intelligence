import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#load the pre-trained model from Part A
model = load_model('full_alzheimers_model.h5')

#i will be using the same csv file as the one used in Part A

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

X_test_np = X_test.values
y_test_np = y_test.values

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class GeneticFeatureSelector:
    def __init__(self, model, X_test_np, y_test_np, num_features=34, alpha=0.8, beta=0.2,
                 pop_size=30, num_generations=20, mutation_rate=0.05, elite_size=2):
        self.model = model
        self.X_test_np = X_test_np
        self.y_test_np = y_test_np
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        self.population = self._generate_initial_population()

    def _generate_individual(self):
        return np.random.choice([0, 1], size=self.num_features)

    def _generate_initial_population(self):
        return [self._generate_individual() for _ in range(self.pop_size)]

    def _evaluate_fitness(self, individual):
        selected_features = np.where(individual == 1)[0]
        if len(selected_features) == 0:
            return -np.inf  # Very low fitness for no features

        X_selected = self.X_test_np[:, selected_features]
        X_padded = np.zeros((X_selected.shape[0], self.num_features))
        X_padded[:, selected_features] = X_selected

        predictions = self.model.predict(X_padded, verbose=0).flatten()
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)  # Prevent log(0)

        loss = log_loss(self.y_test_np, predictions)  # Cross-entropy loss
        accuracy = accuracy_score(self.y_test_np, predictions > 0.5)

        num_used = len(selected_features)
        penalty = (num_used / self.num_features) * (1 - accuracy)

        fitness = 1 / (1 + self.alpha * loss + self.beta * penalty)
        return fitness

    def _tournament_selection(self, fitnesses, k=3):
        selected = np.random.choice(len(self.population), k, replace=False)
        best = selected[np.argmax([fitnesses[i] for i in selected])]
        return self.population[best]

    def _uniform_crossover(self, parent1, parent2):
        mask = np.random.rand(self.num_features) < 0.5
        return np.where(mask, parent1, parent2)

    def _mutate(self, individual):
        return np.array([gene if np.random.rand() > self.mutation_rate else 1 - gene for gene in individual])

    def _apply_elitism(self, fitnesses):
        elite_indices = np.argsort(fitnesses)[-self.elite_size:]
        return [self.population[i] for i in elite_indices]

    def run(self):
        for generation in range(self.num_generations):
            fitnesses = [self._evaluate_fitness(ind) for ind in self.population]
            new_population = self._apply_elitism(fitnesses)

            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection(fitnesses)
                parent2 = self._tournament_selection(fitnesses)
                child = self._uniform_crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            self.population = new_population
            best_fitness = max(fitnesses)
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness:.4f}")

# Run Genetic Algorithm
ga = GeneticFeatureSelector(model, X_test_np, y_test_np, num_features=X_test_np.shape[1])
ga.run()    
