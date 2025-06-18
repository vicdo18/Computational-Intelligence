import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

class GeneticFeatureSelector:
    def __init__(self, model, X_train, y_train, X_test, y_test, n_features):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_features = n_features
        self.best_individual = None
        self.best_fitness = -np.inf
        
    def initialize_population(self, pop_size):
        population = []
        # All features individual
        population.append(np.ones(self.n_features, dtype=int))
        # 50% features individual
        population.append(np.random.choice([0, 1], size=self.n_features, p=[0.5, 0.5]))
        # Random individuals
        for _ in range(pop_size - 2):
            individual = np.random.choice([0, 1], size=self.n_features)
            population.append(individual)
        return np.array(population)
    
    def evaluate_fitness(self, individual):
        selected_features = np.where(individual == 1)[0]
        if len(selected_features) == 0:
            return 0.0  # Penalize no features
        
        # Evaluate model with selected features
        X_train_subset = self.X_train[:, selected_features]
        X_test_subset = self.X_test[:, selected_features]
        
        # Create a new model with the same architecture but different input size
        temp_model = self._create_submodel(len(selected_features))
        
        # Transfer weights (simplification - in practice would retrain)
        temp_model.fit(X_train_subset, self.y_train, epochs=0, verbose=0)
        
        # Evaluate
        y_pred = (temp_model.predict(X_test_subset) > 0.5).astype(int)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Calculate penalty term
        feature_ratio = len(selected_features) / self.n_features
        penalty = feature_ratio * (1 - accuracy)
        
        # Calculate fitness using the new formula
        fitness = 0.8 * accuracy - 0.2 * penalty
        
        # Update best individual
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_individual = individual.copy()
            
        return fitness
    
    def _create_submodel(self, input_dim):
        # Recreate the best architecture from Part A
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(input_dim,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def tournament_selection(self, population, fitness, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            contenders = np.random.choice(len(population), tournament_size, replace=False)
            best = contenders[np.argmax(fitness[contenders])]
            selected.append(population[best])
        return np.array(selected)
    
    def uniform_crossover(self, parent1, parent2):
        mask = np.random.randint(0, 2, size=self.n_features)
        child = parent1 * mask + parent2 * (1 - mask)
        return child
    
    def mutate(self, individual, mutation_rate=0.01):
        mutations = np.random.random(size=self.n_features) < mutation_rate
        individual[mutations] = 1 - individual[mutations]
        return individual
    
    def run(self, pop_size=50, generations=100, crossover_rate=0.9, mutation_rate=0.01, 
            elitism_ratio=0.1):
        population = self.initialize_population(pop_size)
        fitness_history = []
        feature_count_history = []
        
        for gen in range(generations):
            # Evaluate fitness
            fitness = np.array([self.evaluate_fitness(ind) for ind in population])
            
            # Record stats
            fitness_history.append(np.max(fitness))
            feature_count_history.append(np.mean(np.sum(population, axis=1)))
            
            # Check termination conditions
            if gen > 20 and np.std(fitness_history[-20:]) < 0.001:
                break
                
            # Selection
            selected = self.tournament_selection(population, fitness)
            
            # Crossover
            next_pop = []
            elite_size = int(elitism_ratio * pop_size)
            elite_indices = np.argsort(fitness)[-elite_size:]
            next_pop.extend(population[elite_indices])
            
            for i in range(pop_size - elite_size):
                if np.random.random() < crossover_rate and i+1 < len(selected):
                    parent1, parent2 = selected[i], selected[i+1]
                    child = self.uniform_crossover(parent1, parent2)
                else:
                    child = selected[i].copy()
                next_pop.append(child)
            
            # Mutation
            for i in range(elite_size, pop_size):
                next_pop[i] = self.mutate(next_pop[i], mutation_rate)
            
            population = np.array(next_pop)
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'fitness_history': fitness_history,
            'feature_count_history': feature_count_history
        }