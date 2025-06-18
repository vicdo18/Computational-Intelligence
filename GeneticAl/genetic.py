import deap
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import seaborn as sns
import matplotlib.pyplot as plt

# Genetic Algorithm implementation using DEAP library

def initialize_population(pop_size, num_features=32):
    population = []
    
    # Άτομο με όλα τα χαρακτηριστικά
    population.append(np.ones(num_features))
    
    # Άτομα με τυχαία χαρακτηριστικά
    for _ in range(pop_size - 2):
        individual = np.random.randint(0, 2, num_features)
        population.append(individual)
    
    # Άτομο με τα πιο σημαντικά χαρακτηριστικά (π.χ. Age, MMSE, FamilyHistory) 
    important_features = np.zeros(num_features)
    important_features[0] = 1  # Age
    important_features[12] = 1 # MMSE
    important_features[16] = 1 # FamilyHistoryAlzheimers
    population.append(important_features)
    
    return np.array(population)

# Fitness function to evaluate individuals -- Fitness = w1 * accuracy + w2 * (1 - num_features_used / total_features)
def fitness_function(individual, model, X_test, y_test):
    # Επιλογή των ενεργών χαρακτηριστικών
    active_features = individual.astype(bool)
    X_test_reduced = X_test.iloc[:, active_features]
    
    # Αξιολόγηση του μοντέλου
    _, accuracy, _ = model.evaluate(X_test_reduced, y_test, verbose=0)
    
    # Υπολογισμός ποινής για πολλά χαρακτηριστικά
    num_features_used = np.sum(individual)
    feature_penalty = 1 - (num_features_used / len(individual))
    
    # Συνδυασμός των όρων (βάρη 0.7 και 0.3)
    fitness = 0.7 * accuracy + 0.3 * feature_penalty
    
    return fitness

def selection(population, fitness, method='tournament'):
    if method == 'tournament':
        # Τουρνουά μεγέθους 3
        selected = []
        for _ in range(len(population)):
            contestants = np.random.choice(len(population), 3)
            best = contestants[np.argmax(fitness[contestants])]
            selected.append(population[best])
        return np.array(selected)
    else:
        # Ρουλέτα με βάση την κατάταξη
        ranks = np.argsort(fitness)
        prob = ranks / ranks.sum()
        return population[np.random.choice(len(population), size=len(population), p=prob)]

def crossover(parent1, parent2, method='uniform'):
    if method == 'uniform':
        mask = np.random.randint(0, 2, size=len(parent1))
        child = parent1 * mask + parent2 * (1 - mask)
        return child
    else:  # single-point
        point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child

def mutation(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip the bit
    return individual