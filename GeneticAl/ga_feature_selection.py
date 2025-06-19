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

class GeneticFeatureSelector:
    def __init__(self, model, X_test_np, y_test_np, num_features=32, alpha=0.8, beta=0.2,
                 pop_size=30, num_generations=20, crossover_rate=0.6, mutation_rate=0.05, elite_size=2,
                 no_improvement_limit=10, min_improvement_delta=0.01):
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
        self.no_improvement_limit = no_improvement_limit
        self.min_improvement_delta = min_improvement_delta
        self.crossover_rate = crossover_rate

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
        if np.random.rand() > self.crossover_rate:
            # Δεν γίνεται crossover, επέλεξε τυχαία έναν από τους δύο γονείς
            return parent1 if np.random.rand() < 0.5 else parent2

        # Αν γίνεται crossover, κάνε το κανονικά
        mask = np.random.rand(self.num_features) < 0.5
        return np.where(mask, parent1, parent2)

    def _mutate(self, individual):
        mutation_mask = np.random.rand(self.num_features) < self.mutation_rate
        mutated = individual.copy()
        mutated[mutation_mask] = 1 - mutated[mutation_mask]
        return mutated

    def _apply_elitism(self, fitnesses):
        elite_indices = np.argsort(fitnesses)[-self.elite_size:]
        return [self.population[i] for i in elite_indices]
    
    def run(self, verbose=False):
        best_fitness = -np.inf
        generations_no_improve = 0
        best_fitnesses = []

        for generation in range(self.num_generations):
            fitnesses = [self._evaluate_fitness(ind) for ind in self.population]
            current_best = max(fitnesses)
            best_fitnesses.append(current_best)

            if current_best - best_fitness > self.min_improvement_delta:
                best_fitness = current_best
                generations_no_improve = 0
            else:
                generations_no_improve += 1

            if verbose:
                print(f"Generation {generation + 1}, Best Fitness: {current_best:.4f}")

            if generations_no_improve >= self.no_improvement_limit:
                break

            new_population = self._apply_elitism(fitnesses)

            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection(fitnesses)
                parent2 = self._tournament_selection(fitnesses)
                child = self._uniform_crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            self.population = new_population

        return best_fitness, generation + 1


def evaluate_setting(model, X_test_np, y_test_np, pop_size, mutation_rate, crossover_prob, runs=10):
    fitness_scores = []
    generation_counts = []
    all_fitness_curves = []

    for _ in range(runs):
        ga = GeneticFeatureSelector(
            model=model,
            X_test_np=X_test_np,
            y_test_np=y_test_np,
            num_features=X_test_np.shape[1],  
            num_generations=100,
            pop_size=30,
            mutation_rate=0.05,
            crossover_rate=0.6,
            elite_size=2,
            alpha=1.0,
            beta=0.2,
            no_improvement_limit=10,
            min_improvement_delta=0.01
        )

        best_fitness, total_generations = ga.run(verbose=True)
        fitness_curve = []
        best_fitness = -np.inf
        generations_no_improve = 0

        for generation in range(ga.num_generations):
            fitnesses = [ga._evaluate_fitness(ind) for ind in ga.population]
            current_best = max(fitnesses)
            fitness_curve.append(current_best)

            if current_best - best_fitness > ga.min_improvement_delta:
                best_fitness = current_best
                generations_no_improve = 0
            else:
                generations_no_improve += 1

            if generations_no_improve >= ga.no_improvement_limit:
                break

            new_population = ga._apply_elitism(fitnesses)

            while len(new_population) < ga.pop_size:
                parent1 = ga._tournament_selection(fitnesses)
                parent2 = ga._tournament_selection(fitnesses)
                child = ga._uniform_crossover(parent1, parent2)
                child = ga._mutate(child)
                new_population.append(child)

            ga.population = new_population

        fitness_scores.append(best_fitness)
        generation_counts.append(len(fitness_curve))
        all_fitness_curves.append(fitness_curve)

    # Make all curves the same length by padding with last value (??)
    max_len = max(len(curve) for curve in all_fitness_curves)
    for i in range(len(all_fitness_curves)):
        last_val = all_fitness_curves[i][-1]
        while len(all_fitness_curves[i]) < max_len:
            all_fitness_curves[i].append(last_val)

    # Average the curves
    mean_fitness_curve = np.mean(all_fitness_curves, axis=0)

    return np.mean(fitness_scores), np.mean(generation_counts), mean_fitness_curve

configs = [          #final - best results 
    (200, 0.6, 0.01),
]

# configs = [
#     (20, 0.6, 0.00),
#     (20, 0.6, 0.01),
#     (20, 0.6, 0.10),
#     (20, 0.9, 0.01),
#     (20, 0.1, 0.01),
#     (200, 0.6, 0.00),
#     (200, 0.6, 0.01),
#     (200, 0.6, 0.10),
#     (200, 0.9, 0.01),
#     (200, 0.1, 0.01),
# ]

print("\nΑποτελέσματα:\n")
print("| Α/Α | Πληθυσμός | Διασταύρωση | Μετάλλαξη | Μέσος Fitness | Μέσος Γενεών |")
print("|-----|------------|------------|-------------|----------------|----------------|")

# Store all averaged fitness curves for plotting
all_mean_curves = []

for i, (pop_size, crossover_rate, mutation_rate) in enumerate(configs, start=1):
    mean_fit, mean_gen, mean_curve = evaluate_setting(
        model,
        X_test_np,
        y_test_np,
        pop_size,
        mutation_rate,
        crossover_rate,
        runs=10
    )
    print(f"| {i:<3} | {pop_size:<10} | {crossover_rate:<12.2f} | {mutation_rate:<11.2f} | {mean_fit:<14.4f} | {mean_gen:<14.2f} |")
    all_mean_curves.append((f"{i}: P{pop_size} C{crossover_rate} M{mutation_rate}", mean_curve))

# Plot curves
plt.figure(figsize=(12, 6))
for label, curve in all_mean_curves:
    plt.plot(curve, label=label)
plt.xlabel("Γενιές (Generations)")
plt.ylabel("Μέσος Βέλτιστος Fitness (Mean Best Fitness)")
plt.title("Καμπύλη Εξέλιξης για Κάθε Διαμόρφωση")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

#save mean curves with pickle 
import pickle
with open('mean_fitness_curves.pkl', 'wb') as f:
    pickle.dump(all_mean_curves, f)


# Optionally run a single GA run:
# ga = GeneticFeatureSelector(model, X_test_np, y_test_np, num_features=X_test_np.shape[1])
# ga.run()

