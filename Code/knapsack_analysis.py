import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
from sympy.utilities.iterables import  kbins
from itertools import combinations_with_replacement
import sympy

def create_knapsack_problem(length):
    weights = np.random.randint(1, 4, length)
    values = np.random.randint(1, 10, length)
    fitness = mlrose_hiive.Knapsack(weights, values, max_item_count=length, max_weight_pct=1)
    knapsack = mlrose_hiive.opt_probs.discrete_opt.DiscreteOpt(length=len(weights),
                                                               fitness_fn=fitness,
                                                               maximize=True)
    return knapsack

def main():
    length = 3
    knapsack = create_knapsack_problem(length)

    print("Values: " + str(knapsack.fitness_fn.values))
    print("Weights: " + str(knapsack.fitness_fn.weights))

    states = [p for p in range(length)]
    combos = combinations_with_replacement(states, length)

    final_combos = []
    for i in list(combos):
        final_combos.append(i)
        final_combos.append(i[::-1])

    fitness = []
    for i in final_combos:
        print(i)
        fitness.append(knapsack.eval_fitness(i))

    plt.bar([i for i in range(len(fitness))], fitness)
    plt.title("Knapsack Search Space")
    plt.xlabel("Permutation")
    plt.ylabel("Value")
    plt.savefig("knapsack_space.png")


if __name__ == '__main__':
    main()