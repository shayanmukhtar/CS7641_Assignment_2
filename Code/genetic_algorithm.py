import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
import pandas as pd


def solve_with_ga(tsp_problem, experiment_name="GA"):
    # run a grid search on genetic algorithms
    # best_state, best_fitness, curve = mlrose_hiive.genetic_alg(problem=tsp_problem, random_state=0)
    # define list of hyper-parameters and values
    population_sizes = [10, 50]
    mutation_rates = [0.1, 0.3, 0.5]
    max_attempts = 500
    seed = 786

    ga = mlrose_hiive.runners.ga_runner.GARunner(tsp_problem, experiment_name=experiment_name,
                                                 output_directory="./Data/", seed=seed,
                                                 iteration_list=2 ** np.linspace(1, 12), max_attempts=max_attempts,
                                                 population_sizes=population_sizes, mutation_rates=mutation_rates)

    run_stats, run_curves = ga.run()
    # print("Evaluaiting with Genetic Algorithms")
    # print("Coordintates of cities: " + str(tsp_problem.fitness_fn.coords))
    # print('Best state: ' + str(best_state))
    # print('Best Fitness: ' + str(best_fitness))
    # print("\n\n")

    return run_stats, run_curves
