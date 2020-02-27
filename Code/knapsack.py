import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
import pandas as pd
import genetic_algorithm
import simulated_annealing
import mimic
import hillclimbing
import graphs


def create_knapsack_problem(length):
    weights = np.random.randint(1, 20, length)
    values = np.random.randint(1, 5, length)
    fitness = mlrose_hiive.Knapsack(weights, values, max_weight_pct=1)
    knapsack = mlrose_hiive.opt_probs.discrete_opt.DiscreteOpt(length=len(weights),
                                                               fitness_fn=fitness,
                                                               maximize=True)
    return knapsack


def main_20_items():
    t_pct = 0.1
    knapsack = create_knapsack_problem(20)

    # Random Hillclimbing
    rhc_run_stats, rhc_run_curves = hillclimbing.solve_with_hillclimbing(knapsack, "RHC_Knapsack")

    rhc_data_strings = {
        'title': 'RHC - Knapsack',
        'Parameters': ['Restarts'],
        'limit_time': 0,
        'limit_iterations': 0
    }
    graphs.generate_graphs(rhc_run_stats, rhc_run_curves, rhc_data_strings)

    # Mimic

    mimic_run_stats, mimic_run_curves = mimic.solve_with_mimic(knapsack, "MIMIC_Knapsack")
    mimic_data_strings = {
        'title': 'MIMIC - Knapsack',
        'Parameters': ['Population Size', 'Keep Percent'],
        'limit_time': 10,
        'limit_iterations': 50
    }
    graphs.generate_graphs(mimic_run_stats, mimic_run_curves, mimic_data_strings)

    # Solve with Genetic Algorithm
    ga_run_stats, ga_run_curves = genetic_algorithm.solve_with_ga(knapsack, "GA_Knapsack")

    ga_data_strings = {
        'title': 'Genetic Algorithms - Knapsack',
        'Parameters': ['Mutation Rate', 'Population Size'],
        'limit_time': 11,
        'limit_iterations': 800
    }
    graphs.generate_graphs(ga_run_stats, ga_run_curves, ga_data_strings)

    # Simulated Annealing
    sa_run_stats, sa_run_curves = simulated_annealing.solve_with_sim_annealing(knapsack, "SA_Knapsack")

    sa_data_strings = {
        'title': 'Simulated Annealing - Knapsack',
        'Parameters': ['Temperature'],
        'limit_time': 0.5,
        'limit_iterations': 1500
    }
    graphs.generate_graphs(sa_run_stats, sa_run_curves, sa_data_strings)


def main_10_items():
    t_pct = 0.1
    knapsack = create_knapsack_problem(10)

    # Random Hillclimbing
    rhc_run_stats, rhc_run_curves = hillclimbing.solve_with_hillclimbing(knapsack, "RHC_Knapsack_10")

    rhc_data_strings = {
        'title': 'RHC - Knapsack 10 Items',
        'Parameters': ['Restarts']
    }
    graphs.generate_graphs(rhc_run_stats, rhc_run_curves, rhc_data_strings)

    # Mimic

    mimic_run_stats, mimic_run_curves = mimic.solve_with_mimic(knapsack, "MIMIC_Knapsack_10")
    mimic_data_strings = {
        'title': 'MIMIC - Knapsack 10 Items',
        'Parameters': ['Population Size', 'Keep Percent']
    }
    graphs.generate_graphs(mimic_run_stats, mimic_run_curves, mimic_data_strings)

    # Solve with Genetic Algorithm
    ga_run_stats, ga_run_curves = genetic_algorithm.solve_with_ga(knapsack, "GA_Knapsack_10")

    ga_data_strings = {
        'title': 'Genetic Algorithms - Knapsack 10 Items',
        'Parameters': ['Mutation Rate', 'Population Size']
    }
    graphs.generate_graphs(ga_run_stats, ga_run_curves, ga_data_strings)

    # Simulated Annealing
    sa_run_stats, sa_run_curves = simulated_annealing.solve_with_sim_annealing(knapsack, "SA_Knapsack_10")

    sa_data_strings = {
        'title': 'Simulated Annealing - Knapsack 10 Items',
        'Parameters': ['Temperature']
    }
    graphs.generate_graphs(sa_run_stats, sa_run_curves, sa_data_strings)


if __name__ == '__main__':
    main_20_items()
