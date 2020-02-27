import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
import pandas as pd
import genetic_algorithm
import simulated_annealing
import mimic
import hillclimbing
import graphs


def create_six_peeks_problem(length=40, t_pct=0.2):
    problem = mlrose_hiive.fitness.six_peaks.SixPeaks(t_pct)
    fitness = mlrose_hiive.SixPeaks(t_pct)
    six_peaks = mlrose_hiive.opt_probs.discrete_opt.DiscreteOpt(length=length,
                                                                fitness_fn=fitness,
                                                                maximize=True)
    return six_peaks


def main_20_items():
    t_pct = 0.1
    six_peaks = create_six_peeks_problem(20, 0.1)

    # Random Hillclimbing
    rhc_run_stats, rhc_run_curves = hillclimbing.solve_with_hillclimbing(six_peaks, "RHC_6P")

    rhc_data_strings = {
        'title': 'RHC - 6 Peaks',
        'Parameters': ['Restarts'],
        'limit_time': 0,
        'limit_iterations': 0
    }
    graphs.generate_graphs(rhc_run_stats, rhc_run_curves, rhc_data_strings)

    # Mimic

    mimic_run_stats, mimic_run_curves = mimic.solve_with_mimic(six_peaks, "MIMIC_6P")
    mimic_data_strings = {
        'title': 'MIMIC - 6 Peaks',
        'Parameters': ['Population Size', 'Keep Percent'],
        'limit_time': 10,
        'limit_iterations': 100
    }
    graphs.generate_graphs(mimic_run_stats, mimic_run_curves, mimic_data_strings)

    # Solve with Genetic Algorithm
    ga_run_stats, ga_run_curves = genetic_algorithm.solve_with_ga(six_peaks, "GA_6P")

    ga_data_strings = {
        'title': 'Genetic Algorithms - 6 Peaks',
        'Parameters': ['Mutation Rate', 'Population Size'],
        'limit_time': 0.35,
        'limit_iterations': 100
    }
    graphs.generate_graphs(ga_run_stats, ga_run_curves, ga_data_strings)

    # Simulated Annealing
    sa_run_stats, sa_run_curves = simulated_annealing.solve_with_sim_annealing(six_peaks, "SA_6P")

    sa_data_strings = {
        'title': 'Simulated Annealing - 6 Peaks',
        'Parameters': ['Temperature'],
        'limit_time': 0.3,
        'limit_iterations': 1500
    }
    graphs.generate_graphs(sa_run_stats, sa_run_curves, sa_data_strings)


def main_10_items():
    t_pct = 0.1
    six_peaks = create_six_peeks_problem(10, 0.1)

    # Random Hillclimbing
    rhc_run_stats, rhc_run_curves = hillclimbing.solve_with_hillclimbing(six_peaks, "RHC_6P_Length_10")

    rhc_data_strings = {
        'title': 'RHC - 6 Peaks - Length 10',
        'Parameters': ['Restarts']
    }
    graphs.generate_graphs(rhc_run_stats, rhc_run_curves, rhc_data_strings)

    # Mimic

    mimic_run_stats, mimic_run_curves = mimic.solve_with_mimic(six_peaks, "MIMIC_6P_Length_10")
    mimic_data_strings = {
        'title': 'MIMIC - 6 Peaks - Length 10',
        'Parameters': ['Population Size', 'Keep Percent']
    }
    graphs.generate_graphs(mimic_run_stats, mimic_run_curves, mimic_data_strings)

    # Solve with Genetic Algorithm
    ga_run_stats, ga_run_curves = genetic_algorithm.solve_with_ga(six_peaks, "GA_6P_Length_10")

    ga_data_strings = {
        'title': 'Genetic Algorithms - 6 Peaks - Length 10',
        'Parameters': ['Mutation Rate', 'Population Size']
    }
    graphs.generate_graphs(ga_run_stats, ga_run_curves, ga_data_strings)

    # Simulated Annealing
    sa_run_stats, sa_run_curves = simulated_annealing.solve_with_sim_annealing(six_peaks, "SA_6P_Length_10")

    sa_data_strings = {
        'title': 'Simulated Annealing - 6 Peaks - Length 10',
        'Parameters': ['Temperature']
    }
    graphs.generate_graphs(sa_run_stats, sa_run_curves, sa_data_strings)

if __name__ == '__main__':
    main_10_items()