import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
import pandas as pd


def solve_with_mimic(tsp_problem, experiment_name='Mimic'):
    theta_max = [0.25, 0.5, 0.75]
    population_sizes = [10, 40]
    seed = 786
    max_attempts = 1000
    mimic = mlrose_hiive.runners.mimic_runner.MIMICRunner(tsp_problem, experiment_name=experiment_name,
                                                          output_directory="./Data/", seed=seed,
                                                          iteration_list=2 ** np.linspace(1, 12),
                                                          max_attempts=max_attempts, keep_percent_list=theta_max,
                                                          population_sizes=population_sizes)

    run_stats, run_curves = mimic.run()
    return run_stats, run_curves
