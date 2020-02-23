import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
import pandas as pd


def solve_with_sim_annealing(tsp_problem, experiment_name="SA"):
    # best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(tsp_problem, random_state=0)
    temperatures = [1, 10, 100, 500, 1000, 2000, 5000]
    max_attempts = 5000
    seed = 786

    sa = mlrose_hiive.runners.sa_runner.SARunner(problem=tsp_problem, experiment_name=experiment_name,
                                                 output_directory="./Data/", seed=seed,
                                                 iteration_list=2 ** np.linspace(1, 12), max_attempts=max_attempts,
                                                 temperature_list=temperatures)

    run_stats, run_curves = sa.run()
    return run_stats, run_curves
