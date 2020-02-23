import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
import pandas as pd


def solve_with_hillclimbing(tsp_problem, experiement_name="RHC"):
    restart_list = [5, 10, 20]
    seed = 786
    max_attempts = 4000

    rhc = mlrose_hiive.runners.rhc_runner.RHCRunner(problem=tsp_problem, experiment_name=experiement_name,
                                                    output_directory='./Data/', seed=seed,
                                                    iteration_list=2 ** np.linspace(1, 12), max_attempts=max_attempts,
                                                    restart_list=restart_list)

    run_stats, run_curves = rhc.run()

    return run_stats, run_curves