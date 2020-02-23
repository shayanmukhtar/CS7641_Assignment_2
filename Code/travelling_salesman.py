import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
import pandas as pd
import genetic_algorithm
import simulated_annealing
import mimic
import hillclimbing
import graphs

def create_tsp_problem(length=10, width=10, cities=8):
    coords_list = []
    np.random.seed(786)

    for city in range(0, cities):
        new_city = (np.random.randint(0, length), np.random.randint(0, width))
        if new_city not in coords_list:
            coords_list.append(new_city)

    tsp_coords = mlrose_hiive.fitness.TravellingSales(coords=coords_list)
    tsp_problem = mlrose_hiive.TSPOpt(length=cities, fitness_fn=tsp_coords, maximize=True)
    return tsp_problem


def plot_tsp_grid(tsp_problem, data_string="", solution_vector=None):
    coords = tsp_problem.fitness_fn.coords
    x_coords = []
    y_coords = []

    for coord in coords:
        x_coords.append(coord[0])
        y_coords.append(coord[1])

    for i in range(len(coords)):
        plt.plot(x_coords[i], y_coords[i], 'co')
        plt.text(x_coords[i], y_coords[i], str(i))

    if solution_vector is not None:
        for i in range(0, len(solution_vector) - 1):
            x = x_coords[solution_vector[i]]
            y = y_coords[solution_vector[i]]
            dx = x_coords[solution_vector[i+1]] - x
            dy = y_coords[solution_vector[i+1]] - y

            plt.arrow(x, y, dx, dy, color='r', width=0.001/len(solution_vector), length_includes_head=True,
                      head_width=5/len(solution_vector) )

    plt.title("Solution Vector Plotted for " + data_string)
    plt.savefig(data_string + "_travelling_sales_solution_plot")
    plt.close()


def main_10_cities():
    # Create the TSP
    tsp = create_tsp_problem(length=200, width=200, cities=10)
    plot_tsp_grid(tsp, "Genetic_Algorithm")

    # Random Hillclimbing
    rhc_run_stats, rhc_run_curves = hillclimbing.solve_with_hillclimbing(tsp, "RHC_TSP")

    rhc_data_strings = {
        'title': 'RHC - 10 Cities',
        'Parameters': ['Restarts']
    }
    graphs.generate_graphs(rhc_run_stats, rhc_run_curves, rhc_data_strings)

    # Mimic

    mimic_run_stats, mimic_run_curves = mimic.solve_with_mimic(tsp, "MIMIC_TSP")
    mimic_data_strings = {
        'title': 'MIMIC - 10 Cities',
        'Parameters': ['Population Size', 'Keep Percent']
    }
    graphs.generate_graphs(mimic_run_stats, mimic_run_curves, mimic_data_strings)

    # Solve with Genetic Algorithm
    ga_run_stats, ga_run_curves = genetic_algorithm.solve_with_ga(tsp, "GA_TSP")

    ga_data_strings = {
                        'title': 'Genetic Algorithms - 10 Cities',
                        'Parameters': ['Mutation Rate', 'Population Size']
                      }
    graphs.generate_graphs(ga_run_stats, ga_run_curves, ga_data_strings)

    # Simulated Annealing
    sa_run_stats, sa_run_curves = simulated_annealing.solve_with_sim_annealing(tsp, "SA_TSP")

    sa_data_strings = {
                        'title': 'Simulated Annealing - 10 Cities',
                        'Parameters': ['Temperature']
                      }
    graphs.generate_graphs(sa_run_stats, sa_run_curves, sa_data_strings)


def main_5_cities():
    # Create the TSP
    tsp = create_tsp_problem(length=200, width=200, cities=5)
    plot_tsp_grid(tsp, "Genetic_Algorithm")

    # Random Hillclimbing
    rhc_run_stats, rhc_run_curves = hillclimbing.solve_with_hillclimbing(tsp, "RHC_TSP_5_Cities")

    rhc_data_strings = {
        'title': 'RHC - 5 Cities',
        'Parameters': ['Restarts']
    }
    graphs.generate_graphs(rhc_run_stats, rhc_run_curves, rhc_data_strings)

    # Mimic

    mimic_run_stats, mimic_run_curves = mimic.solve_with_mimic(tsp, "MIMIC_TSP_5_Cities")
    mimic_data_strings = {
        'title': 'MIMIC - 5 Cities',
        'Parameters': ['Population Size', 'Keep Percent']
    }
    graphs.generate_graphs(mimic_run_stats, mimic_run_curves, mimic_data_strings)

    # Solve with Genetic Algorithm
    ga_run_stats, ga_run_curves = genetic_algorithm.solve_with_ga(tsp, "GA_TSP_5_Cities")

    ga_data_strings = {
                        'title': 'Genetic Algorithms - 5 Cities',
                        'Parameters': ['Mutation Rate', 'Population Size']
                      }
    graphs.generate_graphs(ga_run_stats, ga_run_curves, ga_data_strings)

    # Simulated Annealing
    sa_run_stats, sa_run_curves = simulated_annealing.solve_with_sim_annealing(tsp, "SA_TSP_5_Cities")

    sa_data_strings = {
                        'title': 'Simulated Annealing - 5 Cities',
                        'Parameters': ['Temperature']
                      }
    graphs.generate_graphs(sa_run_stats, sa_run_curves, sa_data_strings)


if __name__ == '__main__':
    main_5_cities()
