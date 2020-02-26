import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations


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


def main():
    length = 5
    tsp = create_tsp_problem(100, 100, length)
    cities = [a for a in range(length)]
    permutations = [p for p in multiset_permutations(cities)]

    print(permutations)

    fitness = []
    for p in permutations:
        fitness.append(tsp.eval_fitness(p))

    plt.bar([a for a in range(len(fitness))], fitness)
    plt.xlabel("Permutation")
    plt.ylabel("Euclidean Distance Travelled")
    plt.title("Distance of all Permutations of a TSP")
    plt.savefig("tsp_all_permutations.png")

    for p in range(len(permutations)):
        print("Permutation:\t" + str(permutations[p]) + "\tFitness:\t" + str(fitness[p]))

    print(str(np.max(fitness)))


if __name__ == '__main__':
    main()