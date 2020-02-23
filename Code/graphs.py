import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive
import pandas as pd


def mark_inflection_points(x):
    if x['Time'] < x['Next Time']:
        return 1
    else:
        return 0


def generate_graphs(run_stats, run_curves, data_strings):
    print(run_stats)

    # first figure out the rows that split
    run_curves["Inflections"] = 0
    run_curves["Next Time"] = run_curves["Time"].shift(periods=1)
    run_curves['Inflections'] = run_curves.apply(mark_inflection_points, axis=1)

    inflection_points = run_curves.index[run_curves['Inflections'] == 1].tolist()
    inflection_points.insert(0, 0)

    curves = []

    for i in range(len(inflection_points) - 1):
        curves.append(pd.DataFrame(run_curves.iloc[inflection_points[i]: inflection_points[i + 1] - 50],
                                   columns=run_curves.columns))

    curves.append(pd.DataFrame(run_curves.iloc[inflection_points[-1]:],
                               columns=run_curves.columns))

    plt.figure()
    for curve in curves:
        label = ""
        for attribute in data_strings['Parameters']:
            label += attribute + ": " + str(curve[attribute].iloc[0]) + ", "
        plt.plot(curve['Time'], curve['Fitness'], label=label)

    # plt.xlim(0, 1)
    plt.xlabel("Time (s)")
    plt.ylabel("Population Fitness")
    plt.legend()
    plt.savefig(data_strings['title'] + ' Time vs Fitness.png')
    plt.close()

    plt.figure()
    for curve in curves:
        label = ""
        for attribute in data_strings['Parameters']:
            label += attribute + ": " + str(curve[attribute].iloc[0]) + ", "
        plt.plot(range(len(curve['Fitness'])), curve['Fitness'], label=label)

    # plt.xlim(0, 800)
    plt.xlabel("Algorithm Iterations")
    plt.ylabel("Population Fitness")
    plt.legend()
    plt.savefig(data_strings['title'] + ' Iteration vs Fitness.png')
    plt.close()
