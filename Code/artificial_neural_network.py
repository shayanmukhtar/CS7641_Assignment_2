import numpy as np
import mlrose_hiive
import pandas as pd
import process_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
import evaluate_model_learning_complexity
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix, f1_score
import matplotlib.pyplot as plt


def main():
    x_data, y_data = process_dataset.process_census_data()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.6, random_state=86)

    ann = mlrose_hiive.NeuralNetwork(hidden_nodes=[32, 32], activation='relu', algorithm='gradient_descent',
                                     max_iters=200, bias=True, is_classifier=True, learning_rate=0.001,
                                     max_attempts=200, early_stopping=False)

    # ann.fit(x_train, y_train)

    # check training set predictions
    # y_train_pred = ann.predict(x_train)

    # training_accuracy = accuracy_score(y_train, y_train_pred)
    # print("Training Accuracy: " + str(training_accuracy))



    figure = evaluate_model_learning_complexity.plot_learning_curve(ann, "ANN - Gradient Descent", x_train, y_train)
    figure.savefig("ANN_Gradient_Descent.png")


def ann_weights_genetic_alg():
    x_data, y_data = process_dataset.process_census_data()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.6, random_state=86)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    onehot = OneHotEncoder()

    parameters = [
        {'algorithm': ['genetic_alg'], 'hidden_nodes': [[32, 32]],
         'max_iters': [200],
         'pop_size': [50, 100], 'random_state':[85],
         'mutation_prob': [0.1, 0.3], 'learning_rate': [0.00001]}
    ]

    grid_searcher = GridSearchCV(mlrose_hiive.NeuralNetwork(), parameters)

    grid_searcher.fit(x_train_scaled, y_train)

    # form a 2d list of your data
    report = [["Parameters", "Mean Fit Time", "Std Dev Fit Time", "Split 0 Score", "Split 1 Score", "Split 2 Score",
               "Split 3 Score", "Split 4 Score"]]
    for row in range(0, len(grid_searcher.cv_results_['params'])):
        row_data = [str(grid_searcher.cv_results_['params'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['mean_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['std_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split0_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split1_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split2_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split3_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split4_test_score'][row]),
                    ]
        report.append(row_data)

    with open('./ANN_Genetic_Alg.txt', 'w+') as fileOut:
        # print dictionary of scores
        fileOut.write("Grid Search Report\n")
        fileOut.writelines("\n")

        col_width = max(len(word) for row in report for word in row) + 2  # padding
        for row in report:
            fileOut.write("".join(word.ljust(col_width) for word in row) + "\n")

    # plot the learning curve
    title = "Census Data" + " ANN - " + str(grid_searcher.best_params_)
    figure = evaluate_model_learning_complexity.plot_learning_curve(grid_searcher.best_estimator_, title,
                                                                    x_train_scaled, y_train)
    figure.savefig("ANN_Genetic.png")

    # plot a confusion matrix as well
    plt.figure()
    cm = plot_confusion_matrix(grid_searcher.best_estimator_, x_test_scaled, y_test, display_labels=['<=$50K', '>$50K'])
    cm.plot()
    plt.savefig("Confusion_Matrix_Genetic.png")

    # append the F1 score to the output file as well
    f1score = f1_score(y_test, grid_searcher.best_estimator_.predict(x_test_scaled))
    with open('./ANN_Genetic_Alg.txt', 'a') as fileOut:
        fileOut.write("F1 Score was: " + str(f1score))


def ann_weights_simulate_annealing():
    x_data, y_data = process_dataset.process_census_data()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.6, random_state=86)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    onehot = OneHotEncoder()

    # for simulated annealing only, define some decay schedules
    annealing_sched_a = mlrose_hiive.GeomDecay(init_temp=1, decay=0.99, min_temp=0.001)
    annealing_sched_b = mlrose_hiive.GeomDecay(init_temp=1, decay=0.99, min_temp=0.1)
    annealing_sched_c = mlrose_hiive.GeomDecay(init_temp=1, decay=0.9999, min_temp=0.001)

    parameters = [
        {'algorithm': ['simulated_annealing'], 'hidden_nodes': [[32, 32]],
         'max_iters': [200, 1000],
         'learning_rate': [0.000001], 'random_state':[85],
         'schedule': [annealing_sched_a, annealing_sched_b, annealing_sched_c]}
    ]

    grid_searcher = GridSearchCV(mlrose_hiive.NeuralNetwork(), parameters)

    grid_searcher.fit(x_train_scaled, y_train)

    # form a 2d list of your data
    report = [["Parameters", "Mean Fit Time", "Std Dev Fit Time", "Split 0 Score", "Split 1 Score", "Split 2 Score", "Split 3 Score", "Split 4 Score"]]
    for row in range(0, len(grid_searcher.cv_results_['params'])):
        row_data = [str(grid_searcher.cv_results_['params'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['mean_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['std_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split0_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split1_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split2_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split3_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split4_test_score'][row]),
                    ]
        report.append(row_data)

    with open('./ANN_Simulated_Annealing.txt', 'w+') as fileOut:
        # print dictionary of scores
        fileOut.write("Grid Search Report\n")
        fileOut.writelines("\n")

        col_width = max(len(word) for row in report for word in row) + 2  # padding
        for row in report:
            fileOut.write("".join(word.ljust(col_width) for word in row) + "\n")

    # plot the learning curve
    title = "Census Data" + " ANN - " + str(grid_searcher.best_params_)
    figure = evaluate_model_learning_complexity.plot_learning_curve(grid_searcher.best_estimator_, title,
                                                                    x_train_scaled, y_train)
    figure.savefig("ANN_Simulated_Annealing.png")

    # plot a confusion matrix as well
    plt.figure()
    cm = plot_confusion_matrix(grid_searcher.best_estimator_, x_test_scaled, y_test, display_labels=['<=$50K', '>$50K'])
    cm.plot()
    plt.savefig("Confusion_Matrix_Simulated_Annealing.png")

    # append the F1 score to the output file as well
    f1score = f1_score(y_test, grid_searcher.best_estimator_.predict(x_test_scaled))
    with open('./ANN_Simulated_Annealing.txt', 'a') as fileOut:
        fileOut.write("F1 Score was: " + str(f1score))


def ann_weights_randomized_hillclimbing():
    x_data, y_data = process_dataset.process_census_data()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.6, random_state=86)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    onehot = OneHotEncoder()

    parameters = [
        {'algorithm': ['random_hill_climb'], 'hidden_nodes': [[32, 32]],
         'max_iters': [200, 500], 'random_state':[85],
         'restarts': [10, 20], 'learning_rate': [0.000001]}
    ]

    grid_searcher = GridSearchCV(mlrose_hiive.NeuralNetwork(), parameters)

    grid_searcher.fit(x_train_scaled, y_train)

    # form a 2d list of your data
    report = [["Parameters", "Mean Fit Time", "Std Dev Fit Time", "Split 0 Score", "Split 1 Score", "Split 2 Score", "Split 3 Score", "Split 4 Score"]]
    for row in range(0, len(grid_searcher.cv_results_['params'])):
        row_data = [str(grid_searcher.cv_results_['params'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['mean_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['std_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split0_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split1_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split2_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split3_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split4_test_score'][row]),
                    ]
        report.append(row_data)

    with open('./ANN_Random_Hillclimbing.txt', 'w+') as fileOut:
        # print dictionary of scores
        fileOut.write("Grid Search Report\n")
        fileOut.writelines("\n")

        col_width = max(len(word) for row in report for word in row) + 2  # padding
        for row in report:
            fileOut.write("".join(word.ljust(col_width) for word in row) + "\n")

    # plot the learning curve
    title = "Census Data" + " ANN - " + str(grid_searcher.best_params_)
    figure = evaluate_model_learning_complexity.plot_learning_curve(grid_searcher.best_estimator_, title,
                                                                    x_train_scaled, y_train)
    figure.savefig("ANN_Random_Hillclimbing.png")

    # plot a confusion matrix as well
    plt.figure()
    cm = plot_confusion_matrix(grid_searcher.best_estimator_, x_test_scaled, y_test, display_labels=['<=$50K', '>$50K'])
    cm.plot()
    plt.savefig("Confusion_Matrix_RHC.png")

    # append the F1 score to the output file as well
    f1score = f1_score(y_test, grid_searcher.best_estimator_.predict(x_test_scaled))
    with open('./ANN_Random_Hillclimbing.txt', 'a') as fileOut:
        fileOut.write("F1 Score was: " + str(f1score))


def ann_weights_gradient_descent():
    x_data, y_data = process_dataset.process_census_data()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.6, random_state=86)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    onehot = OneHotEncoder()

    y_train_hot = onehot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test_hot = onehot.fit_transform(y_test.reshape(-1, 1)).todense()

    parameters = [
        {'algorithm': ['gradient_descent'], 'hidden_nodes': [[32, 32]],
         'max_iters': [500], 'random_state':[85],
         'learning_rate': [0.00001]}
    ]

    grid_searcher = GridSearchCV(mlrose_hiive.NeuralNetwork(), parameters)

    grid_searcher.fit(x_train_scaled, y_train)

    # form a 2d list of your data
    report = [["Parameters", "Mean Fit Time", "Std Dev Fit Time", "Split 0 Score", "Split 1 Score", "Split 2 Score", "Split 3 Score", "Split 4 Score"]]
    for row in range(0, len(grid_searcher.cv_results_['params'])):
        row_data = [str(grid_searcher.cv_results_['params'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['mean_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['std_fit_time'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split0_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split1_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split2_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split3_test_score'][row]),
                    '{:02.4f}'.format(grid_searcher.cv_results_['split4_test_score'][row]),
                    ]
        report.append(row_data)

    with open('./Gradient_Descent.txt', 'w+') as fileOut:
        # print dictionary of scores
        fileOut.write("Grid Search Report\n")
        fileOut.writelines("\n")

        col_width = max(len(word) for row in report for word in row) + 2  # padding
        for row in report:
            fileOut.write("".join(word.ljust(col_width) for word in row) + "\n")

    # plot the learning curve
    title = "Census Data" + " ANN - " + str(grid_searcher.best_params_)
    figure = evaluate_model_learning_complexity.plot_learning_curve(grid_searcher.best_estimator_, title,
                                                                    x_train_scaled, y_train)
    figure.savefig("ANN_Gradient_Descent.png")

    # plot a confusion matrix as well
    plt.figure()
    cm = plot_confusion_matrix(grid_searcher.best_estimator_, x_test_scaled, y_test, display_labels=['<=$50K', '>$50K'])
    cm.plot()
    plt.savefig("Confusion_Matrix_Gradient_Descent.png")

    # append the F1 score to the output file as well
    f1score = f1_score(y_test, grid_searcher.best_estimator_.predict(x_test_scaled))
    with open('./Gradient_Descent.txt', 'a') as fileOut:
        fileOut.write("F1 Score was: " + str(f1score))


if __name__ == '__main__':
    print("Starting simulated annealing")
    ann_weights_simulate_annealing()
    print("Starting genetic algorithm")
    # ann_weights_genetic_alg()
    print("Starting random hillclimbing")
    ann_weights_randomized_hillclimbing()
    print("Starting gradient descent")
    # ann_weights_gradient_descent()
    # main()
