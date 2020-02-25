import numpy as np
import mlrose_hiive
import pandas as pd
import process_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
import evaluate_model_learning_complexity
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def gradient_descent_benchmark():
    x_data, y_data = process_dataset.process_census_data()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.6, random_state=86)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    ann = mlrose_hiive.NeuralNetwork(hidden_nodes=[32, 32], activation='relu', algorithm='gradient_descent',
                                     max_iters=1000, bias=True, is_classifier=True, learning_rate=0.00001,
                                     max_attempts=200, early_stopping=True, curve=True, random_state=1)

    ann.fit(x_train_scaled, y_train)

    # predictions = ann.predict(x_test_scaled)
    score = accuracy_score(y_test, ann.predict(x_test_scaled))
    print("accuracy was: " + str(score))

    # figure = evaluate_model_learning_complexity.plot_learning_curve(ann, "DONOTUSE.png", x_train_scaled, y_train_hot)
    # figure.savefig("ANN_Gradient_Descent.png")

    plt.figure()
    plt.plot(ann.fitness_curve)
    plt.show()




if __name__ == '__main__':
    gradient_descent_benchmark()