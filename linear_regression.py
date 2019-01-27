"""
    Various tutorials on ML
   --- Linear regression ---
"""

import pickle

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import sklearn
from matplotlib import style
from sklearn import linear_model


def main():
    """
        Main method
    """
    data = pd.read_csv("student-mat.csv", sep=";")
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

    # G = final grade
    predict = "G3"

    # x = data used for training (all by G3 columns)
    # y = target to predict (only G3 column)
    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])

    best_score = 0
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1
    )

    # train 30 time and keep the best model
    for _ in range(30):
        # train on 90% of the dataset, keep 10% for testing
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            x, y, test_size=0.1
        )

        # train with a linear model
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)

        # calculate accuracy
        accuracy = linear.score(x_test, y_test)

        if accuracy > best_score:
            print(f"New best score: {accuracy}")
            best_score = accuracy
            # save model with pickle
            with open("student_model.pickle", "wb") as pickle_file:
                pickle.dump(linear, pickle_file)

    print(f"Best Score: {best_score}")

    # load model with pickle
    pickle_in = open("student_model.pickle", "rb")
    linear = pickle.load(pickle_in)

    print(f"Coefficient = {linear.coef_}")
    print(f"Intercept   = {linear.intercept_}")
    print()

    # show examples of predicted grade vs actual grades (5 first)
    predictions = linear.predict(x_test)

    for x in range(5):
        print(f"Values: {x_test[x]}")
        print(f"Predicted: {round(predictions[x], 2):>5} => Real grade: {y_test[x]:<2}")

    # plot data with matplotlib
    vars = ["G1", "G2", "studytime", "failures", "absences"]

    style.use("ggplot")
    for var in vars:
        pyplot.scatter(data[var], data["G3"])
        pyplot.xlabel(var)
        pyplot.ylabel("Final Grade")
        pyplot.show()


if __name__ == "__main__":
    main()
