"""
    Various tutorials on ML
   --- Linear regression ---
"""

import pandas as pd
import numpy as np
import sklearn
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

    # train on 90% of the dataset, keep 10% fro testing
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1
    )

    # train with a linear model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # calculate accuracy
    accuracy = linear.score(x_test, y_test)

    print(f"Accuracy = {accuracy}")

    print(f"Coefficient = {linear.coef_}")
    print(f"Intercept   = {linear.intercept_}")
    print()

    predictions = linear.predict(x_test)

    for x in range(len(predictions)):
        print(f"Values: {x_test[x]}")
        print(f"Predicted: {round(predictions[x], 2):>5} => Real grade: {y_test[x]:<2}")


if __name__ == "__main__":
    main()
