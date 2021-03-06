"""
    Various tutorials on ML
   --- KNN: K Nearest Neighbors ---
"""

import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

from loguru import logger


def run():
    """
        launch knn example
    """

    # data: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    logger.info("Loading /knn/car.data")
    data = pd.read_csv("./knn/car.data")

    # convert non numerical data to numerical data
    # giving numpy lists
    label_encoder = preprocessing.LabelEncoder()
    buying = label_encoder.fit_transform(list(data["buying"]))
    maint = label_encoder.fit_transform(list(data["maint"]))
    door = label_encoder.fit_transform(list(data["door"]))
    persons = label_encoder.fit_transform(list(data["persons"]))
    lug_boot = label_encoder.fit_transform(list(data["lug_boot"]))
    safety = label_encoder.fit_transform(list(data["safety"]))
    cls = label_encoder.fit_transform(list(data["class"]))

    # we'll predict the class
    predict = "class"
    x = list(zip(buying, maint, door, persons, lug_boot, safety))
    y = list(cls)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1
    )

    model = KNeighborsClassifier(n_neighbors=9)

    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    logger.info(accuracy)

    predicted = model.predict(x_test)
    names = ["unacc", "acc", "good", "vgood"]

    for x in range(len(predicted)):
        logger.info("Predicted: " + names[predicted[x]] + " => Data: " + str(x_test[x]))
        logger.info("Actual: " + names[y_test[x]])
        n = model.kneighbors([x_test[x]], 9, True)
        logger.info(f"N: {n}")


if __name__ == "__main__":
    run()
