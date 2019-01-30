"""
    Various tutorials on ML
   --- SVM: Support Vector Machines ---
"""

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from loguru import logger
from sklearn.neighbors import KNeighborsClassifier


def run():
    """
        launch svm example
    """

    # Load data from sklearn dataset
    cancer = datasets.load_breast_cancer()
    logger.info(f"Features: {cancer.feature_names}")
    logger.info(f"Labels: {cancer.target_names}")

    x = cancer.data  # All of the features
    y = cancer.target  # All of the labels

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.2
    )

    # logger.info(f"x_train: {x_train}")
    # logger.info(f"y_train: {y_train}")

    clf = svm.SVC(kernel="linear", C=2)  # C = soft margin
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    logger.info(f"accuracy of SVM: {accuracy}")

    # compare with KNN
    clf = KNeighborsClassifier(n_neighbors=11)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    logger.info(f"accuracy of KNN: {accuracy}")


if __name__ == "__main__":
    run()
