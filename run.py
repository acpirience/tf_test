"""
    generic launcher
"""

from loguru import logger
from linear_regression import linear_regression as lr
from knn import knn
from svm import svm


LINEAR_REGRESSION = False
KNN = False
SVM = True


def main():
    if LINEAR_REGRESSION:
        logger.info("Starting Linear Regression")
        lr.run()
        logger.info("Stopping Linear Regression")

    if KNN:
        logger.info("Starting KNN: K Nearest Neighbors")
        knn.run()
        logger.info("Stopping KNN: K Nearest Neighbors")

    if SVM:
        logger.info("Starting SVM: Support Vector Machines")
        svm.run()
        logger.info("Stopping SVM: Support Vector Machines")


if __name__ == "__main__":
    main()
