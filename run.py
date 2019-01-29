"""
    generic launcher
"""

from loguru import logger
from linear_regression import linear_regression as lr
from knn import knn


LINEAR_REGRESSION = False
KNN = True


def main():
    if LINEAR_REGRESSION:
        logger.info("Starting Linear Regression")
        lr.run()
        logger.info("Stopping Linear Regression")

    if KNN:
        logger.info("Starting KNN: K Nearest Neighbors")
        knn.run()
        logger.info("Stopping KNN: K Nearest Neighbors")


if __name__ == "__main__":
    main()
