"""
    generic launcher
"""

from loguru import logger
from linear_regression import linear_regression


def main():
    logger.info("Starting Linear Regression")
    linear_regression.run()
    logger.info("Stopping Linear Regression")


if __name__ == "__main__":
    main()
