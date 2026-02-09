"""Modeling layer exports."""

from veldra.modeling.binary import BinaryTrainingOutput, train_binary_with_cv
from veldra.modeling.regression import RegressionTrainingOutput, train_regression_with_cv

__all__ = [
    "BinaryTrainingOutput",
    "RegressionTrainingOutput",
    "train_binary_with_cv",
    "train_regression_with_cv",
]
