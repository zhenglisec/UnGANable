"""
Classifier API for applying all attacks. Use the :class:`.Classifier` wrapper to be able to apply an attack to a
existing model.
"""
from art1.estimators.classification.blackbox import BlackBoxClassifier
from art1.estimators.classification.catboost import CatBoostARTClassifier
from art1.estimators.classification.detector_classifier import DetectorClassifier
from art1.estimators.classification.ensemble import EnsembleClassifier
from art1.estimators.classification.GPy import GPyGaussianProcessClassifier
from art1.estimators.classification.keras import KerasClassifier
from art1.estimators.classification.lightgbm import LightGBMClassifier
from art1.estimators.classification.mxnet import MXClassifier
from art1.estimators.classification.pytorch import PyTorchClassifier
from art1.estimators.classification.scikitlearn import SklearnClassifier
from art1.estimators.classification.tensorflow import (
    TFClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
)
from art1.estimators.classification.xgboost import XGBoostClassifier
