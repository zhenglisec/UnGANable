"""
This module contains the Estimator API.
"""
from art1.estimators.estimator import (
    BaseEstimator,
    LossGradientsMixin,
    NeuralNetworkMixin,
    DecisionTreeMixin,
)

from art1.estimators.keras import KerasEstimator
from art1.estimators.mxnet import MXEstimator
from art1.estimators.pytorch import PyTorchEstimator
from art1.estimators.scikitlearn import ScikitlearnEstimator
from art1.estimators.tensorflow import TensorFlowEstimator, TensorFlowV2Estimator

from art1.estimators import certification
from art1.estimators import classification
from art1.estimators import encoding
from art1.estimators import generation
from art1.estimators import object_detection
from art1.estimators import poison_mitigation
from art1.estimators import regression
from art1.estimators import speech_recognition
