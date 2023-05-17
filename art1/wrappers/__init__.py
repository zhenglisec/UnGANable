"""
Module providing wrappers for :class:`.Classifier` instances to simulate different capacities and behaviours, like
black-box gradient estimation.
"""
from art1.wrappers.wrapper import ClassifierWrapper
from art1.wrappers.expectation import ExpectationOverTransformations
from art1.wrappers.query_efficient_bb import QueryEfficientBBGradientEstimation
