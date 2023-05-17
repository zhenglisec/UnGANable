"""
Module containing estimators for object detection.
"""
from art1.estimators.object_detection.object_detector import ObjectDetectorMixin

from art1.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
from art1.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
