"""
Module containing estimators for speech recognition.
"""
from art1.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin

from art1.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
from art1.estimators.speech_recognition.tensorflow_lingvo import TensorFlowLingvoASR
