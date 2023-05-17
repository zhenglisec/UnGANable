"""
Module implementing preprocessing defences against adversarial attacks.
"""
from art1.defences.preprocessor.feature_squeezing import FeatureSqueezing
from art1.defences.preprocessor.gaussian_augmentation import GaussianAugmentation
from art1.defences.preprocessor.inverse_gan import InverseGAN, DefenseGAN
from art1.defences.preprocessor.jpeg_compression import JpegCompression
from art1.defences.preprocessor.label_smoothing import LabelSmoothing
from art1.defences.preprocessor.mp3_compression import Mp3Compression
from art1.defences.preprocessor.pixel_defend import PixelDefend
from art1.defences.preprocessor.preprocessor import Preprocessor
from art1.defences.preprocessor.resample import Resample
from art1.defences.preprocessor.spatial_smoothing import SpatialSmoothing
from art1.defences.preprocessor.spatial_smoothing_pytorch import SpatialSmoothingPyTorch
from art1.defences.preprocessor.spatial_smoothing_tensorflow import SpatialSmoothingTensorFlowV2
from art1.defences.preprocessor.thermometer_encoding import ThermometerEncoding
from art1.defences.preprocessor.variance_minimization import TotalVarMin
from art1.defences.preprocessor.video_compression import VideoCompression
