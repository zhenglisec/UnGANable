"""
Module providing evasion attacks under a common interface.
"""
from art1.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from art1.attacks.evasion.adversarial_patch.adversarial_patch_numpy import AdversarialPatchNumpy
from art1.attacks.evasion.adversarial_patch.adversarial_patch_tensorflow import AdversarialPatchTensorFlowV2
from art1.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
from art1.attacks.evasion.adversarial_asr import CarliniWagnerASR
from art1.attacks.evasion.auto_attack import AutoAttack
from art1.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art1.attacks.evasion.brendel_bethge import BrendelBethgeAttack
from art1.attacks.evasion.boundary import BoundaryAttack
from art1.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod
from art1.attacks.evasion.decision_tree_attack import DecisionTreeAttack
from art1.attacks.evasion.deepfool import DeepFool
from art1.attacks.evasion.dpatch import DPatch
from art1.attacks.evasion.dpatch_robust import RobustDPatch
from art1.attacks.evasion.elastic_net import ElasticNet
from art1.attacks.evasion.fast_gradient import FastGradientMethod
from art1.attacks.evasion.frame_saliency import FrameSaliencyAttack
from art1.attacks.evasion.feature_adversaries import FeatureAdversaries
from art1.attacks.evasion.hclu import HighConfidenceLowUncertainty
from art1.attacks.evasion.hop_skip_jump import HopSkipJump
from art1.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR
from art1.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch
from art1.attacks.evasion.iterative_method import BasicIterativeMethod
from art1.attacks.evasion.newtonfool import NewtonFool
from art1.attacks.evasion.pixel_threshold import PixelAttack
from art1.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art1.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)
from art1.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch,
)
from art1.attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import (
    ProjectedGradientDescentTensorFlowV2,
)
from art1.attacks.evasion.saliency_map import SaliencyMapMethod
from art1.attacks.evasion.shadow_attack import ShadowAttack
from art1.attacks.evasion.shapeshifter import ShapeShifter
from art1.attacks.evasion.simba import SimBA
from art1.attacks.evasion.spatial_transformation import SpatialTransformation
from art1.attacks.evasion.square_attack import SquareAttack
from art1.attacks.evasion.pixel_threshold import ThresholdAttack
from art1.attacks.evasion.universal_perturbation import UniversalPerturbation
from art1.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation
from art1.attacks.evasion.virtual_adversarial import VirtualAdversarialMethod
from art1.attacks.evasion.wasserstein import Wasserstein
from art1.attacks.evasion.zoo import ZooAttack
