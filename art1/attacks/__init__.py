"""
Module providing adversarial attacks under a common interface.
"""
from art1.attacks.attack import Attack, EvasionAttack, PoisoningAttack, PoisoningAttackBlackBox, PoisoningAttackWhiteBox
from art1.attacks.attack import PoisoningAttackTransformer, ExtractionAttack, InferenceAttack, AttributeInferenceAttack
from art1.attacks.attack import ReconstructionAttack

from art1.attacks import evasion
from art1.attacks import extraction
from art1.attacks import inference
from art1.attacks import poisoning
