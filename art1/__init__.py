"""
The Adversarial Robustness Toolbox (ART).
"""
import logging.config

# Project Imports
from art1 import attacks
from art1 import defences
from art1 import estimators
from art1 import metrics
from art1 import wrappers

# Semantic Version
__version__ = "1.6.0"

# pylint: disable=C0103

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "std": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "datefmt": "%Y-%m-%d %H:%M",}
    },
    "handlers": {
        "default": {"class": "logging.NullHandler",},
        "test": {"class": "logging.StreamHandler", "formatter": "std", "level": logging.INFO,},
    },
    "loggers": {"art1": {"handlers": ["default"]}, "tests": {"handlers": ["test"], "level": "INFO", "propagate": True},},
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)
