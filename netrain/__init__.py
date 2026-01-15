"""
NetRain - Advanced Neural Training Framework for Deep Tree Echo LLMs

A comprehensive training system for hierarchical recursive language models
with echo architectures and tree-structured attention mechanisms.
"""

__version__ = "1.0.0"
__author__ = "NetRain Team"

from .models import DeepTreeEchoTransformer
from .training import EchoTrainer
from .data import EchoDataLoader
from .utils import ConfigManager

__all__ = [
    "DeepTreeEchoTransformer",
    "EchoTrainer",
    "EchoDataLoader",
    "ConfigManager"
]