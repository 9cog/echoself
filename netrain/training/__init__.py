"""
NetRain Training Module - Training logic for Deep Tree Echo models
"""

from .trainer import EchoTrainer
from .optimizer import create_optimizer, create_scheduler
from .metrics import MetricsTracker

__all__ = [
    "EchoTrainer",
    "create_optimizer",
    "create_scheduler",
    "MetricsTracker"
]