"""
Logging utilities for NetRain
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(config: Dict[str, Any]):
    """
    Setup logging based on configuration.
    
    Args:
        config: Logging configuration
    """
    level = config.get('level', 'INFO')
    log_dir = config.get('log_dir', 'logs/deep_tree_echo')
    
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set up logger
    logger = logging.getLogger('netrain')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def get_logger(name: str = 'netrain') -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)