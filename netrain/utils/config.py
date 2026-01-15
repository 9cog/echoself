"""
Configuration management for NetRain
"""

from typing import Dict, Any, Optional
import yaml
import json
from pathlib import Path


class ConfigManager:
    """
    Manage configuration for Deep Tree Echo training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()
        self._setup_derived_configs()
    
    def _validate_config(self):
        """Validate configuration structure."""
        required_keys = ['model', 'data', 'training']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Validate model configuration
        if 'architecture' not in self.config['model']:
            raise ValueError("Missing model architecture configuration")
        
        # Validate training configuration
        if 'optimizer' not in self.config['training']:
            raise ValueError("Missing optimizer configuration")
    
    def _setup_derived_configs(self):
        """Setup derived configuration values."""
        # Ensure all sub-configs exist
        if 'echo_training' not in self.config:
            self.config['echo_training'] = {}
        
        if 'hardware' not in self.config:
            self.config['hardware'] = {'device': 'cuda'}
        
        if 'logging' not in self.config:
            self.config['logging'] = {'level': 'INFO'}
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config['model']
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config['data']
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config['training']
    
    @property
    def echo_training_config(self) -> Dict[str, Any]:
        """Get echo training configuration."""
        return self.config.get('echo_training', {})
    
    @property
    def hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration."""
        return self.config.get('hardware', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key (e.g., 'model.architecture.n_layers')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value by dot-separated key.
        
        Args:
            key: Dot-separated configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str):
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        path = Path(path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def from_file(cls, path: str) -> 'ConfigManager':
        """
        Load configuration from file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            ConfigManager instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls(config)
    
    def merge(self, other_config: Dict[str, Any]):
        """
        Merge another configuration into this one.
        
        Args:
            other_config: Configuration to merge
        """
        def deep_merge(base: dict, update: dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(self.config, other_config)
    
    def __repr__(self) -> str:
        return f"ConfigManager({list(self.config.keys())})"