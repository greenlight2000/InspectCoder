"""
Configuration file for InspectCoder project paths.
This file centralizes all external dependencies, model paths, and environment configurations.
Loads configuration from config.yml file.
"""

import os
import yaml
from pathlib import Path

class PathConfig:
    """Centralized path configuration for InspectCoder project."""
    
    def __init__(self):
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        config_file = Path(__file__).parent / "config.yml"
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get(self, *keys):
        """Get configuration value by key path."""
        value = self._config
        for key in keys:
            value = value[key]
        return value

# Create a global instance for easy access
config = PathConfig()
