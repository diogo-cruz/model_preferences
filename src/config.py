#!/usr/bin/env python3

import yaml
import os
from typing import Dict, Any

class Config:
    """Configuration manager for the model preferences experiment."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            # Default to config.yaml in project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, 'config.yaml')
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set value
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """
        Save configuration to file.
        
        Args:
            path: Path to save to. If None, uses original config path.
        """
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, indent=2, default_flow_style=False)
    
    @property
    def experiment_name(self) -> str:
        return self.get('experiment.name', 'model_preferences')
    
    @property
    def experiment_seed(self) -> int:
        return self.get('experiment.seed', 42)
    
    @property
    def model_name(self) -> str:
        return self.get('model.name', 'gpt-4o-mini')
    
    @property
    def model_fallback(self) -> str:
        return self.get('model.fallback_name', 'gpt-4o-mini')
    
    @property
    def model_max_tokens(self) -> int:
        return self.get('model.max_tokens', 10)
    
    @property
    def model_temperature(self) -> float:
        return self.get('model.temperature', 0.0)
    
    @property
    def model_timeout(self) -> int:
        return self.get('model.timeout', 30)
    
    @property
    def task_count(self) -> int:
        return self.get('tasks.count', 100)
    
    @property
    def test_task_count(self) -> int:
        return self.get('tasks.test_count', 20)
    
    @property
    def small_test_count(self) -> int:
        return self.get('tasks.small_test_count', 5)
    
    @property
    def include_reverse_comparisons(self) -> bool:
        return self.get('comparisons.include_reverse', True)
    
    @property
    def max_test_comparisons(self) -> int:
        return self.get('comparisons.max_test_comparisons', 50)
    
    def get_project_path(self, path_key: str) -> str:
        """
        Get absolute path for project directories.
        
        Args:
            path_key: Key in paths config (e.g., 'experiments', 'logs')
            
        Returns:
            Absolute path to directory
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_path = self.get(f'paths.{path_key}', path_key)
        return os.path.join(project_root, relative_path)

# Global config instance
_config = None

def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

def reload_config():
    """Reload configuration from file."""
    global _config
    _config = None
    return get_config()