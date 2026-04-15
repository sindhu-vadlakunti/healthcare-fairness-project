"""Configuration loader for the healthcare fairness project."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage project configuration."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent
            config_path = project_root / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with dots, e.g., 'data.raw_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent
    
    def get_data_dir(self, subdir: str = None) -> Path:
        """
        Get data directory path.
        
        Args:
            subdir: Subdirectory name (e.g., 'raw', 'processed')
            
        Returns:
            Path to data directory
        """
        project_root = self.get_project_root()
        data_dir = project_root / "data"
        
        if subdir:
            data_dir = data_dir / subdir
        
        # Create directory if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        
        return data_dir
    
    def get_models_dir(self) -> Path:
        """Get models directory path."""
        project_root = self.get_project_root()
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    def get_results_dir(self) -> Path:
        """Get results directory path."""
        project_root = self.get_project_root()
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def get_logs_dir(self) -> Path:
        """Get logs directory path."""
        project_root = self.get_project_root()
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir


# Global configuration instance
_config = None


def get_config(config_path: str = None) -> ConfigLoader:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigLoader instance
    """
    global _config
    if _config is None:
        _config = ConfigLoader(config_path)
    return _config
