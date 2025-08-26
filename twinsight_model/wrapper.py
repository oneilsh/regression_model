"""
CoxModelWrapper - Simplified API for Cox regression modeling

This wrapper provides a clean, simplified interface around the existing
twinsight_model functionality while maintaining backward compatibility.
"""

import os
import pickle
import yaml
from typing import Dict, List, Union, Any, Optional
import pandas as pd
import logging

# Import existing functionality
from .dataloader_cox import load_configuration, load_data_from_bigquery
from .preprocessing_cox import create_preprocessor, apply_preprocessing
from .model_cox import CoxPHFitter, evaluate_cox_model, run_end_to_end_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoxModelWrapper:
    """
    Simplified wrapper for Cox proportional hazards modeling on All of Us data.
    
    This class provides a clean API that wraps the existing twinsight_model
    functionality, making it easier to use while maintaining full compatibility.
    """
    
    def __init__(self, config: Union[str, Dict]):
        """
        Initialize the Cox model wrapper.
        
        Args:
            config: Either a path to a YAML config file or a config dictionary
        """
        logger.info("Initializing CoxModelWrapper v0.2.0")
        
        # Load configuration
        if isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(f"Configuration file not found: {config}")
            self.config = load_configuration(config)
            self.config_source = f"file: {config}"
        elif isinstance(config, dict):
            self.config = config.copy()
            self.config_source = "dictionary"
        else:
            raise TypeError("config must be a file path (str) or dictionary")
        
        # Initialize state
        self.data = None
        self.model = None
        self.preprocessor = None
        self.training_stats = None
        self.data_summary = None
        
        # Extract key config info
        self.outcome_name = self.config.get('outcome', {}).get('name', 'unknown')
        self.feature_names = self.config.get('model_features_final', [])
        self.duration_col = self.config.get('model_io_columns', {}).get('duration_col', 'time_to_event_days')
        self.event_col = self.config.get('model_io_columns', {}).get('event_col', 'event_observed')
        
        logger.info(f"Configuration loaded from {self.config_source}")
        logger.info(f"Outcome: {self.outcome_name}")
        logger.info(f"Features: {len(self.feature_names)} features configured")
    
    def load_data(self):
        """
        Load and prepare data according to the configuration.
        
        This wraps the existing dataloader_cox.load_data_from_bigquery functionality.
        """
        logger.info("Loading data from BigQuery...")
        
        try:
            self.data = load_data_from_bigquery(self.config)
            logger.info(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            
            # Generate basic data summary (for later use)
            self._generate_data_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _generate_data_summary(self):
        """
        Generate a summary of the loaded data (internal method).
        """
        if self.data is None:
            logger.warning("No data loaded yet")
            return
        
        summary = {
            'total_patients': len(self.data),
            'total_features': len(self.feature_names),
            'outcome_name': self.outcome_name,
            'data_shape': self.data.shape,
            'config_source': self.config_source
        }
        
        # Add basic outcome statistics if available
        if self.event_col in self.data.columns:
            summary['events_observed'] = int(self.data[self.event_col].sum())
            summary['events_censored'] = int(len(self.data) - self.data[self.event_col].sum())
        
        self.data_summary = summary
        logger.info(f"Data summary generated: {summary['total_patients']} patients, {summary['events_observed'] if 'events_observed' in summary else 'unknown'} events")
    
    def train_data_summary(self) -> Dict:
        """
        Return a summary of the training data.
        
        Returns:
            Dict containing data summary information
        """
        if self.data_summary is None:
            if self.data is not None:
                self._generate_data_summary()
            else:
                raise RuntimeError("No data loaded. Call load_data() first.")
        
        return self.data_summary.copy()
    
    def train(self, split: float = 0.8, **model_kwargs):
        """
        Train the Cox proportional hazards model.
        
        Args:
            split: Train/test split ratio (default 0.8)
            **model_kwargs: Additional arguments for CoxPHFitter (e.g., penalizer, l1_ratio)
        
        Returns:
            bool: True if training successful
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        logger.info(f"Training Cox PH model with {split:.1%} train split...")
        logger.info(f"Model parameters: {model_kwargs}")
        
        try:
            # Use existing end-to-end pipeline, but capture the results
            test_size = 1.0 - split
            
            # For now, use a simplified approach - we'll enhance this later
            self.model, self.training_stats, model_path, preprocessor_path = run_end_to_end_pipeline(
                config_path=None,  # We'll pass config directly
                preloaded_data_df=self.data,
                test_size=test_size,
                random_state=42,
                **model_kwargs
            )
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def get_prediction(self, pt_data: List[Dict]) -> List[Dict]:
        """
        Get predictions for patient data.
        
        Args:
            pt_data: List of patient data dictionaries
            
        Returns:
            List of prediction dictionaries
            
        Note: This is a placeholder implementation for now
        """
        if self.model is None:
            raise RuntimeError("No trained model. Call train() first.")
        
        logger.info(f"Getting predictions for {len(pt_data)} patients...")
        
        # TODO: Implement prediction logic
        # For now, return placeholder
        predictions = []
        for i, patient in enumerate(pt_data):
            predictions.append({
                'patient_id': i,
                'hazard_ratio': 1.0,  # Placeholder
                '5_year_survival_prob': 0.5,  # Placeholder
                'note': 'Prediction logic not yet implemented'
            })
        
        return predictions
    
    def save_pickle(self, filepath: str):
        """
        Save the trained model to a pickle file.
        
        Args:
            filepath: Path where to save the model
            
        Note: This is a placeholder implementation for now
        """
        if self.model is None:
            raise RuntimeError("No trained model. Call train() first.")
        
        logger.info(f"Saving model to {filepath}...")
        
        # TODO: Implement privacy-compliant saving
        # For now, just save the model object
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'training_stats': self.training_stats,
                'feature_names': self.feature_names,
                'version': '0.2.0'
            }, f)
        
        logger.info(f"Model saved successfully to {filepath}")
    
    def get_train_stats(self) -> Dict:
        """
        Return training statistics.
        
        Returns:
            Dict containing training metrics
        """
        if self.training_stats is None:
            raise RuntimeError("No training stats available. Call train() first.")
        
        return self.training_stats.copy()
    
    def get_input_schema(self) -> Dict:
        """
        Return the required input schema for predictions.
        
        Returns:
            Dict describing expected input format
        """
        return {
            'required_features': self.feature_names,
            'outcome': self.outcome_name,
            'example_input': {feature: f"<{feature}_value>" for feature in self.feature_names},
            'note': 'Input schema generation not fully implemented yet'
        }
    
    def __repr__(self):
        """String representation of the wrapper."""
        status = []
        if self.data is not None:
            status.append(f"data_loaded({self.data.shape[0]} rows)")
        if self.model is not None:
            status.append("model_trained")
        
        status_str = ", ".join(status) if status else "empty"
        return f"CoxModelWrapper(outcome={self.outcome_name}, status={status_str})"
