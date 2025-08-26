"""
twinsight_model v0.2.0 - Simplified Cox regression modeling for All of Us data

This version focuses on:
- Minimal dependencies (no MLflow)
- Clean API design
- Privacy-compliant data handling
"""

__version__ = "0.2.0"

# New simplified API (recommended)
from .wrapper import CoxModelWrapper

# Legacy imports (for backward compatibility)
# from .dataloader_cox import load_configuration, load_data_from_bigquery
# from .preprocessing_cox import create_preprocessor, apply_preprocessing  
# from .model_cox import run_end_to_end_pipeline
