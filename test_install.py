#!/usr/bin/env python3
"""
Minimal installation test script for twinsight_model v0.2.0

Run this after installation to verify the package works in the AoU environment.
"""

import sys
import importlib
import warnings

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    # Test standard packages that should be in AoU
    standard_packages = [
        'pandas', 'numpy', 'sklearn', 'yaml', 'joblib'
    ]
    
    for pkg in standard_packages:
        try:
            importlib.import_module(pkg)
            print(f"✓ {pkg} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {pkg}: {e}")
            return False
    
    # Test our additional dependencies
    additional_packages = [
        'lifelines', 'google.cloud.bigquery'
    ]
    
    for pkg in additional_packages:
        try:
            importlib.import_module(pkg)
            print(f"✓ {pkg} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {pkg}: {e}")
            return False
    
    # Test our package
    try:
        import twinsight_model
        print(f"✓ twinsight_model v{getattr(twinsight_model, '__version__', 'unknown')} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import twinsight_model: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without requiring data."""
    print("\nTesting basic functionality...")
    
    try:
        # Test configuration loading
        from twinsight_model.dataloader_cox import load_configuration
        print("✓ Configuration loader available")
        
        # Test preprocessing components
        from twinsight_model.preprocessing_cox import OutlierCapper
        print("✓ Preprocessing components available")
        
        # Test model components
        from lifelines import CoxPHFitter
        print("✓ Cox regression model available")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def check_version_conflicts():
    """Check for potential version conflicts."""
    print("\nChecking for version conflicts...")
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        
        print(f"pandas version: {pd.__version__}")
        print(f"numpy version: {np.__version__}")
        print(f"scikit-learn version: {sklearn.__version__}")
        
        # Check for numpy 2.x warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            np.array([1, 2, 3])
            if w:
                for warning in w:
                    if "numpy" in str(warning.message).lower():
                        print(f"⚠ NumPy warning: {warning.message}")
        
        return True
        
    except Exception as e:
        print(f"✗ Version check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("twinsight_model Installation Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Functionality Test", test_basic_functionality),
        ("Version Conflict Check", check_version_conflicts)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Installation appears successful.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
