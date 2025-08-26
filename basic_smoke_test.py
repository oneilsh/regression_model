#!/usr/bin/env python3
"""
Basic smoke test for twinsight_model v0.2.0

This test verifies that the current functionality still works after the dependency fixes.
Run this to ensure we haven't broken anything before proceeding with the wrapper development.
"""

import sys
import traceback

def test_configuration_loading():
    """Test that configuration loading works."""
    print("Testing configuration loading...")
    try:
        from twinsight_model.dataloader_cox import load_configuration
        
        # Test with the existing config file
        config = load_configuration('configuration_cox.yaml')
        
        # Basic validation
        assert 'outcome' in config, "Config missing 'outcome' section"
        assert 'features' in config, "Config missing 'features' section"
        assert 'model_features_final' in config, "Config missing 'model_features_final'"
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - Outcome: {config['outcome']['name']}")
        print(f"  - Features: {len(config['model_features_final'])} final features")
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        traceback.print_exc()
        return False

def test_preprocessing_components():
    """Test that preprocessing components can be imported and created."""
    print("\nTesting preprocessing components...")
    try:
        from twinsight_model.preprocessing_cox import OutlierCapper, create_preprocessor
        import pandas as pd
        import numpy as np
        
        # Create some test data
        test_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B']
        })
        
        # Test OutlierCapper
        capper = OutlierCapper(lower_bound_quantile=0.1, upper_bound_quantile=0.9)
        capper.fit(test_data[['numeric_col']])
        
        print("✓ OutlierCapper works")
        
        # Test preprocessor creation (basic test)
        test_data['categorical_col'] = test_data['categorical_col'].astype('category')
        preprocessor = create_preprocessor(test_data)
        
        print("✓ Preprocessor creation works")
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        traceback.print_exc()
        return False

def test_model_components():
    """Test that model components can be imported."""
    print("\nTesting model components...")
    try:
        from lifelines import CoxPHFitter
        from twinsight_model.model_cox import evaluate_cox_model
        
        # Test CoxPHFitter instantiation
        model = CoxPHFitter()
        print("✓ CoxPHFitter can be instantiated")
        
        # Test that our evaluation function exists
        print("✓ evaluate_cox_model function available")
        return True
        
    except Exception as e:
        print(f"✗ Model components test failed: {e}")
        traceback.print_exc()
        return False

def test_version_info():
    """Test version information."""
    print("\nTesting version information...")
    try:
        import twinsight_model
        import pandas as pd
        import lifelines
        
        print(f"✓ twinsight_model version: {getattr(twinsight_model, '__version__', 'unknown')}")
        print(f"✓ pandas version: {pd.__version__}")
        print(f"✓ lifelines version: {lifelines.__version__}")
        return True
        
    except Exception as e:
        print(f"✗ Version info test failed: {e}")
        return False

def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("TWINSIGHT_MODEL v0.2.0 SMOKE TEST")
    print("=" * 60)
    
    tests = [
        ("Version Info", test_version_info),
        ("Configuration Loading", test_configuration_loading),
        ("Preprocessing Components", test_preprocessing_components),
        ("Model Components", test_model_components),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 60)
    print(f"SMOKE TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All smoke tests passed! Ready for next development phase.")
        print("\nNext steps:")
        print("1. Create basic CoxModelWrapper class")
        print("2. Implement simplified API methods")
        print("3. Add privacy-compliant data handling")
    else:
        print("❌ Some tests failed. Fix issues before proceeding.")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
