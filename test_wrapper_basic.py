#!/usr/bin/env python3
"""
Basic test script for CoxModelWrapper v0.2.0

This script tests the wrapper's basic functionality without requiring actual data.
"""

import sys
import traceback

def test_wrapper_import():
    """Test that the wrapper can be imported."""
    print("Testing wrapper import...")
    try:
        from twinsight_model.wrapper import CoxModelWrapper
        print("✓ CoxModelWrapper imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import CoxModelWrapper: {e}")
        traceback.print_exc()
        return False

def test_wrapper_config_loading():
    """Test wrapper initialization with config file."""
    print("\nTesting wrapper config loading...")
    try:
        from twinsight_model.wrapper import CoxModelWrapper
        
        # Test with existing config file
        wrapper = CoxModelWrapper('configuration_cox.yaml')
        
        print(f"✓ Wrapper initialized: {wrapper}")
        print(f"  - Outcome: {wrapper.outcome_name}")
        print(f"  - Features: {len(wrapper.feature_names)}")
        print(f"  - Config source: {wrapper.config_source}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        traceback.print_exc()
        return False

def test_wrapper_dict_config():
    """Test wrapper initialization with dictionary config."""
    print("\nTesting wrapper with dict config...")
    try:
        from twinsight_model.wrapper import CoxModelWrapper
        
        # Test with minimal config dict
        config_dict = {
            'outcome': {'name': 'test_outcome'},
            'model_features_final': ['age', 'bmi', 'smoking'],
            'model_io_columns': {
                'duration_col': 'time_to_event_days',
                'event_col': 'event_observed'
            }
        }
        
        wrapper = CoxModelWrapper(config_dict)
        
        print(f"✓ Wrapper initialized with dict: {wrapper}")
        print(f"  - Outcome: {wrapper.outcome_name}")
        print(f"  - Features: {wrapper.feature_names}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dict config failed: {e}")
        traceback.print_exc()
        return False

def test_wrapper_methods():
    """Test wrapper methods (without data)."""
    print("\nTesting wrapper methods...")
    try:
        from twinsight_model.wrapper import CoxModelWrapper
        
        config_dict = {
            'outcome': {'name': 'test_outcome'},
            'model_features_final': ['age', 'bmi'],
            'model_io_columns': {
                'duration_col': 'time_to_event_days',
                'event_col': 'event_observed'
            }
        }
        
        wrapper = CoxModelWrapper(config_dict)
        
        # Test get_input_schema (should work without data)
        schema = wrapper.get_input_schema()
        print(f"✓ get_input_schema(): {len(schema['required_features'])} features")
        
        # Test methods that should fail without data
        try:
            wrapper.train_data_summary()
            print("✗ train_data_summary() should fail without data")
            return False
        except RuntimeError as e:
            print("✓ train_data_summary() correctly fails without data")
        
        try:
            wrapper.train()
            print("✗ train() should fail without data")
            return False
        except RuntimeError as e:
            print("✓ train() correctly fails without data")
        
        try:
            wrapper.get_prediction([{'age': 65}])
            print("✗ get_prediction() should fail without model")
            return False
        except RuntimeError as e:
            print("✓ get_prediction() correctly fails without model")
        
        return True
        
    except Exception as e:
        print(f"✗ Method testing failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all wrapper tests."""
    print("=" * 60)
    print("COXMODELWRAPPER BASIC TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_wrapper_import),
        ("Config File Loading", test_wrapper_config_loading),
        ("Dict Config Loading", test_wrapper_dict_config),
        ("Method Tests", test_wrapper_methods),
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
    print(f"WRAPPER TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All wrapper tests passed!")
        print("\nNext steps:")
        print("1. Test load_data() with mock data")
        print("2. Implement full training pipeline") 
        print("3. Add prediction logic")
        print("4. Add privacy-compliant data handling")
    else:
        print("❌ Some tests failed. Fix issues before proceeding.")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
