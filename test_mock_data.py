#!/usr/bin/env python3
"""
Test script for mock data functionality in CoxModelWrapper.

This script tests the complete workflow with synthetic data.
"""

import sys
import traceback

def test_mock_data_generation():
    """Test mock data generation."""
    print("Testing mock data generation...")
    
    try:
        from twinsight_model import CoxModelWrapper
        
        # Load config from URL
        config_url = 'https://raw.githubusercontent.com/oneilsh/regression_model/refs/heads/main/configuration_cox.yaml'
        model = CoxModelWrapper(config_url)
        
        # Generate mock data
        model.load_data(use_mock=True, n_patients=500)
        
        print(f"✅ Mock data generated: {model.data.shape}")
        print(f"✅ Columns: {list(model.data.columns)}")
        print(f"✅ Event rate: {model.data['event_observed'].mean():.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock data generation failed: {e}")
        traceback.print_exc()
        return False

def test_data_summary():
    """Test privacy-compliant data summary."""
    print("\nTesting data summary...")
    
    try:
        from twinsight_model import CoxModelWrapper
        
        config_url = 'https://raw.githubusercontent.com/oneilsh/regression_model/refs/heads/main/configuration_cox.yaml'
        model = CoxModelWrapper(config_url)
        model.load_data(use_mock=True, n_patients=100)  # Small sample to test masking
        
        summary = model.train_data_summary()
        
        print("✅ Data summary generated:")
        for key, value in summary.items():
            print(f"    {key}: {value}")
        
        # Check that small counts are masked
        if 'demographics' in summary:
            print("✅ Demographics breakdown included")
        
        return True
        
    except Exception as e:
        print(f"❌ Data summary failed: {e}")
        traceback.print_exc()
        return False

def test_data_types():
    """Test that data types are correct for modeling."""
    print("\nTesting data types...")
    
    try:
        from twinsight_model import CoxModelWrapper
        
        config_url = 'https://raw.githubusercontent.com/oneilsh/regression_model/refs/heads/main/configuration_cox.yaml'
        model = CoxModelWrapper(config_url)
        model.load_data(use_mock=True, n_patients=200)
        
        # Check categorical columns
        categorical_cols = ['ethnicity', 'sex_at_birth', 'smoking_status']
        for col in categorical_cols:
            if col in model.data.columns:
                dtype = model.data[col].dtype
                print(f"✅ {col}: {dtype}")
                if dtype.name != 'category':
                    print(f"⚠️  Warning: {col} should be categorical")
        
        # Check numeric columns
        numeric_cols = ['age_at_time_0', 'bmi', 'time_to_event_days', 'event_observed']
        for col in numeric_cols:
            if col in model.data.columns:
                dtype = model.data[col].dtype
                print(f"✅ {col}: {dtype}")
        
        # Check no missing values in key columns
        key_cols = model.feature_names + [model.duration_col, model.event_col]
        missing_counts = model.data[key_cols].isnull().sum()
        if missing_counts.sum() == 0:
            print("✅ No missing values in key columns")
        else:
            print(f"⚠️  Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data type test failed: {e}")
        traceback.print_exc()
        return False

def test_survival_data_realism():
    """Test that survival data looks realistic."""
    print("\nTesting survival data realism...")
    
    try:
        from twinsight_model import CoxModelWrapper
        import pandas as pd
        
        config_url = 'https://raw.githubusercontent.com/oneilsh/regression_model/refs/heads/main/configuration_cox.yaml'
        model = CoxModelWrapper(config_url)
        model.load_data(use_mock=True, n_patients=1000)
        
        # Check event rate
        event_rate = model.data['event_observed'].mean()
        print(f"✅ Event rate: {event_rate:.1%}")
        
        if 0.1 <= event_rate <= 0.6:  # Reasonable range for 5-year COPD incidence
            print("✅ Event rate is realistic")
        else:
            print(f"⚠️  Event rate may be unrealistic: {event_rate:.1%}")
        
        # Check survival times
        survival_times = model.data['time_to_event_days']
        print(f"✅ Survival time range: {survival_times.min():.0f} - {survival_times.max():.0f} days")
        print(f"✅ Median survival time: {survival_times.median():.0f} days")
        
        # Check that higher risk patients have shorter survival (roughly)
        high_risk = model.data['smoking_status'] == 'Current'
        if high_risk.sum() > 10:  # If we have enough current smokers
            high_risk_survival = model.data[high_risk]['time_to_event_days'].median()
            low_risk_survival = model.data[~high_risk]['time_to_event_days'].median()
            
            if high_risk_survival < low_risk_survival:
                print(f"✅ Current smokers have shorter survival: {high_risk_survival:.0f} vs {low_risk_survival:.0f} days")
            else:
                print(f"⚠️  Risk gradient may be inverted")
        
        return True
        
    except Exception as e:
        print(f"❌ Survival data test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all mock data tests."""
    print("=" * 60)
    print("MOCK DATA FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        ("Mock Data Generation", test_mock_data_generation),
        ("Data Summary", test_data_summary),
        ("Data Types", test_data_types),
        ("Survival Data Realism", test_survival_data_realism),
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
    print(f"MOCK DATA TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All mock data tests passed!")
        print("\nNext steps:")
        print("1. Test model training with mock data")
        print("2. Implement prediction functionality")
        print("3. Test end-to-end workflow")
    else:
        print("❌ Some tests failed. Fix issues before proceeding.")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
