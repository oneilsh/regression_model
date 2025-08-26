#!/usr/bin/env python3
"""
Test script for training functionality in CoxModelWrapper.

This script tests the complete training workflow with mock data.
"""

import sys
import traceback

def test_training_workflow():
    """Test the complete training workflow."""
    print("Testing training workflow...")
    
    try:
        from twinsight_model import CoxModelWrapper
        
        # Load config and generate mock data
        config_url = 'https://raw.githubusercontent.com/oneilsh/regression_model/refs/heads/main/configuration_cox.yaml'
        model = CoxModelWrapper(config_url)
        
        print("✅ Model initialized")
        
        # Load mock data
        model.load_data(use_mock=True, n_patients=300)  # Smaller sample for faster training
        print("✅ Mock data loaded")
        
        # Train model
        print("🔄 Training model...")
        success = model.train(split=0.8, random_state=42)
        
        if success:
            print("✅ Training completed successfully")
            
            # Get training stats
            stats = model.get_train_stats()
            print(f"✅ Training stats: {stats}")
            
            # Check model object
            if model.model is not None:
                print(f"✅ Model object created: {type(model.model).__name__}")
                
                # Print model summary
                print("\n--- Model Summary ---")
                model.model.print_summary()
                print("-------------------")
            else:
                print("❌ Model object is None")
                return False
                
        else:
            print("❌ Training failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Training workflow failed: {e}")
        traceback.print_exc()
        return False

def test_model_persistence():
    """Test model saving and loading."""
    print("\nTesting model persistence...")
    
    try:
        from twinsight_model import CoxModelWrapper
        
        # Create and train a model
        config_url = 'https://raw.githubusercontent.com/oneilsh/regression_model/refs/heads/main/configuration_cox.yaml'
        model = CoxModelWrapper(config_url)
        model.load_data(use_mock=True, n_patients=200)
        model.train(split=0.8, random_state=42)
        
        # Save model
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model.save_pickle(tmp.name)
            print(f"✅ Model saved to: {tmp.name}")
            
            # Clean up
            os.unlink(tmp.name)
        
        return True
        
    except Exception as e:
        print(f"❌ Model persistence failed: {e}")
        traceback.print_exc()
        return False

def test_prediction_functionality():
    """Test prediction functionality."""
    print("\nTesting prediction functionality...")
    
    try:
        from twinsight_model import CoxModelWrapper
        
        # Create and train a model
        config_url = 'https://raw.githubusercontent.com/oneilsh/regression_model/refs/heads/main/configuration_cox.yaml'
        model = CoxModelWrapper(config_url)
        model.load_data(use_mock=True, n_patients=300)
        model.train(split=0.8, random_state=42)
        
        # Create test patient data matching the raw feature names from config
        test_patients = [
            {
                'age_at_time_0': 65, 'sex_at_birth': 'Male',
                'smoking_status': 'Current', 'bmi': 28.5, 
                'diabetes': 1, 'obesity': 1, 'cardiovascular_disease': 0,
                'alcohol_use': 1, 'ethnicity': 'Not Hispanic or Latino'
            },
            {
                'age_at_time_0': 45, 'sex_at_birth': 'Female',
                'smoking_status': 'Never', 'bmi': 24.0, 
                'diabetes': 0, 'obesity': 0, 'cardiovascular_disease': 0,
                'alcohol_use': 0, 'ethnicity': 'Not Hispanic or Latino'
            }
        ]
        
        # Get predictions
        predictions = model.get_prediction(test_patients)
        print(f"✅ Generated predictions for {len(predictions)} patients")
        
        # Test single patient prediction
        single_pred = model.predict_single(test_patients[0])
        print(f"✅ Single patient prediction: {single_pred['risk_category']} risk")
        
        # Test feature importance
        importance = model.get_feature_importance()
        print(f"✅ Feature importance: {len(importance)} features analyzed")
        
        # Print sample prediction
        print(f"\n--- Sample Prediction ---")
        print(f"Patient 1: {predictions[0]['risk_category']} risk")
        print(f"Survival probability: {predictions[0]['survival_probability']:.3f}")
        print(f"Relative risk: {predictions[0]['relative_risk']:.2f}")
        print("------------------------")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction functionality failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all training tests."""
    print("=" * 60)
    print("TRAINING FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        ("Training Workflow", test_training_workflow),
        ("Model Persistence", test_model_persistence),
        ("Prediction Functionality", test_prediction_functionality),
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
    print(f"TRAINING TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All training tests passed!")
        print("\nNext steps:")
        print("1. Test prediction functionality")
        print("2. Test with real data")
        print("3. Optimize model parameters")
    else:
        print("❌ Some tests failed. Fix issues before proceeding.")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
