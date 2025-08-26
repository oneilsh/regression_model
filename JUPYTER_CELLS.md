# Jupyter Development Cells for twinsight_model

Copy-paste these cells directly into Jupyter. No shell access needed!

## 🔧 Cell 1: Fresh Environment Setup
```python
# Run once in fresh AoU environment
print("🔧 Setting up twinsight_model...")

# Install lifelines with its dependencies (should be safe with 0.27.5)
! pip install lifelines==0.27.5

# Install our package without dependencies
! pip install --no-deps git+https://github.com/oneilsh/regression_model.git

print("✅ Setup complete!")
```

## 🔄 Cell 2: Development Iteration
```python
# Run after each GitHub push to update your package
print("🔄 Updating twinsight_model...")

! pip uninstall twinsight_model -y -q
! pip install --no-deps --force-reinstall git+https://github.com/oneilsh/regression_model.git

# Reload modules to pick up changes
import importlib
import sys

for module in ['twinsight_model', 'twinsight_model.wrapper']:
    if module in sys.modules:
        importlib.reload(sys.modules[module])

print("✅ Package updated!")
print("⚠️  IMPORTANT: If you get method signature errors, restart your kernel!")
```

## 🧪 Cell 3: Quick Test
```python
# Test that everything works
print("🧪 Testing wrapper...")

try:
    from twinsight_model import CoxModelWrapper
    
    # Test with URL config (real COPD configuration)
    config_url = 'https://raw.githubusercontent.com/oneilsh/regression_model/refs/heads/main/configuration_cox.yaml'
    model = CoxModelWrapper(config_url)
    print(f"✅ Wrapper with URL config: {model}")
    
    schema = model.get_input_schema()
    print(f"✅ Schema: {len(schema['required_features'])} features")
    print(f"✅ Outcome: {model.outcome_name}")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
```

## 🚀 Cell 4: Mock Data Test
```python
# Test complete workflow with mock data
print("🚀 Testing mock data workflow...")

try:
    from twinsight_model import CoxModelWrapper
    
    # Load real config and generate mock data
    config_url = 'https://raw.githubusercontent.com/oneilsh/regression_model/refs/heads/main/configuration_cox.yaml'
    model = CoxModelWrapper(config_url)
    
    # Generate mock data
    model.load_data(use_mock=True, n_patients=500)
    
    # Get data summary
    summary = model.train_data_summary()
    
    print(f"✅ Mock data loaded: {model.data.shape}")
    print(f"✅ Event rate: {model.data['event_observed'].mean():.1%}")
    print(f"✅ Summary: {summary['total_patients']} patients, {summary['events_observed']} events")
    
    print("🎉 Mock data workflow successful!")
    
    # Train model
    model.train(split=0.8, random_state=42)
    print(f"✅ Model trained: {model.model}")
    
    # Test predictions with correct feature names
    test_patient = {
        'age_at_time_0': 65, 'sex_at_birth': 'Male',
        'smoking_status': 'Current', 'bmi': 28.5, 
        'diabetes': 1, 'obesity': 1, 'cardiovascular_disease': 0,
        'alcohol_use': 1, 'ethnicity': 'Not Hispanic or Latino'
    }
    
    prediction = model.predict_single(test_patient)
    print(f"✅ Risk: {prediction['risk_category']}")
    print(f"✅ Survival: {prediction['survival_probability']:.3f}")
    print(f"✅ Relative Risk: {prediction['relative_risk']:.2f}")
    
    print("🎉 Complete workflow successful!")
    
except Exception as e:
    print(f"❌ Mock data test failed: {e}")
    import traceback
    traceback.print_exc()
```

## 🧪 Cell 5: All-in-One (Alternative)
```python
# Complete workflow - run this instead of cells 2+3+4
print("🚀 Full development workflow...")

# Update package
! pip uninstall twinsight_model -y -q
! pip install --no-deps --force-reinstall git+https://github.com/oneilsh/regression_model.git

# Reload and test
import importlib
import sys
for module in ['twinsight_model', 'twinsight_model.wrapper']:
    if module in sys.modules:
        importlib.reload(sys.modules[module])

try:
    from twinsight_model import CoxModelWrapper
    config_url = 'https://raw.githubusercontent.com/oneilsh/regression_model/refs/heads/main/configuration_cox.yaml'
    model = CoxModelWrapper(config_url)
    model.load_data(use_mock=True, n_patients=200)
    print(f"🎉 Success: {model}")
    print(f"📊 Data: {model.data.shape}, Events: {model.data['event_observed'].sum()}")
except Exception as e:
    print(f"❌ Failed: {e}")
```

## 📋 Usage

**Fresh environment:** Run Cell 1 once
**Development:** After pushing to GitHub, run Cell 2 + Cell 3 (or just Cell 4)
**No shell needed** - everything uses `!` commands in Jupyter

## 🎯 Key Points

- `--no-deps` prevents pandas conflicts
- `--force-reinstall` ensures you get the latest version
- Module reloading picks up code changes immediately
- Completely Jupyter-native workflow
