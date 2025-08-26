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

## 🚀 Cell 4: All-in-One (Alternative)
```python
# Complete workflow - run this instead of cells 2+3
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
    config = {
        'outcome': {'name': 'test'},
        'model_features_final': ['age', 'bmi'],
        'model_io_columns': {'duration_col': 'time_to_event_days', 'event_col': 'event_observed'}
    }
    model = CoxModelWrapper(config)
    print(f"🎉 Success: {model}")
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
