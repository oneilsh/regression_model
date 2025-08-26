"""
Simple Jupyter-only development workflow for twinsight_model.

Just copy-paste these cells into Jupyter - no shell needed!
"""

# Cell 1: One-time setup (run once per fresh environment)
SETUP_CELL = '''
# One-time setup for fresh AoU environment
print("🔧 Setting up twinsight_model dependencies...")

# Install only the dependencies we need that aren't in AoU
! pip install lifelines==0.27.5 --no-deps
! pip install protobuf==3.20.3 --no-deps  

# Install our package without dependencies
! pip install --no-deps git+https://github.com/oneilsh/regression_model.git

print("✅ Setup complete!")
'''

# Cell 2: Development iteration (run after each GitHub push)
DEV_CELL = '''
# Development iteration - update package only
print("🔄 Updating twinsight_model...")

# Uninstall old version and install new one without touching dependencies
! pip uninstall twinsight_model -y -q
! pip install --no-deps git+https://github.com/oneilsh/regression_model.git

# Force reload of modules
import importlib
import sys

modules_to_reload = [
    'twinsight_model',
    'twinsight_model.wrapper', 
    'twinsight_model.dataloader_cox',
    'twinsight_model.preprocessing_cox',
    'twinsight_model.model_cox'
]

for module in modules_to_reload:
    if module in sys.modules:
        importlib.reload(sys.modules[module])

print("✅ Package updated and modules reloaded!")
'''

# Cell 3: Quick test (run after updates)
TEST_CELL = '''
# Quick test of wrapper functionality
print("🧪 Testing wrapper...")

try:
    from twinsight_model import CoxModelWrapper
    
    # Test with minimal config
    config = {
        'outcome': {'name': 'test_outcome'},
        'model_features_final': ['age', 'bmi', 'smoking'],
        'model_io_columns': {
            'duration_col': 'time_to_event_days',
            'event_col': 'event_observed'
        }
    }
    
    model = CoxModelWrapper(config)
    print(f"✅ Wrapper created: {model}")
    
    # Test schema
    schema = model.get_input_schema()
    print(f"✅ Schema: {len(schema['required_features'])} features")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
'''

# Cell 4: Full workflow (combines everything)
FULL_WORKFLOW_CELL = '''
# Complete development workflow - run this for everything at once
print("🚀 Full development workflow...")

# Update package
! pip uninstall twinsight_model -y -q
! pip install --no-deps --force-reinstall git+https://github.com/oneilsh/regression_model.git

# Reload modules
import importlib
import sys
for module in ['twinsight_model', 'twinsight_model.wrapper']:
    if module in sys.modules:
        importlib.reload(sys.modules[module])

# Test
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
'''

def print_cells():
    """Print all the cells for easy copy-paste."""
    print("📋 JUPYTER DEVELOPMENT CELLS")
    print("="*60)
    
    print("\n🔧 CELL 1: One-time setup (fresh environment)")
    print("="*60)
    print(SETUP_CELL.strip())
    
    print("\n\n🔄 CELL 2: Development iteration (after GitHub push)")  
    print("="*60)
    print(DEV_CELL.strip())
    
    print("\n\n🧪 CELL 3: Quick test")
    print("="*60)
    print(TEST_CELL.strip())
    
    print("\n\n🚀 CELL 4: Full workflow (all-in-one)")
    print("="*60)
    print(FULL_WORKFLOW_CELL.strip())
    
    print("\n\n📋 USAGE INSTRUCTIONS")
    print("="*60)
    print("1. Fresh environment: Copy-paste CELL 1")
    print("2. After GitHub changes: Copy-paste CELL 2 + CELL 3")
    print("3. Or just use CELL 4 for everything")
    print("4. No shell needed - everything uses ! commands in Jupyter")

if __name__ == "__main__":
    print_cells()
