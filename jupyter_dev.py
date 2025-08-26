"""
Jupyter-friendly development tools for twinsight_model.

Run this in Jupyter to safely update and test your package during development.
All functions are designed to work within Jupyter cells.
"""

import subprocess
import sys
import os
import importlib
import tempfile
import shutil
from pathlib import Path

def safe_update_package():
    """
    Safely update twinsight_model package without affecting dependencies.
    
    Usage in Jupyter:
        exec(open('jupyter_dev.py').read())
        safe_update_package()
    """
    print("🔄 Safely updating twinsight_model...")
    
    try:
        # Use Jupyter ! commands for maximum compatibility
        import subprocess
        import sys
        
        # Uninstall old version quietly
        result1 = subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'twinsight_model', '-y', '-q'], 
                                capture_output=True, text=True)
        
        # Install new version without dependencies
        result2 = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'git+https://github.com/oneilsh/regression_model.git',
            '--no-deps', '--force-reinstall'
        ], capture_output=True, text=True)
        
        if result2.returncode == 0:
            print("✅ Package updated successfully!")
            reload_modules()
            return True
        else:
            print(f"❌ Update failed: {result2.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Update error: {e}")
        return False

def reload_modules():
    """
    Reload twinsight_model modules to pick up changes.
    
    Usage in Jupyter:
        reload_modules()
    """
    print("🔄 Reloading modules...")
    
    modules_to_reload = [
        'twinsight_model',
        'twinsight_model.wrapper',
        'twinsight_model.dataloader_cox',
        'twinsight_model.preprocessing_cox',
        'twinsight_model.model_cox'
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
                print(f"  ✅ Reloaded {module_name}")
            except Exception as e:
                print(f"  ⚠️  Failed to reload {module_name}: {e}")
    
    print("✅ Module reload complete")

def quick_test():
    """
    Quick test of the wrapper functionality.
    
    Usage in Jupyter:
        quick_test()
    """
    print("🧪 Running quick test...")
    
    try:
        # Test import
        from twinsight_model import CoxModelWrapper
        print("  ✅ Import successful")
        
        # Test wrapper creation
        config_dict = {
            'outcome': {'name': 'test_outcome'},
            'model_features_final': ['age', 'bmi', 'smoking'],
            'model_io_columns': {
                'duration_col': 'time_to_event_days',
                'event_col': 'event_observed'
            }
        }
        
        wrapper = CoxModelWrapper(config_dict)
        print(f"  ✅ Wrapper created: {wrapper}")
        
        # Test schema
        schema = wrapper.get_input_schema()
        print(f"  ✅ Schema: {len(schema['required_features'])} features")
        
        print("🎉 Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def dev_workflow():
    """
    Complete development workflow: update, reload, test.
    
    Usage in Jupyter:
        exec(open('jupyter_dev.py').read())
        dev_workflow()
    """
    print("🚀 Starting development workflow...")
    
    if safe_update_package():
        if quick_test():
            print("\n✅ Development workflow complete!")
            print("\nYou can now use:")
            print("  from twinsight_model import CoxModelWrapper")
            print("  model = CoxModelWrapper('configuration_cox.yaml')")
            return True
        else:
            print("\n❌ Tests failed after update")
            return False
    else:
        print("\n❌ Package update failed")
        return False

def clone_and_edit():
    """
    Clone the repo locally for editing (if you want to modify files directly).
    
    Usage in Jupyter:
        clone_and_edit()
    """
    print("📥 Cloning repository for local editing...")
    
    repo_path = "regression_model_local"
    
    try:
        # Remove existing clone if present
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        
        # Clone fresh copy
        result = subprocess.run([
            'git', 'clone', 'https://github.com/oneilsh/regression_model.git', repo_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Repository cloned to: {repo_path}")
            print(f"\nTo edit files, navigate to: {os.path.abspath(repo_path)}")
            print("\nTo install your local changes:")
            print(f"  cd {repo_path}")
            print("  pip install -e . --no-deps")
            return True
        else:
            print(f"❌ Clone failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Clone error: {e}")
        return False

def install_local_editable(repo_path="regression_model_local"):
    """
    Install local repository in editable mode.
    
    Usage in Jupyter (after clone_and_edit):
        install_local_editable()
    """
    print(f"📦 Installing local repository in editable mode...")
    
    if not os.path.exists(repo_path):
        print(f"❌ Repository not found at {repo_path}")
        print("Run clone_and_edit() first")
        return False
    
    try:
        # Install in editable mode without dependencies
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-e', repo_path, '--no-deps'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Local editable installation successful!")
            reload_modules()
            return quick_test()
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

# Convenience function for Jupyter
def dev():
    """Shorthand for development workflow."""
    return dev_workflow()

if __name__ == "__main__":
    print("🛠️  JUPYTER DEVELOPMENT TOOLS LOADED")
    print("\nAvailable functions:")
    print("  dev_workflow()     - Complete update, reload, test cycle")
    print("  dev()              - Shorthand for dev_workflow()")
    print("  safe_update_package() - Update package without dependency conflicts")
    print("  reload_modules()   - Reload modules after changes")
    print("  quick_test()       - Test wrapper functionality")
    print("  clone_and_edit()   - Clone repo for local editing")
    print("  install_local_editable() - Install local changes")
    print("\nQuick start:")
    print("  dev()")
else:
    # When imported, just run the dev workflow
    dev_workflow()
