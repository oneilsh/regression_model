#!/usr/bin/env python3
"""
Development installation script for AoU environment.

This script allows you to update just the twinsight_model package without
reinstalling dependencies, avoiding pandas conflicts.
"""

import subprocess
import sys
import os
import shutil
import importlib

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✓ {description} successful")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_package_installed():
    """Check if twinsight_model is already installed."""
    try:
        import twinsight_model
        print(f"✓ twinsight_model currently installed: v{getattr(twinsight_model, '__version__', 'unknown')}")
        return True
    except ImportError:
        print("ℹ twinsight_model not currently installed")
        return False

def install_fresh():
    """Fresh installation with dependency management."""
    print("=" * 60)
    print("FRESH INSTALLATION")
    print("=" * 60)
    
    # First install compatible lifelines if not present
    try:
        import lifelines
        print(f"✓ lifelines already available: v{lifelines.__version__}")
    except ImportError:
        if not run_command("pip install lifelines==0.27.5", "Installing lifelines 0.27.5"):
            return False
    
    # Install our package without dependencies (they should already be satisfied)
    success = run_command(
        "pip install git+https://github.com/oneilsh/regression_model.git --no-deps",
        "Installing twinsight_model (no dependencies)"
    )
    
    return success

def update_existing():
    """Update existing installation without touching dependencies."""
    print("=" * 60) 
    print("UPDATE EXISTING INSTALLATION")
    print("=" * 60)
    
    # Uninstall old version
    run_command("pip uninstall twinsight_model -y", "Removing old twinsight_model")
    
    # Install new version without dependencies
    success = run_command(
        "pip install git+https://github.com/oneilsh/regression_model.git --no-deps",
        "Installing updated twinsight_model (no dependencies)"
    )
    
    return success

def test_installation():
    """Test the installation."""
    print("\n" + "=" * 60)
    print("TESTING INSTALLATION")
    print("=" * 60)
    
    # Force reload of modules
    if 'twinsight_model' in sys.modules:
        importlib.reload(sys.modules['twinsight_model'])
    
    try:
        # Test basic import
        from twinsight_model import CoxModelWrapper
        print("✓ CoxModelWrapper import successful")
        
        # Test wrapper creation
        config_dict = {
            'outcome': {'name': 'test_outcome'},
            'model_features_final': ['age', 'bmi'],
            'model_io_columns': {
                'duration_col': 'time_to_event_days',
                'event_col': 'event_observed'
            }
        }
        
        wrapper = CoxModelWrapper(config_dict)
        print(f"✓ Wrapper creation successful: {wrapper}")
        
        # Test schema
        schema = wrapper.get_input_schema()
        print(f"✓ Schema generation successful: {len(schema['required_features'])} features")
        
        return True
        
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main development installation workflow."""
    print("🚀 TWINSIGHT_MODEL DEVELOPMENT INSTALLER")
    print("This script safely updates your package without breaking dependencies")
    
    # Check current state
    already_installed = check_package_installed()
    
    # Choose installation method
    if already_installed:
        if update_existing():
            print("✓ Update completed")
        else:
            print("✗ Update failed")
            return False
    else:
        if install_fresh():
            print("✓ Fresh installation completed")
        else:
            print("✗ Fresh installation failed")
            return False
    
    # Test the result
    if test_installation():
        print("\n🎉 Development installation successful!")
        print("\nNext steps:")
        print("1. Test your wrapper: from twinsight_model import CoxModelWrapper")
        print("2. Run: python test_wrapper_basic.py")
        print("3. Develop and iterate!")
        return True
    else:
        print("\n❌ Installation test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
