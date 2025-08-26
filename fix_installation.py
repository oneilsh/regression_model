#!/usr/bin/env python3
"""
Emergency fix script for AoU installation issues.

This script:
1. Downgrades pandas back to AoU-compatible version
2. Installs lifelines 0.29.0 (compatible with pandas 2.0.3)
3. Reinstalls twinsight_model
4. Tests the installation

Run this in Jupyter with: exec(open('fix_installation.py').read())
"""

import subprocess
import sys
import importlib

def run_pip_command(command):
    """Run a pip command and return success status."""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip'] + command.split(), 
                              capture_output=True, text=True, check=True)
        print(f"✓ Success: {' '.join(command.split())}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {' '.join(command.split())}")
        print(f"Error: {e.stderr}")
        return False

def test_import(module_name):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} imports successfully")
        return True
    except ImportError as e:
        print(f"✗ {module_name} import failed: {e}")
        return False

def main():
    print("=" * 60)
    print("FIXING AoU INSTALLATION ISSUES")
    print("=" * 60)
    
    print("\n1. Downgrading pandas to AoU-compatible version...")
    if not run_pip_command("install pandas==2.0.3 --force-reinstall"):
        print("Failed to downgrade pandas. Continuing anyway...")
    
    print("\n2. Installing compatible lifelines version...")
    if not run_pip_command("install lifelines==0.29.0 --force-reinstall"):
        print("Failed to install lifelines 0.29.0")
        return False
    
    print("\n3. Reinstalling twinsight_model...")
    if not run_pip_command("uninstall twinsight_model -y"):
        print("Note: twinsight_model was not installed or uninstall failed")
    
    if not run_pip_command("install git+https://github.com/oneilsh/regression_model"):
        print("Failed to reinstall twinsight_model")
        return False
    
    print("\n4. Testing imports...")
    success = True
    
    # Test basic packages
    for module in ['pandas', 'numpy', 'sklearn', 'lifelines']:
        if not test_import(module):
            success = False
    
    # Test our package
    if not test_import('twinsight_model'):
        success = False
        print("\n⚠️  twinsight_model import failed. You may need to:")
        print("   1. Restart your kernel (Kernel → Restart)")
        print("   2. Try the import again")
        print("   3. If still failing, check the environment")
    else:
        # Test specific import
        try:
            from twinsight_model.dataloader_cox import load_configuration
            print("✓ twinsight_model.dataloader_cox imports successfully")
        except ImportError as e:
            print(f"✗ Detailed import failed: {e}")
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Installation fix completed successfully!")
        print("You should now be able to import twinsight_model")
    else:
        print("❌ Some issues remain. Try restarting your kernel.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()
