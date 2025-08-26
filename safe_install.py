#!/usr/bin/env python3
"""
Ultra-safe installation for AoU environment.

This script installs twinsight_model while being extremely careful 
not to disturb existing pandas/dependency versions.
"""

import subprocess
import sys
import pandas as pd

def check_pandas_version():
    """Check current pandas version."""
    print(f"Current pandas version: {pd.__version__}")
    return pd.__version__

def install_with_pip_constraints():
    """Install using pip constraints to prevent version changes."""
    
    # Get current versions of critical packages
    import pandas
    import numpy 
    import sklearn
    
    current_versions = {
        'pandas': pandas.__version__,
        'numpy': numpy.__version__,
        'scikit-learn': sklearn.__version__
    }
    
    print("Current package versions:")
    for pkg, version in current_versions.items():
        print(f"  {pkg}: {version}")
    
    print("\nInstalling twinsight_model with version constraints...")
    
    # Create constraints
    constraints = [
        f"pandas=={current_versions['pandas']}",
        f"numpy=={current_versions['numpy']}",
        f"scikit-learn=={current_versions['scikit-learn']}"
    ]
    
    # First ensure lifelines is compatible
    try:
        import lifelines
        print(f"lifelines already available: v{lifelines.__version__}")
    except ImportError:
        print("Installing lifelines 0.27.5...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'lifelines==0.27.5'] + 
                      [f'--constraint=<(echo "{c}")' for c in constraints], check=True)
    
    # Install our package with constraints
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        'git+https://github.com/oneilsh/regression_model.git',
        '--upgrade'
    ]
    
    # Add version pins to prevent upgrades
    for constraint in constraints:
        cmd.extend(['--constraint', constraint])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Installation successful!")
        return True
    else:
        print(f"❌ Installation failed: {result.stderr}")
        return False

def simple_no_deps_install():
    """Simplest approach: install without any dependency checking."""
    print("Installing twinsight_model without dependencies...")
    
    try:
        # Uninstall old version if present
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'twinsight_model', '-y'], 
                      capture_output=True)
        
        # Install new version without deps
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/oneilsh/regression_model.git',
            '--no-deps'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Installation successful!")
            
            # Test import
            try:
                import twinsight_model
                print(f"✅ Import successful: v{getattr(twinsight_model, '__version__', 'unknown')}")
                return True
            except ImportError as e:
                print(f"❌ Import failed: {e}")
                return False
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main safe installation workflow."""
    print("🛡️  ULTRA-SAFE TWINSIGHT_MODEL INSTALLER")
    print("=" * 50)
    
    pandas_before = check_pandas_version()
    
    # Try simple approach first
    if simple_no_deps_install():
        pandas_after = check_pandas_version()
        
        if pandas_before == pandas_after:
            print(f"🎉 Success! Pandas version unchanged: {pandas_after}")
            return True
        else:
            print(f"⚠️  Pandas version changed from {pandas_before} to {pandas_after}")
            print("You may need to run fix_installation.py")
            return False
    else:
        print("❌ Installation failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
