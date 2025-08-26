# Installation Guide for All of Us (AoU) Workbench

This guide provides step-by-step instructions for installing twinsight_model v0.2.0 in the AoU environment.

## Quick Start (Recommended)

### Step 1: Install from GitHub
```bash
pip install git+https://github.com/yourusername/regression_model.git@v0.2.0
```

### Step 2: Test Installation
```python
# Run this in a new notebook cell
exec(open('test_install.py').read())
```

## Detailed Installation (If Quick Start Fails)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/regression_model.git
cd regression_model
```

### Step 2: Install with Minimal Dependencies
```bash
# Install only our additional dependencies first
pip install lifelines>=0.30.0 protobuf>=3.20.0,<4.0.0

# Then install the package in development mode
pip install -e .
```

### Step 3: Verify Installation
```python
# Test in a notebook
import twinsight_model
print(f"Installed version: {getattr(twinsight_model, '__version__', 'unknown')}")

# Run full test
exec(open('test_install.py').read())
```

## Troubleshooting

### NumPy Version Conflicts
If you see "numpy 1.x cannot be run in numpy 2.x" warnings:
```bash
# This should NOT be needed with v0.2.0, but if it persists:
pip install numpy==1.24.4 --force-reinstall
```

### BigQuery Connection Issues
```python
# Test BigQuery connection
from google.cloud import bigquery
client = bigquery.Client()
print("BigQuery client created successfully")
```

### Import Errors
If you get import errors for standard packages:
1. Restart your kernel: Kernel → Restart Kernel
2. Try the installation again
3. If still failing, check if you're in the correct AoU environment

## Version Information

- **twinsight_model**: 0.2.0
- **Compatible with AoU base packages**:
  - pandas: 2.0.3
  - numpy: 1.24.4  
  - scikit-learn: 1.6.0
  - PyYAML: 6.0.1

## Next Steps

Once installation is successful:
1. Run `test_install.py` to verify everything works
2. Try the basic API test (see BASIC_TEST.md)
3. Proceed with your analysis workflow

## Getting Help

If installation fails:
1. Save the error output
2. Note your Python version: `python --version`
3. Note your pip version: `pip --version`
4. Check if you're in a virtual environment: `which python`
