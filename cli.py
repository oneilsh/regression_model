# python cli.py --data prediction_model_file.csv - Running this will pass the provided data file to twinsight-cli and handle errors


"""
CLI script for running the twinsight-cli command with a specified data file.
This script parses the command-line arguments, executes the twinsight-cli command,
and handles any errors that may occur during execution.
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Run the twinsight-cli command with provided data file."
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the prediction model data file (e.g., prediction_model_file.csv)'
    )
    args = parser.parse_args()

    command = ["twinsight-cli", "--data", args.data]
    try:
        result = subprocess.run(command, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("twinsight-cli command not found. Please ensure it is installed and accessible.")
        sys.exit(1)

if __name__ == "__main__":
    main()
