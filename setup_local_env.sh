# Set up the working environment without Dockerfile
# FIXME: be tested
#!/bin/bash
# Exit on any error
set -e  # should we do this?

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
# Use the same name as the venv directory for conda environment
CONDA_ENV_NAME="$(basename "${VENV_DIR}")"

echo "Setting up aria environment..."

# Ask user to choose between venv and conda
echo "Please choose your preferred environment manager:"
echo "1) Python venv (default)"
echo "2) Conda"
read -p "Enter your choice (1/2): " env_choice

# Default to venv if no choice is made
env_choice=${env_choice:-1}

# 1. Create virtual environment if it doesn't exist
if [ "$env_choice" = "1" ]; then
    # Using Python venv
    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment using venv..."
        python3 -m venv "${VENV_DIR}"
    else
        echo "Virtual environment already exists."
    fi

    # Activate virtual environment
    echo "Activating venv virtual environment..."
    source "${VENV_DIR}/bin/activate"

elif [ "$env_choice" = "2" ]; then
    # Check if conda is installed
    if ! command -v conda &> /dev/null; then
        echo "Conda is not installed or not in PATH. Please install conda first."
        exit 1
    fi

    # Check if the conda environment exists
    if ! conda info --envs | grep -q "${CONDA_ENV_NAME}"; then
        echo "Creating conda environment '${CONDA_ENV_NAME}'..."
        conda create -y -n "${CONDA_ENV_NAME}" python=3
    else
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
    fi

    # Activate conda environment
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# 2. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

# 3. Download solver binaries
echo "Downloading solver binaries..."
(cd "${SCRIPT_DIR}/bin_solvers" && python "download.py")
# cvc5, mathsat, z3.

# 4. Run tests
echo "Running tests..."
if [ -f "${SCRIPT_DIR}/unit_test.sh" ]; then
    bash "${SCRIPT_DIR}/unit_test.sh"
else
    echo "Warning: unit_test.sh not found"
fi

echo "Setup completed successfully!"
