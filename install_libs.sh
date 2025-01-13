#!/usr/bin/env bash
set -e  # Exit on first error

# 1. Clone OpenSpiel (if not present).
if [ ! -d "open_spiel" ]; then
  echo "Cloning OpenSpiel into the current directory..."
  git clone https://github.com/deepmind/open_spiel.git
fi

# 2. Enter the open_spiel folder.
cd open_spiel

# 3. Run the optional install script (installs system packages).
echo "Running install.sh..."
./install.sh

# 4. Install Python requirements (adjust for your Python version if needed).
echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p ./build
cd build
CXX=g++ cmake -DPython_TARGET_VERSION=3.6 -DCMAKE_CXX_COMPILER=${CXX} ../open_spiel
