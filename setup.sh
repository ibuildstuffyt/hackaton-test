#!/bin/bash

# Setup script for FPV Simulator
# This script helps you set up the environment

echo "=========================================="
echo "FPV Simulator Setup"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3.8 or higher from python.org"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed!"
    echo "Please install pip3"
    exit 1
fi

echo "✅ pip3 found"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take a minute..."
echo ""

pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dependencies installed successfully!"
else
    echo ""
    echo "❌ Error installing dependencies"
    echo "Try running: pip3 install --user -r requirements.txt"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Connect your DJI controller via USB"
echo "2. Test controller: python3 test_controller.py"
echo "3. Run simulator: python3 main.py"
echo ""
echo "See QUICK_START.md for detailed instructions"
echo ""




