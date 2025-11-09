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

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "⚠️  GitHub CLI (gh) is not installed"
    echo "Installing GitHub CLI..."
    
    # Check if Homebrew is installed
    if command -v brew &> /dev/null; then
        echo "Installing via Homebrew..."
        brew install gh
        if [ $? -eq 0 ]; then
            echo "✅ GitHub CLI installed successfully!"
        else
            echo "❌ Error installing GitHub CLI via Homebrew"
            echo "You can install it manually from: https://cli.github.com/"
        fi
    else
        echo "❌ Homebrew is not installed"
        echo "Please install GitHub CLI manually from: https://cli.github.com/"
        echo "Or install Homebrew first: https://brew.sh/"
    fi
else
    echo "✅ GitHub CLI found: $(gh --version | head -n 1)"
fi

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




