#!/bin/bash
# Quick setup and test script for SNN Optical Flow project

echo "=========================================="
echo "SNN Optical Flow - Quick Setup"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Running setup verification tests..."
echo "=========================================="

# Run test script
python test_setup.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Quick start commands:"
echo "  1. Activate environment:   source venv/bin/activate"
echo "  2. Test training:          python train.py --config snn/configs/lightweight.yaml"
echo "  3. Monitor training:       tensorboard --logdir ./logs"
echo "  4. Evaluate model:         python evaluate.py --checkpoint checkpoints/best_model.pth"
echo ""
