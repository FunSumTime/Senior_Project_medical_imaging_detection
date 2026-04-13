#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

# Create the folder for your models
mkdir -p saved_models

# Download the DenseNet models from your GitHub Release
echo "Downloading Model 1..."
curl -L -o saved_models/densenet_model1_unfrozen.keras "https://github.com/FunSumTime/Senior_Project_medical_imaging_detection/releases/download/models-v1/densenet_model1_unfrozen.keras"

echo "Downloading Model 2..."
curl -L -o saved_models/densenet_model2_unfrozen.keras "https://github.com/FunSumTime/Senior_Project_medical_imaging_detection/releases/download/models-v1/densenet_model2_unfrozen.keras"

# Install your Python dependencies
pip install -r requirements.txt