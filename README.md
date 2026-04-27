# Pneumonia Detection & Anomaly Heatmapping

## Overview
This repository contains the source code for a senior capstone project developed at Utah Tech University. The project utilizes Convolutional Neural Networks (CNNs) to detect pneumonia anomalies in medical imaging datasets. 

To ensure the model's decisions are medically interpretable, the system integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to generate visual heatmaps, highlighting the specific regions of the scans that influenced the network's predictions.

## Features
* **Pneumonia Classification:** Automated detection of pneumonia indicators from medical scans using a pre-trained CNN architecture.
* **Visual Interpretability (Grad-CAM):** Generates diagnostic heatmaps overlaid on the original images to isolate anomalies and provide visual context for the model's output.
* **Remote Access (ngrok):** Configured to easily tunnel the local application or API to the public internet for remote testing and demonstration.

## Tech Stack
* **Language:** Python 
* **Machine Learning:** [TensorFlow / Keras]
* **Computer Vision:** OpenCV, Grad-CAM implementation
* **Deployment/Tunneling:** ngrok

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/FunSumTime/Senior_Project_medical_imaging_detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install and configure ngrok (if not already installed):
   * Download from [ngrok.com](https://ngrok.com/download) or install via a package manager.
   * Authenticate your ngrok agent with your auth token:
     ```bash
     ngrok config add-authtoken [your-auth-token]
     ```

## Usage
To run the model inference and access it remotely, follow these steps:

1. **Start the Local Application:**
   Run the main script that hosts the inference API or web interface. *(Update `app.py` and the port number if yours are different)*:
   ```bash
   python app.py
   ```
   *The local server should now be running (e.g., on `http://localhost:5000`).*

2. **Expose the Local Server with ngrok:**
   Open a **new** terminal window and start an ngrok HTTP tunnel pointing to the port your app is running on (e.g., port 5000):
   ```bash
   ngrok http 5000
   ```

3. **Access the Application Remotely:**
   Once ngrok is running, your terminal will display a Forwarding URL (e.g., `https://<random-string>.ngrok-free.app`). Copy this URL to access your pneumonia detection interface from anywhere.

## Project Status
**Phase:** Core pipeline completed. 
The current iteration successfully demonstrates pneumonia detection and heatmapping for medical imagery and can be easily shared remotely via ngrok. Future development may expand the CNN architecture to support generalized anomaly detection across a wider variety of datasets and conditions.

## Contributors
* **Austin Espinoza** - *Lead Developer / Computer Science Senior at Utah Tech University*
* **Aston Haycock** - *Architecture & Heatmap Collaboration*

## Acknowledgments
* Utah Tech University Computer Science Department
* Cutris Larsen
