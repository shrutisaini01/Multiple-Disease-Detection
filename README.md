
# Multiple Disease Prediction System
website link: https://prediction-app-shruti.streamlit.app/

This web application, built with Streamlit, predicts the likelihood of three diseases: **Diabetes**, **Heart Disease**, and **Parkinson's Disease**. Each prediction is powered by a separate machine learning model trained specifically for that condition, and the models are stored as `.sav` files.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Model Information](#model-information)

## Overview

The application provides a streamlined, user-friendly interface for disease risk prediction:
- **Diabetes Prediction:** Considers health metrics like pregnancies, glucose level, BMI, and more.
- **Heart Disease Prediction:** Analyzes factors such as age, cholesterol levels, and heart rate.
- **Parkinson's Prediction:** Uses voice and movement metrics, including jitter and shimmer values.

Each disease model is pre-trained, loaded from separate `.sav` files, and provides real-time predictions based on user input.

## Features

- **Interactive Sidebar Navigation:** Easily switch between the three disease prediction models.
- **Custom Input Fields:** Each disease model has specific fields for user data entry.
- **Real-Time Prediction Output:** Instant feedback on disease likelihood.

## Requirements

- **Python 3.7+**
- **Streamlit**
- **streamlit-option-menu**
- **scikit-learn** (for loading the pre-trained models)
- **Pickle** (used to load serialized models)

These dependencies are specified in the `requirements.txt` file. Installing this file will automatically set up your environment with all necessary packages.

## Setup and Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/multipleDiseasePrediction.git
    cd multipleDiseasePrediction
    ```

2. **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

   The `requirements.txt` file includes:
   - `streamlit`: The main framework for building the web application.
   - `streamlit-option-menu`: To create the sidebar navigation.
   - `scikit-learn`: Required for handling and making predictions with the pre-trained models.
   - Additional dependencies for handling machine learning models.

3. **Place the Trained Model Files:**
   - The project expects separate saved models for each disease in the `savedModels` directory:
     - `trainedModel_RF.sav` for Diabetes prediction.
     - `heartTrainedModel.sav` for Heart Disease prediction.
     - `parkinsonTrainedModel.sav` for Parkinson's Disease prediction.
   - If using a different folder structure, update the file paths in the code.

4. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Project Structure

- `app.py`: Main Streamlit application script.
- `savedModels/`: Folder containing `.sav` files for each disease model.
  - `trainedModel_RF.sav` – Diabetes Prediction Model
  - `heartTrainedModel.sav` – Heart Disease Prediction Model
  - `parkinsonTrainedModel.sav` – Parkinson's Prediction Model
- `requirements.txt`: Contains required libraries for setting up the project.

## How to Use

1. Run the app and open it in your browser (usually at `http://localhost:8501`).
2. Use the sidebar to select a disease prediction model.
3. Input relevant health data for the chosen condition in the provided fields.
4. Click the `Predict` button to get an immediate prediction.

## Model Information

The models are trained using public datasets and saved as `.sav` files for quick loading:
- **Diabetes Prediction Model**: Trained on the PIMA Indian Diabetes Dataset.
- **Heart Disease Prediction Model**: Uses data from the UCI Heart Disease Dataset.
- **Parkinson's Prediction Model**: Based on voice metrics from the UCI Parkinson’s dataset.

These models use a machine learning algorithm (Random Forest) and were developed for educational purposes.

