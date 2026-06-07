# Electricity Theft Detection through Load Signature Analysis

A Streamlit-based system for detecting electricity theft in Jammu & Kashmir using load signature analysis, feature engineering, and machine learning.

## Overview

This repository includes:

- `app.py`: Streamlit dashboard for data exploration and electricity theft prediction.
- `src/`: Core modules for preprocessing, feature engineering, model training, and theft detection.
- `data/`: Dataset generator and sample consumer data file.
- `models/`: Trained model results and saved artifacts.

## Features

- Synthetic dataset generation for J&K electricity consumers.
- Feature engineering from load signatures.
- Machine learning models for theft detection.
- Interactive Streamlit dashboard with visualizations and risk indicators.

## Requirements

The project requires Python and the packages listed in `requirements.txt`.

## Setup and Run

On Windows, run the provided helper:

```bat
setup_and_run.bat
```

This script will:

1. Install required packages from `requirements.txt`
2. Generate the J&K electricity dataset
3. Train machine learning models
4. Launch the Streamlit dashboard

Alternatively, run manually:

```bat
pip install -r requirements.txt
python data/generate_dataset.py
python src/model_training.py
streamlit run app.py
```

Then open the displayed URL in your browser (typically `http://localhost:8501`).

## Project Structure

- `app.py`: UI and dashboard layout
- `src/preprocessing.py`: Data preprocessing logic
- `src/feature_engineering.py`: Feature creation functions
- `src/model_training.py`: Model training and evaluation
- `src/detection.py`: Prediction and model selection utilities
- `data/generate_dataset.py`: Synthetic dataset generator

## Notes

- The dataset file `data/jk_electricity_consumers.csv` is used by the dashboard.
- The dashboard and model code are designed for experimentation and demonstration.
- Keep package versions in `requirements.txt` to ensure compatibility.
