@echo off
echo =================================================
echo  Electricity Theft Detection System - Setup
echo  Jammu and Kashmir Smart Grid
echo =================================================
echo.

echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
echo.

echo [2/4] Generating J^&K dataset (5,000 consumers)...
python data/generate_dataset.py
echo.

echo [3/4] Training ML models...
python src/model_training.py
echo.

echo [4/4] Launching Streamlit dashboard...
echo.
echo  Open http://localhost:8501 in your browser
echo.
streamlit run app.py
