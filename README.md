# Predicting KRW Movements for Travel Decisions: Will the Won Be Higher or Lower in 6 Months?

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://krw-exchange-prediction-ftadvxntbytubfqgjghx9z.streamlit.app/)

## Project Overview
This project predicts whether the Korean Won (KRW) exchange rate will be higher or lower 6 months from now using daily global asset and macro-economic data.
- **Target variable**: `1` if the exchange rate increases (KRW weakens), `0` otherwise.
- **Goal**: To provide insights for travel decisions (e.g., buying USD now or later).

## Architecture
1. **Machine Learning Pipeline (`train_model.py`)**: Data cleaning, Time-Series cross-validation, and Random Forest model training.
2. **Interactive UI (`app.py`)**: A Streamlit dashboard for performing What-If analysis based on current economic indicators.

## How to Run locally
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Train the model (outputs `best_model.pkl`):
   ```sh
   python train_model.py
   ```
3. Run the Streamlit Dashboard:
   ```sh
   streamlit run app.py
   ```
