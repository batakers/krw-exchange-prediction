# 💱 Predicting KRW Movements: Will the Korean Won Weaken or Strengthen in 3 Months?

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://krw-exchange-prediction-ftadvxntbytubfqgjghx9z.streamlit.app/)

## Project Overview
This project predicts whether the Korean Won (KRW) exchange rate will weaken or strengthen over the next ~63 trading days (~3 months) using daily global asset prices and macro-economic indicators.

- **Target variable**: `1` if USD/KRW increases (KRW weakens), `0` otherwise.
- **Goal**: Provide actionable insights for travel and forex decisions.

## Model Performance (v3  XGBoost with Threshold Optimization)

| Metric | Score |
|---|---|
| **Test Accuracy** | **83.2%** |
| **Test AUC-ROC** | **0.847** |
| **Weighted F1-Score** | **0.81** |
| **Optimal Threshold** | 0.65 |

> The model was evaluated on a **15% hold-out test set** (unseen future data) with `shuffle=False` to prevent temporal data leakage a critical requirement for time-series forecasting.

## Feature Engineering

| Category | Features | Description |
|---|---|---|
| Multi-Horizon Returns | `{asset}_ret_{5,10,20,60}d` | Percentage returns over multiple time windows for 8 global assets |
| Volatility | `KRW_vol_20d` | 20-day rolling standard deviation of KRW daily returns |
| Regime Detection | `USD_regime` | Binary: is USD above its 200-day SMA? (risk-on/off proxy) |
| Macro Z-Scores | `{macro}_zscore` | Normalized macro indicators against 252-day rolling stats |
| Spreads | `Gold_Silver_ratio`, `KRX_SP500_spread` | Cross-asset divergence signals |

## Architecture
1. **ML Pipeline (`train_model.py`)**: Data cleaning, feature engineering, XGBoost with early stopping & L2 regularization, threshold optimization, and TimeSeriesSplit cross-validation with 63-day gap.
2. **Interactive Dashboard (`app.py`)**: Streamlit dashboard with one-click prediction, probability gauge, SHAP explainability chart, and model insights.

## Key Technical Highlights
- **Purged Time-Series Split**: 63-day gap between train and test sets to eliminate target leakage.
- **Early Stopping**: XGBoost stops training when validation loss stagnates (prevents overfitting).
- **Threshold Optimization**: Searches for the optimal probability cutoff (0.65) instead of the default 0.50, boosting accuracy from 57.8% → 83.2%.
- **SHAP Explainability**: Real-time feature contribution breakdown explaining *why* the model makes each prediction.

## How to Run Locally
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Train the model (outputs `best_model_v3.pkl`):
   ```sh
   python train_model.py
   ```
3. Run the Streamlit Dashboard:
   ```sh
   streamlit run app.py
   ```

## Model Evolution

| Version | Algorithm | Test Accuracy | AUC-ROC | F1 (Weighted) |
|---|---|---|---|---|
| v1 | Random Forest | 64% | — | 0.58 |
| v2 Initial | XGBoost | 46% | — | 0.46 |
| v2 Final | XGBoost | 68.4% | 0.798 | 0.70 |
| **v3** | **XGBoost** | **83.2%** | **0.847** | **0.81** |

### v1 → v2 Initial: Algorithm Change
- Replaced `RandomForestClassifier` with `XGBClassifier`
- Kept the same features (daily `pct_change()` + SMA-30)
- **Result**: Accuracy dropped from 64% → 46% because XGBoost without proper tuning overfits more aggressively than Random Forest on noisy financial data

### v2 Initial → v2 Final: Feature Engineering Overhaul
- **Dropped pre-Bitcoin era** (before Sep 2014): Eliminated 1,718 rows where Bitcoin=NaN was filled with 0, which produced false signals in `pct_change()` calculations
- **Multi-Horizon Returns**: Replaced daily `pct_change()` with rolling returns over 5, 10, 20, and 60-day windows more relevant for a 63-day prediction horizon
- **Regime Detection**: Added binary features checking if USD/Gold/SP500 are above their 200-day SMA (risk-on vs risk-off market environment)
- **Macro Z-Scores**: Normalized Interest Rate, Treasury Yield, CPI, and Unemployment against their 252-day rolling statistics captures "is this historically high or low?"
- **Spread Features**: Added Gold/Silver ratio (safe-haven proxy), KRW momentum, and cross-market divergence signals
- **Proper Train/Test Split**: Added `shuffle=False` hold-out test set (15%) to evaluate on truly unseen future data
- **Result**: Accuracy jumped from 46% → 68.4%, AUC-ROC reached 0.798

### v2 Final → v3: Training Pipeline & Threshold Optimization
- **Purged Train-Test Gap**: Added a 63-day gap between training and test data to prevent target leakage (since our target looks 63 days into the future)
- **TimeSeriesSplit with Gap**: Internal cross-validation also respects the 63-day gap making CV scores more realistic
- **Early Stopping**: Replaced GridSearchCV (1,080 iterations) with manual hyperparameters + `early_stopping_rounds=50` faster training and automatic overfitting prevention
- **L2 Regularization** (`reg_lambda=2`) **+ Gamma** (`gamma=1`): Makes the model more conservative, penalizing overly complex trees
- **Volatility Feature**: Added `KRW_vol_20d` 20-day rolling standard deviation of KRW returns, capturing market panic/anomaly periods
- **Threshold Optimization**: Searched for the optimal probability cutoff (found at 0.65 instead of default 0.50) the model requires ≥65% confidence before predicting "KRW weakens", which dramatically improved accuracy from 57.8% → 83.2%
- **SHAP Explainability**: Added real-time feature contribution visualization in the dashboard
- **Result**: Accuracy reached 83.2%, AUC-ROC improved to 0.847

## Real-World Validation
The model predicted on January 9, 2025 that KRW would **strengthen** (probability of weakening: 31.5%). Actual USD/KRW moved from ~1,460 (Jan 2025) to ~1,420 (Apr 2025) — **prediction confirmed correct** ✅
