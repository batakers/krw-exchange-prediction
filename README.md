# 💱 Predicting KRW Movements: Will the Korean Won Weaken or Strengthen in 3 Months?

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://krw-exchange-prediction-ftadvxntbytubfqgjghx9z.streamlit.app/)

## Project Overview
This project predicts whether the Korean Won (KRW) exchange rate will weaken or strengthen over the next ~63 trading days (~3 months) using daily global asset prices and macro-economic indicators.

- **Target variable**: `1` if USD/KRW increases (KRW weakens), `0` otherwise.
- **Goal**: Provide actionable insights for travel and forex decisions.

## Model Performance (v3.1  XGBoost, corrected evaluation)

> **Update (Nov 2026):** The earlier headline of 80.5% accuracy came from an evaluation with two bugs: the last 63 rows of the hold-out set had no real 3-month outcome but were labelled "strengthen", and the decision threshold was tuned on the test set itself. Both are fixed and the numbers below come from the corrected pipeline. Details in [Evaluation Fixes (Nov 2026)](#evaluation-fixes-nov-2026).

| Metric | Score |
|---|---|
| **Test Accuracy** (threshold chosen on validation) | **78.2%** |
| **Best constant guess** (always "strengthen", known only in hindsight) | 72.7% |
| **Test AUC-ROC** | **0.711** |
| **Weighted F1-Score** | 0.76 |
| **Recall for "KRW weakens"** | 0.37 |
| **Threshold** (chosen on the validation window) | 0.60 |
| **Test period** | 2023-06-08 to 2024-10-11 (344 trading days) |

> The model was evaluated on a **15% hold-out test set** (unseen future data) with `shuffle=False` and a 63-day purge gap. Because 72.7% of test days were "KRW strengthens", accuracy alone overstates skill: the model beats a constant guess by about 5.5 points and catches 37% of the "weaken" periods. AUC-ROC (0.71) is the more informative number. A **Lookahead Bias Fix** shifts government-released macro data (CPI, Unemployment) by 30 trading days to reflect real-world publication lag.

> ⚠️ **Data note:** the `KRW` column in `data/XAU BTC Silver SP500 dataset.csv` tracks the market USD/KRW rate in 2008–2009 (peaks near 1,508 and 1,571) but diverges from about 2015 onward. For example, it reads ~1,125 at the end of 2024, when the market rate was around 1,450–1,470. All results in this README are on the dataset as provided and should be re-run once the KRW series is re-sourced (e.g. FRED `DEXKOUS` or Yahoo Finance `KRW=X`).

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
- **Purged Time-Series Split**: 63-day gap between train and test sets (and before the validation window) to eliminate target leakage.
- **Early Stopping**: XGBoost stops training when validation loss stagnates (prevents overfitting).
- **Threshold Selection on Validation**: The probability cutoff is chosen on a purged validation window before the test period (0.60 in the current run), never on the test set. On the hold-out it moves accuracy from 71.8% (default 0.50) to 78.2%.
- **SHAP Explainability**: Real-time feature contribution breakdown explaining *why* the model makes each prediction.
- **Lookahead Bias Fix**: Government-released macro data (CPI, Unemployment) shifted by 30 trading days to reflect real-world publication lag.
- **Nested Cross-Validation**: `RandomizedSearchCV` embedded inside each Walk-Forward fold for dynamic hyperparameter optimization.

## Walk-Forward Validation: Beyond a Single Lucky Score

> *A model that scores 80%+ on one test set might just be lucky. Walk-Forward Validation forces it to prove consistency across multiple market regimes.*

Working with static historical financial data presents a unique danger: a model might perform exceptionally well on a single 15% hold-out test set purely by chance, simply because that specific time period had "easy" or highly trending market conditions.

To prove that the hold-out accuracy is not just a "lucky final score", this project employs **Nested Walk-Forward Validation (Expanding Window Backtesting)** across 5 distinct time folds:

1. The model is trained on a historical window, evaluated on the next unseen block of time, and then the window **expands**.
2. A **63-day purged gap** is enforced between train and test sets to absolutely guarantee zero data leakage.
3. Inside each fold, `RandomizedSearchCV` (20 combinations × 3 inner folds) dynamically searches for the best hyperparameters no single static configuration is assumed optimal for all market conditions.

### Walk-Forward Results

*Training-majority baseline* = always predict the most common direction in that fold's training data (a fair, no-hindsight guess). *Best constant* = always predict whichever direction was more common in the test fold (hindsight, a strict bar).

| Fold | Train Period | Test Period | Accuracy | Training-majority baseline | Best constant | AUC-ROC | F1-Weighted |
|---|---|---|---|---|---|---|---|
| 1 | 2015-10 to 2017-01 | 2017-04 to 2018-10 | 60.1% | 50.7% | 50.7% | 0.666 | 0.592 |
| 2 | 2015-10 to 2018-07 | 2018-10 to 2020-04 | 48.3% | 28.4% | 71.7% | 0.547 | 0.494 |
| 3 | 2015-10 to 2020-01 | 2020-04 to 2021-10 | 50.9% | 64.0% | 64.0% | 0.610 | 0.503 |
| 4 | 2015-10 to 2021-07 | 2021-10 to 2023-04 | 45.4% | 68.5% | 68.5% | 0.554 | 0.449 |
| 5 | 2015-10 to 2023-01 | 2023-04 to 2024-10 | **73.0%** | 31.8% | 68.2% | **0.685** | **0.700** |
| **Mean** | | | **55.5%** | 48.7% | 64.6% | **0.612** | **0.548** |
| **Std** | | | ±10.0% | | | ±0.056 | ±0.089 |

### Honest Interpretation

- **AUC-ROC is above 0.50 in every fold (0.55–0.69).** The model ranks "weaken" days above "strengthen" days better than chance in every regime, so there is a real but weak signal.
- **Accuracy is less convincing.** It beats the training-majority guess in 3 of 5 folds and the best constant guess in only 2 of 5 (folds 1 and 5). In folds 3 and 4, simply repeating the training majority would have scored higher.
- **Fold 5 (most training data)** is the strongest at 73.0%, which is consistent with the model improving as it sees more data, but one fold is not enough to prove that trend.
- The decision threshold (0.60) comes from the main model's validation window and is reused in every fold rather than re-tuned per fold.

> **Takeaway:** Treat this as a research prototype with a weak, consistent ranking signal, not as a tool for real currency or travel decisions. The next steps are re-sourcing the KRW series (see the data note above) and reporting results against these baselines.

## Evaluation Fixes (Nov 2026)

An audit of the evaluation found two bugs that inflated the published 80.5% accuracy. Both are fixed in `train_model.py`, `experiment_bitcoin.py` and the dashboard backtest in `app.py`.

1. **Fake labels at the end of the data.** The target compares today's KRW with the value 63 rows later. For the last 63 rows that future value is `NaN`, and `NaN > x` evaluates to `False`, so those rows were labelled 0 ("strengthen") and survived `dropna()`. They made up 63 of the 354 test rows (18%). They are now set to `NaN` and dropped.
2. **Threshold tuned on the test set.** `find_optimal_threshold` was called with the test labels. It now runs on a validation window: the last 10% of the training data, with a 63-day purge gap before it (the same gap now also separates the training rows from that window).
3. **Baselines.** The hold-out and every walk-forward fold now report a training-majority baseline and the best constant guess, and the model file stores these metrics.
4. **Dashboard.** Model Insights reads the hold-out metrics from `best_model_v3.pkl` instead of hardcoded numbers, and the January 2025 case study computes the model's call and scores it on the 63-day horizon.
5. **Compatibility.** `experiment_bitcoin.py` uses `.ffill()` instead of `fillna(method='ffill')`, which pandas 3 removed.

Results were produced with pandas 3.1.1, scikit-learn 1.9.1 and xgboost 3.4.1. In the same environment the old pipeline gives 81.9% accuracy and 0.775 AUC-ROC (vs the published 80.5% and 0.746), so library versions shift results slightly; the before/after comparison above uses one environment.

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

## Model Evolution: The Journey

| Version | Algorithm | Prediction Horizon | Test Accuracy | AUC-ROC | F1 (Weighted) |
|---|---|---|---|---|---|
| v1 | Random Forest | **6 months** (126 days) | 64% | - | 0.58 |
| v2 Initial | XGBoost | **3 months** (63 days) | 46% | - | 0.46 |
| v2 Final | XGBoost | 3 months (63 days) | 68.4% | 0.798 | 0.70 |
| v3 | XGBoost | 3 months (63 days) | 80.5%† | 0.746† | 0.78† |
| **v3.1 (evaluation fix)** | **XGBoost** | **3 months (63 days)** | **78.2%** | **0.711** | **0.76** |

*† Measured with 63 mislabelled test rows and a threshold tuned on the test set. v1–v3 figures are kept as a record of the project's history and were all measured under the old evaluation. See [Evaluation Fixes (Nov 2026)](#evaluation-fixes-nov-2026).*

---

### ⚠️ Model Limitations & Market Realities

While this project demonstrates strong predictive power based on macroeconomic fundamentals, real-world forex markets are influenced by factors beyond pure data. We transparently acknowledge the following operational blindspots:

1. **Central Bank Interventions**: The Bank of Korea (BOK) actively intervenes in the FX market to stabilize the Won when it hits psychological levels (e.g., 1,400 or 1,450). A purely macro-driven ML model is blind to these political support/resistance actions.
2. **Geopolitical Shocks**: Fluctuations driven by unexpected geopolitical events (e.g., North Korean tensions) are not instantly captured by historical numeric features.
3. **Forex Spreads (Tourist Warning)**: While the model might predict a 1% strengthening of the KRW, physical money changers and retail banks typically charge a 2% to 4% transaction spread. Travelers should only delay currency exchange if the predicted movement magnitude clearly outweighs the bank's spread costs.
4. **Holiday Liquidity Gaps**: Major Korean holidays (Chuseok and Seollal) cause severe drops in market liquidity, often leading to unpredictable price gaps when markets reopen. The model currently treats these days as standard trading periods.

---

### v1: The Starting Point (Random Forest, 6-Month Horizon)

The project began with a simple goal: predict whether the Korean Won would weaken or strengthen **6 months (~126 trading days) into the future** using a `RandomForestClassifier`. The features were basic: daily percentage changes (`pct_change()`) and 30-day Simple Moving Averages (SMA-30) across 12 global economic indicators.

The initial results looked reasonable on paper with **64% accuracy**. However, a deeper analysis revealed a critical flaw: the model was evaluated on the **same data it was trained on** (full dataset evaluation), meaning the "64%" was heavily inflated by data leakage. The model wasn't truly learning patterns; it was partially memorizing the training data.

**Features used**: Daily `pct_change()`, SMA-30 for Gold/SP500/Interest Rate  
**Problems identified**: No proper train/test split, Bitcoin NaN filled with 0 creating false signals, 6-month horizon was too ambitious for the available features

---

### v2 Initial: The Algorithm Switch Gone Wrong (XGBoost, 46%)

To improve performance, the algorithm was upgraded from Random Forest to `XGBClassifier` (XGBoost), a more powerful gradient boosting algorithm. At the same time, the prediction horizon was shortened from **6 months to 3 months (63 trading days)** because:
1. Predicting 6 months ahead with daily macro indicators was too noisy
2. A 3-month window is more actionable for travel and forex decisions
3. Shorter horizons generally produce more reliable predictions

However, the result was shocking: accuracy **dropped to 46%**, which is worse than flipping a coin (50%). What went wrong?

The problem was **severe overfitting**. When evaluated on the full dataset, XGBoost showed **100% accuracy** (a classic red flag known as "The Memorization Monster"). But when a proper hold-out test set was introduced with `shuffle=False`, the model's true performance was exposed: it was essentially guessing randomly on unseen future data.

**Root cause**: XGBoost is a much more aggressive learner than Random Forest. Without proper regularization and with weak features (daily pct_change), it memorized every detail of the training data but learned nothing generalizable.

---

### v2 Final: The Feature Engineering Breakthrough (68.4%)

The key realization: **the quality of features matters far more than the choice of algorithm**. The entire feature engineering pipeline was rebuilt from scratch:

1. **Dropped pre-Bitcoin era** (before Sep 2014): Eliminated 1,718 rows where Bitcoin=NaN was filled with 0. This was silently creating false "0% return" signals that corrupted the model's learning
2. **Multi-Horizon Returns**: Replaced single-day `pct_change()` with rolling returns over 5, 10, 20, and 60-day windows. Daily returns are too noisy for a 63-day prediction; multi-horizon returns capture trends at the right time-scale
3. **Regime Detection**: Added binary features checking if USD/Gold/SP500 are above their 200-day SMA (a standard technique to identify risk-on vs risk-off market environments)
4. **Macro Z-Scores**: Instead of feeding raw Interest Rate or CPI values, they were normalized against their 252-day (1-year) rolling statistics. This transforms the question from "what is the interest rate?" to "is this interest rate *historically high or low*?"
5. **Spread Features**: Added Gold/Silver ratio (safe-haven demand proxy), KRW momentum (mean-reversion signal), and cross-market divergence indicators

The model was now evaluated properly with a **15% hold-out test set** (`shuffle=False` to respect chronological order).

**Result**: Accuracy jumped from 46% to **68.4%**, and AUC-ROC reached **0.798**. The model was finally learning real patterns, not memorizing noise.

---

### v3: The Final Optimization (80.5%, later corrected to 78.2%)

The v2 model had strong discriminative power (AUC-ROC 0.798) but still had room for improvement in its training pipeline. Four critical upgrades were made:

**1. Purged Train-Test Gap (Anti-Leakage)**  
Since the target looks 63 days into the future, the last 63 rows of training data have targets that temporally overlap with the test set. A 63-day gap was added between train and test data, and `TimeSeriesSplit(gap=63)` was applied to internal cross-validation. This is an industry-grade technique known as *purged cross-validation*.

**2. Lookahead Bias Fix (Point-in-Time Compliance)**  
Government-released macro data (`Inflation_CPI`, `Unemployment`) is published with a ~30-day lag. The original pipeline used these values as if they were available in real-time, creating a subtle lookahead bias. After applying a 30-trading-day shift, accuracy dropped from 83.2% to 80.5% (old evaluation), suggesting that ~3% of the original accuracy was attributable to this bias. This fix ensures the model only uses information that would have been publicly available at prediction time.

**3. Regularization & Early Stopping (Anti-Overfitting)**  
- `gamma=1`: Penalizes unnecessary tree splits
- `reg_lambda=2`: L2 regularization on leaf weights
- `early_stopping_rounds=50`: Model automatically stops training when validation performance stagnates
- Added `KRW_vol_20d` (volatility feature) to capture market panic/anomaly periods

**4. Threshold Optimization (The Game-Changer)**  
The default classification threshold of 0.50 was suboptimal for this imbalanced dataset. A systematic search across thresholds revealed that **0.55** maximizes accuracy. The model now requires ≥55% confidence before predicting "KRW weakens."

This single change boosted accuracy from 76.5% (default threshold) to **80.5%** (optimal threshold), while maintaining the same AUC-ROC of **0.746**.

> **Correction (Nov 2026):** this threshold search was run on the test set, so the 80.5% was optimistic. The threshold is now chosen on a validation window before the test period. With the label fix as well, the hold-out result is 78.2% accuracy (71.8% at 0.50) and AUC-ROC 0.711.

**4. SHAP Explainability**  
Added real-time feature contribution visualization in the dashboard, answering "Why does the model predict this?" for every single prediction.

---

### Full Comparison

| Aspect | v1 | v2 Initial | v2 Final | v3 (Final) |
|---|---|---|---|---|
| **Algorithm** | Random Forest | XGBoost | XGBoost | XGBoost |
| **Prediction Horizon** | 6 months (126 days) | 3 months (63 days) | 3 months (63 days) | 3 months (63 days) |
| **Features** | Daily pct_change, SMA-30 | Daily pct_change, SMA-30 | Multi-horizon returns, Regime, Z-Scores, Spreads | Multi-horizon returns, Regime, Z-Scores, Spreads, Volatility |
| **Data Period** | Full (2000-2025) | Full (2000-2025) | Post-Bitcoin (2014+) | Post-Bitcoin (2014+) |
| **Train/Test Split** | No hold-out | Hold-out 15% | Hold-out 15% | Hold-out 15% + 63-day gap |
| **Hyperparameter Tuning** | GridSearchCV | GridSearchCV | GridSearchCV (216 combos) | Manual + Early Stopping |
| **Regularization** | Default | Default | Default | gamma=1, reg_lambda=2 |
| **Threshold** | Default (0.50) | Default (0.50) | Default (0.50) | Optimized (0.55) on test set† |
| **Lookahead Bias Fix** | None | None | None | CPI & Unemployment shifted 30 days |
| **Explainability** | None | None | None | SHAP |
| **Test Accuracy** | 64%* | 46% | 68.4% | **80.5%**† |
| **Test AUC-ROC** | - | - | 0.798 | **0.746**† |

*\*v1 accuracy was inflated due to the lack of a proper hold-out test set*
*† Superseded by v3.1: threshold chosen on validation (0.60), mislabelled rows removed, 78.2% accuracy and 0.711 AUC-ROC on the hold-out.*

### Key Takeaways

1. **A powerful algorithm alone is not enough.** Upgrading from Random Forest to XGBoost without fixing the underlying data issues made the model *worse* (64% to 46%). The algorithm amplified the noise instead of learning from it.

2. **Feature engineering is the most impactful lever.** The jump from 46% to 68.4% came entirely from redesigning *what* the model sees: multi-horizon returns, regime detection, and macro z-scores. The algorithm stayed the same.

3. **The prediction horizon must match the feature granularity.** Predicting 6 months ahead with daily price changes is like trying to forecast the weather next season by looking out the window today. Shortening to 3 months and using 5-60 day rolling returns created a much better signal-to-noise ratio.

4. **Where you tune the threshold matters as much as the threshold.** Tuning it on the test set lifted accuracy to 80.5%. Choosing it on a validation window gives 78.2% on the same kind of hold-out, and only about 5.5 points of that is above a constant "strengthen" guess.

5. **Fixing lookahead bias costs accuracy and that's a good thing.** Applying a 30-day shift to CPI and Unemployment reduced accuracy from 83.2% to 80.5% (both measured under the old evaluation). The drop shows the original model was partially "cheating" by using government data before it was actually published.

6. **Proper evaluation prevents false confidence.** Without a chronological hold-out test set and temporal gap, financial models will always appear more accurate than they truly are.

7. **Check the labels at the edges.** With a 63-day forward target, the last 63 rows have no outcome yet. `NaN > x` evaluates to `False` in pandas, so they were silently labelled "strengthen" and kept, which made 18% of the original test labels fake. Always drop rows whose future value is missing before building a forward-looking target.

## Real-World Validation

> *"Trust is earned, not claimed."* One live prediction cannot prove or disprove a model, but it should be scored honestly on the model's own horizon.

### Case Study: January 9, 2025 Prediction

| | Details |
|---|---|
| **Prediction Date** | January 9, 2025 (last row of the dataset) |
| **Model Says** | KRW will **Strengthen** (↓) over the next 63 trading days |
| **Probability of Weakening** | 18.9% (v3.1 model, threshold 0.60). The original v3 model gave 31.5%. |

### What Actually Happened?

| Period | Market USD/KRW | Movement vs prediction date |
|---|---|---|
| Early January 2025 (prediction made) | ~1,460 | - |
| April 9, 2025 (≈ +63 trading days, the model's horizon) | 1,484.1 close, the won's weakest level since March 2009 | ↑ KRW weaker |
| Later in Q2 2025 | The won recovered | After the 3-month window had closed |

**Result at the model's 63-day horizon: Incorrect** ❌

An earlier version of this README marked this case as correct by comparing against later prices (~1,420 and ~1,383), which fall outside the 63-day horizon. On the model's own 63-day horizon, the won had weakened during the April 2025 tariff shock. This is a good example of the limitations listed above: a sudden policy shock is not visible in the features.

> The interactive dashboard includes a **full backtesting section** where you can examine every prediction the model made on unseen test data, complete with hit rates and individual ✅/❌ results.

---

## Research: Data Trade-off Analysis Why Pre-2014 Data Was Discarded

> *A common critique of this project is: "You're throwing away six years of valuable macro-economic history (2008–2014), including the 2008 Financial Crisis, just to keep Bitcoin as a feature. Is that trade-off really worth it?"*

Instead of defending this decision with assumptions, we ran a **controlled experiment** to answer this question empirically.

### Experiment Design

Three scenarios were evaluated using the same training pipeline (XGBoost, threshold chosen on a validation window, 63-day gaps, 15% hold-out). The dataset starts in January 2008. Each scenario's hold-out is its own last 15%, so the windows share an end date but start at different dates.

| Scenario | Data Range | Bitcoin | Samples | Features |
|---|---|---|---|---|
| **A**: Current Approach | 2014–2025 | ✅ Included | 2,474 | 40 |
| **B**: Full History | 2008–2025 | ❌ Dropped | 4,009 | 36 |
| **C**: Hybrid | 2008–2025 | ✅ Zero-filled pre-2014 | 2,512 | 40 |

### Results (corrected evaluation, Nov 2026)

| Scenario | Test Accuracy | Best constant guess | AUC-ROC | F1-Weighted | Threshold (validation) |
|---|---|---|---|---|---|
| **A: 2014+ WITH Bitcoin** | 71.5% | 69.9% | **0.7345** | **0.7250** | 0.55 |
| B: 2008+ NO Bitcoin | 41.5% | 61.0% | 0.4847 | 0.2834 | 0.30 |
| C: 2008+ Bitcoin filled | **73.5%** | 69.0% | 0.6898 | 0.7089 | 0.60 |

*The previous version of this table (A: 80.5% / 0.746, B: 65.0% / 0.523, C: 75.7% / 0.761) was measured with mislabelled test rows and thresholds tuned on the test set.*

**🏆 Winner by AUC-ROC: Scenario A.** Scenario C is close and has slightly higher accuracy, so the margin is smaller than the original write-up suggested.

### Analysis

1. **Removing Bitcoin collapses model performance.** Scenario B's AUC-ROC of 0.4847 is no better than chance (0.50), and its accuracy (41.5%) is far below a constant guess (61.0%). Despite having about **60% more data** (4,009 vs 2,474 samples), the model without Bitcoin cannot distinguish between KRW weakening and strengthening periods in this dataset.

2. **The 2008–2014 history does not help on its own.** Adding pre-Bitcoin macro history (including the Global Financial Crisis, European Debt Crisis, and Taper Tantrum) without Bitcoin *degrades* performance.

3. **The Hybrid result is mixed.** Scenario C (full history + Bitcoin zero-filled before 2014) ranks days less well than A (AUC 0.69 vs 0.73) but scores slightly higher accuracy (73.5% vs 71.5%). Against a constant guess, both add only a few points (C +4.5, A +1.6). Keeping Bitcoin clearly matters; whether to keep the 2008–2014 rows is a closer call.

*These are single hold-out results on the dataset as provided; see the data note at the top about the KRW series.*

### Conclusion: A Market Regime Shift

This experiment provides **empirical evidence** of a fundamental market regime shift around 2014:

- **Pre-2014**: Currency markets were driven primarily by traditional macro fundamentals (interest rates, trade balances, central bank policy). Bitcoin did not exist as a meaningful asset class.
- **Post-2014**: The rise of cryptocurrency as a global asset class introduced Bitcoin as a powerful **proxy for global risk appetite and liquidity conditions**. Bitcoin's correlation with emerging market currencies (including KRW) during risk-on/risk-off events makes it an indispensable feature.

> **Keeping Bitcoin is a data-driven design choice, supported by a ~25-point AUC-ROC gap (0.73 vs 0.48) between keeping and removing it. Discarding the 2008–2014 rows is a closer call: the zero-filled hybrid comes within 0.05 AUC and slightly ahead on accuracy.**

*The full experiment code is available in [`experiment_bitcoin.py`](experiment_bitcoin.py).*

---

## 🚀 Future Roadmap: From Script to Production-Grade System
> **Status:** *Work in Progress (On-going)*

While the current model shows a weak but consistent signal and a carefully engineered feature set, the architecture is currently a static, local script. The next phase of this project is dedicated to transforming this prototype into a fully automated, production-grade **End-to-End MLOps Pipeline**.

### Phase 1: Data Engineering & Automated ETL 
*Current State: Static CSV file.*
- [ ] **Dynamic Extraction (API Integration):** Replace the static `dataset.csv` with a daily automated extraction script using `yfinance` for global asset prices and `fredapi` for macro-economic indicators.
- [ ] **Upstream Feature Engineering:** Move the calculation of multi-horizon returns, volatility bands, and z-scores from the application runtime into the extraction pipeline to ensure the Streamlit dashboard remains lightweight.
- [ ] **Lightweight Database:** Implement a relational database (e.g., **SQLite** or **PostgreSQL**) to store and serve the cleaned, transformed daily data.

### Phase 2: MLOps & Experiment Tracking
*Current State: Manual hyperparameter tuning and model saving (`joblib`).*
- [ ] **MLflow Integration:** Wrap the training pipeline (`train_model.py`) with MLflow to automatically track hyperparameters, metrics (AUC-ROC, Accuracy), and execution times across different experiments.
- [ ] **Model Registry:** Implement automated version control for the XGBoost model artifacts, allowing for easy rollbacks if model performance degrades on live unseen data.
- [ ] **Automated Retraining:** Set up a scheduled trigger to incrementally retrain the model when a certain threshold of new data is accumulated.

### Phase 3: Cloud Architecture & Deployment
*Current State: Local execution (`localhost:8501`).*
- [ ] **Containerization:** Package the Streamlit application, ML model, and dependencies into an isolated environment using **Docker** to ensure consistency across different platforms.
- [ ] **Cloud Deployment (Azure/AWS):** Deploy the Docker container to a cloud platform (e.g., Azure Container Instances or Streamlit Community Cloud) to make the dashboard publicly accessible 24/7.
- [ ] **Pipeline Orchestration:** Use **GitHub Actions** or **Apache Airflow** to orchestrate the daily ETL pipeline, ensuring the dashboard always displays the latest market prediction at 00:00 UTC without manual intervention.

---
*This repository is actively maintained. Feedback, issues, and pull requests regarding the roadmap above are highly welcome!*
