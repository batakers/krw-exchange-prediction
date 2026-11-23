# 💱 Predicting KRW Movements: Will the Korean Won Weaken or Strengthen in 3 Months?

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://krw-exchange-prediction-ftadvxntbytubfqgjghx9z.streamlit.app/)

## Project Overview
This project predicts whether the Korean Won (KRW) exchange rate will weaken or strengthen over the next ~63 trading days (~3 months) using daily global asset prices and macro-economic indicators.

- **Target variable**: `1` if USD/KRW increases (KRW weakens), `0` otherwise.
- **Goal**: Provide actionable insights for travel and forex decisions.

## Model Performance (v3.2  XGBoost, corrected evaluation on real market data)

> **Update (Nov 2026):** Two audits changed this project's results.
> 1. **Evaluation bugs** (fixed in v3.1): the last 63 rows of the hold-out set had no real 3-month outcome but were labelled "strengthen", and the decision threshold was tuned on the test set. See [Evaluation Fixes](#evaluation-fixes-nov-2026).
> 2. **Wrong data** (fixed in v3.2): the `KRW` and `KRX` columns did not contain the real USD/KRW rate or KOSPI index. They are now replaced with FRED `DEXKOUS` and Yahoo Finance `^KS11`. See [Data Fix](#data-fix-nov-2026).
>
> **On the real data, the model shows no predictive skill.** The earlier 80.5% (and the 78.2% after the evaluation fix) were artifacts of the wrong KRW series.

| Metric | Score |
|---|---|
| **Test Accuracy** (threshold chosen on validation) | **41.3%** |
| **Best constant guess** (always "weaken", known only in hindsight) | 64.0% |
| **Training-majority guess** (always "weaken", no hindsight) | 64.0% |
| **Test AUC-ROC** | **0.246** |
| **Weighted F1-Score** | 0.41 |
| **Threshold** (chosen on the validation window) | 0.45 |
| **Test period** | 2023-06-08 to 2024-10-11 (344 trading days, 64% "KRW weakens") |

> The model was evaluated on a **15% hold-out test set** (unseen future data) with `shuffle=False` and a 63-day purge gap. It scores 22.7 points below simply predicting "weaken" every day. An AUC-ROC of 0.25 means its ranking is *inverted* in this period: 22 of the 40 features relate to the target in the opposite direction in 2023–2024 compared with 2015–2022 (for example, 60-day Bitcoin returns have AUC 0.37 on the training period and 0.74 on the test period). Relationships learned in one regime did not carry over to the next. A **Lookahead Bias Fix** shifts government-released macro data (CPI, Unemployment) by 30 trading days to reflect real-world publication lag.

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
- **Threshold Selection on Validation**: The probability cutoff is chosen on a purged validation window before the test period (0.45 in the current run), never on the test set.
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
| 1 | 2015-10 to 2017-01 | 2017-04 to 2018-10 | 46.5% | 48.6% | 51.4% | 0.357 | 0.333 |
| 2 | 2015-10 to 2018-07 | 2018-10 to 2020-04 | 47.8% | 32.5% | 67.5% | 0.409 | 0.455 |
| 3 | 2015-10 to 2020-01 | 2020-04 to 2021-10 | 36.5% | 54.1% | 54.1% | 0.421 | 0.361 |
| 4 | 2015-10 to 2021-07 | 2021-10 to 2023-04 | **69.3%** | 75.1% | 75.1% | **0.746** | **0.711** |
| 5 | 2015-10 to 2023-01 | 2023-04 to 2024-10 | 44.1% | 60.4% | 60.4% | 0.403 | 0.444 |
| **Mean** | | | **48.8%** | 54.1% | 61.7% | **0.467** | **0.461** |
| **Std** | | | ±12.3% | | | ±0.158 | ±0.149 |

*Std is the sample standard deviation across the five folds, as shown in the dashboard. `train_model.py` prints the population version (±11.0%, ±0.141, ±0.134).*

### Honest Interpretation

- **AUC-ROC is below 0.50 in four of five folds** (0.36–0.42), and only fold 4 (0.75) is clearly above chance. The mean of 0.47 is no better than random.
- **Accuracy beats the training-majority guess in only 1 of 5 folds** (fold 2) and never beats the best constant guess. Repeating the most common recent direction would have done better than the model.
- **The good fold does not repeat.** Fold 4 (late 2021 to early 2023, a strong-dollar period) looks promising on its own, but the next fold falls back to 0.40 AUC.
- The decision threshold (0.45) comes from the main model's validation window and is reused in every fold rather than re-tuned per fold.

> **Takeaway:** With real USD/KRW data, this feature set and model do not predict the 3-month direction of the won. The project is most useful as a case study in evaluation and data validation: every earlier "signal" disappeared once the labels, the threshold selection and the data source were fixed.

## Evaluation Fixes (Nov 2026)

An audit of the evaluation found two bugs that inflated the published 80.5% accuracy. Both are fixed in `train_model.py`, `experiment_bitcoin.py` and the dashboard backtest in `app.py`.

1. **Fake labels at the end of the data.** The target compares today's KRW with the value 63 rows later. For the last 63 rows that future value is `NaN`, and `NaN > x` evaluates to `False`, so those rows were labelled 0 ("strengthen") and survived `dropna()`. They made up 63 of the 354 test rows (18%). They are now set to `NaN` and dropped.
2. **Threshold tuned on the test set.** `find_optimal_threshold` was called with the test labels. It now runs on a validation window: the last 10% of the training data, with a 63-day purge gap before it (the same gap now also separates the training rows from that window).
3. **Baselines.** The hold-out and every walk-forward fold now report a training-majority baseline and the best constant guess, and the model file stores these metrics.
4. **Dashboard.** Model Insights reads the hold-out metrics from `best_model_v3.pkl` instead of hardcoded numbers, and the January 2025 case study computes the model's call and scores it on the 63-day horizon.
5. **Compatibility.** `experiment_bitcoin.py` uses `.ffill()` instead of `fillna(method='ffill')`, which pandas 3 removed.
6. **No automatic winner in the Bitcoin experiment.** `experiment_bitcoin.py` used to print a fixed conclusion for whichever scenario had the highest AUC. It now prints each scenario's constant-guess baseline instead, because the scenarios are scored on different test windows.

Results were produced with pandas 3.1.1, scikit-learn 1.9.1 and xgboost 3.4.1. On the original dataset, the old pipeline gives 81.9% accuracy and 0.775 AUC-ROC in this environment (vs the published 80.5% and 0.746), and the fixed pipeline gives 78.2% and 0.711 (v3.1). Those v3.1 numbers were then superseded by the data fix below.

## Data Fix (Nov 2026)

The `KRW` and `KRX` columns in `data/XAU BTC Silver SP500 dataset.csv` did not match the series they are named after:

| Date | Old `KRW` | Market USD/KRW (FRED `DEXKOUS`) | Old `KRX` | KOSPI (Yahoo `^KS11`) |
|---|---|---|---|---|
| 2008-01-02 | 937.12 | 936.60 | 1,853.45 | 1,853.45 |
| 2021-01-04 | 1,111.12 | 1,081.62 | 3,263.88 | 2,944.45 |
| 2024-12-30 | 1,124.46 | 1,472.32 | 3,177.28 | 2,399.49 |
| 2025-01-09 | 1,126.79 | 1,459.23 | 3,196.32 | 2,521.90 |

From 2022 onward, no day of the old `KRW` column was within 1% of the market rate, and the gap is not a simple date shift. Because `KRW` is both the target and the source of several features, every earlier result measured a different series.

**What changed:** both columns were replaced; all other columns are unchanged.

- `KRW` = FRED [`DEXKOUS`](https://fred.stlouisfed.org/series/DEXKOUS) (South Korean won to U.S. dollar spot rate, noon buying rate in New York).
- `KRX` = Yahoo Finance `^KS11` (KOSPI Composite Index) daily close.
- For each date in the dataset, the value is the latest observation on or before that date (113 `KRW` and 284 `KRX` rows were filled from the previous trading day because of U.S. or Korean holidays).

The raw downloads were checked with per-year checksums before being written into the CSV.

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
| v3.1 (evaluation fix) | XGBoost | 3 months (63 days) | 78.2%‡ | 0.711‡ | 0.76‡ |
| **v3.2 (real USD/KRW and KOSPI data)** | **XGBoost** | **3 months (63 days)** | **41.3%** | **0.246** | **0.41** |

*† Measured with 63 mislabelled test rows and a threshold tuned on the test set. v1–v3 figures are kept as a record of the project's history and were all measured under the old evaluation and the original dataset. See [Evaluation Fixes (Nov 2026)](#evaluation-fixes-nov-2026).*

*‡ Measured on the original dataset, whose `KRW` and `KRX` columns were wrong (see [Data Fix](#data-fix-nov-2026)).*

---

### ⚠️ Model Limitations & Market Realities

Beyond the lack of predictive skill shown in the results above, real-world forex markets are influenced by factors beyond pure data. We acknowledge the following operational blindspots:

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

### v3: The Final Optimization (80.5%, later corrected; see v3.1 and v3.2)

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

> **Correction (Nov 2026):** this threshold search was run on the test set, so the 80.5% was optimistic. The threshold is now chosen on a validation window before the test period. With the label fix, the result on the original dataset was 78.2% accuracy and AUC-ROC 0.711 (v3.1). After replacing the wrong KRW and KRX data, it is 41.3% accuracy and AUC-ROC 0.246 (v3.2).

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
*† Superseded. v3.1 (threshold chosen on validation, mislabelled rows removed): 78.2% accuracy and 0.711 AUC-ROC on the original dataset. v3.2 (real USD/KRW and KOSPI data): 41.3% accuracy and 0.246 AUC-ROC.*

### Key Takeaways

1. **A powerful algorithm alone is not enough.** Upgrading from Random Forest to XGBoost without fixing the underlying data issues made the model *worse* (64% to 46%). The algorithm amplified the noise instead of learning from it.

2. **Feature engineering is the most impactful lever.** The jump from 46% to 68.4% came entirely from redesigning *what* the model sees: multi-horizon returns, regime detection, and macro z-scores. The algorithm stayed the same.

3. **The prediction horizon must match the feature granularity.** Predicting 6 months ahead with daily price changes is like trying to forecast the weather next season by looking out the window today. Shortening to 3 months and using 5-60 day rolling returns created a much better signal-to-noise ratio.

4. **Where you tune the threshold matters as much as the threshold.** Tuning it on the test set lifted accuracy to 80.5%. Choosing it on a validation window gave 78.2% on the original dataset, only about 5.5 points above a constant "strengthen" guess.

5. **Fixing lookahead bias costs accuracy and that's a good thing.** Applying a 30-day shift to CPI and Unemployment reduced accuracy from 83.2% to 80.5% (both measured under the old evaluation). The drop shows the original model was partially "cheating" by using government data before it was actually published.

6. **Proper evaluation prevents false confidence.** Without a chronological hold-out test set and temporal gap, financial models will always appear more accurate than they truly are.

7. **Check the labels at the edges.** With a 63-day forward target, the last 63 rows have no outcome yet. `NaN > x` evaluates to `False` in pandas, so they were silently labelled "strengthen" and kept, which made 18% of the original test labels fake. Always drop rows whose future value is missing before building a forward-looking target.

8. **Validate the data source before modelling.** The `KRW` column did not contain the USD/KRW rate. A few spot checks against an official series (here FRED `DEXKOUS`) would have caught it, and every result built on it, including the apparent signal, disappeared once the real series was used.

## Real-World Validation

> *"Trust is earned, not claimed."* One live prediction cannot prove or disprove a model, but it should be scored honestly on the model's own horizon.

### Case Study: January 9, 2025 Prediction

| | Details |
|---|---|
| **Prediction Date** | January 9, 2025 (last row of the dataset) |
| **Model Says** | KRW will **Strengthen** (↓) over the next 63 trading days |
| **Probability of Weakening** | 40.9% (v3.2 model, threshold 0.45). Earlier models gave 31.5% (v3) and 18.9% (v3.1). |

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

| Scenario | Data Range | Bitcoin | Samples | Features | Test window |
|---|---|---|---|---|---|
| **A**: Current Approach | 2014–2025 | ✅ Included | 2,474 | 40 | 2023-04-28 to 2024-10-11 (372 days) |
| **B**: Full History | 2008–2025 | ❌ Dropped | 4,009 | 36 | 2022-06-02 to 2024-10-11 (602 days) |
| **C**: Hybrid | 2008–2025 | ✅ Zero-filled pre-2014 | 2,512 | 40 | 2023-04-21 to 2024-10-11 (377 days) |

Scenario C keeps only 38 more rows than A: a zero-filled price makes the Bitcoin return features undefined before the Bitcoin series starts (September 2014), so those rows are dropped. In practice C is almost the same experiment as A.

### Results (real USD/KRW and KOSPI data, Nov 2026)

| Scenario | Test Accuracy | Best constant guess | AUC-ROC | F1-Weighted | Threshold (validation) |
|---|---|---|---|---|---|
| A: 2014+ WITH Bitcoin | 61.8% | 61.8% | 0.518 | 0.581 | 0.30 |
| B: 2008+ NO Bitcoin | 63.3% | 59.3% | 0.661 | 0.633 | 0.30 |
| C: 2008+ Bitcoin filled | 63.7% | 61.0% | 0.496 | 0.598 | 0.30 |

*Earlier versions of this table were measured on the wrong KRW series (and, before that, with mislabelled test rows and thresholds tuned on the test set). Those versions reported Scenario A as the clear winner; that result does not hold on real data.*

### Analysis

1. **No scenario shows a reliable signal.** Every threshold landed at 0.30, the bottom of the search grid, so A and C predict "weaken" on about 80% of test days (B on 59%). Scenario A's accuracy equals its constant guess, and its AUC (0.52) is at chance. Scenario C's AUC (0.50) is also at chance; its 2.7-point lead over the constant guess comes from the low threshold, not from ranking days well.

2. **Scenario B ranks days best, but mostly because of its longer test window.** B's test window starts in June 2022; on the June 2022 to April 2023 part of it (the strong-dollar period), its AUC is 0.79. On the same window as A (2023-04-28 onward), B's AUC drops to 0.60 and its accuracy (56.5%) is below the constant guess (61.8%).

3. **The original claim is not supported.** Earlier write-ups said removing Bitcoin "collapses" performance. On real data, removing Bitcoin did not hurt. On the shared window (2023-04-28 to 2024-10-11), A ties the constant guess (61.8%), B falls below it (56.5%), and C edges past it (63.4%) with an AUC below chance (0.48), which points to the low threshold rather than skill.

*These are single hold-out results with different test windows, so differences of a few points or a few hundredths of AUC are within noise.*

### Conclusion

On real USD/KRW data, this experiment does not show that Bitcoin is an essential feature, and it does not show that the 2008–2014 history helps either. The earlier conclusion (a "market regime shift" that made Bitcoin indispensable) came from the wrong KRW series. The decision to start in 2014 should be treated as a modelling choice, not an empirically supported one, until a feature set shows skill against constant baselines on a shared walk-forward window.

*The full experiment code is available in [`experiment_bitcoin.py`](experiment_bitcoin.py).*

---

## 🚀 Future Roadmap: From Script to Production-Grade System
> **Status:** *Work in Progress (On-going)*

The current model does not yet show predictive skill on real data, and the architecture is a static, local script. The first priority is a feature set that beats the constant-guess baselines in walk-forward validation; only then is it worth transforming this prototype into a fully automated, production-grade **End-to-End MLOps Pipeline**.

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
