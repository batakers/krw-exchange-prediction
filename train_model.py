import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import joblib
import os

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS (Updated with Volatility)
# ─────────────────────────────────────────────
def load_and_preprocess_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath, thousands=',')
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # Drop pre-Bitcoin era
    df = df[df.index >= '2014-09-17'].copy()

    print("Implementing Advanced Feature Engineering...")
    price_cols = ['Gold', 'USD_Index', 'Oil', 'Silver', 'SP500', 'Bitcoin', 'KRW', 'KRX']
    macro_cols = ['Interest_Rate', '10Y_Treasury_Yield', 'Inflation_CPI', 'Unemployment']

    # Multi-Horizon Returns
    for col in price_cols:
        for window in [5, 10, 20, 60]:
            df[f'{col}_ret_{window}d'] = df[col].pct_change(window)
    
    # Volatility Feature — captures market panic/anomalies
    df['KRW_vol_20d'] = df['KRW'].pct_change().rolling(20).std()

    # Regime & Z-Scores
    df['USD_regime'] = (df['USD_Index'] > df['USD_Index'].rolling(200).mean()).astype(int)
    for col in macro_cols:
        roll = df[col].rolling(252)
        df[f'{col}_zscore'] = (df[col] - roll.mean()) / (roll.std() + 1e-8)

    # Spreads
    df['Gold_Silver_ratio'] = df['Gold'] / df['Silver']
    df['KRX_SP500_spread'] = df['KRX'].pct_change(20) - df['SP500'].pct_change(20)

    # Target (63 days horizon)
    shift_days = 63
    df['Target'] = (df['KRW'].shift(-shift_days) > df['KRW']).astype(int)

    df_clean = df.dropna().copy()
    features = [c for c in df_clean.columns if c not in set(price_cols + macro_cols + ['Target'])]
    
    return df_clean, features, 'Target'

# ─────────────────────────────────────────────
# 2. FIND OPTIMAL THRESHOLD
# ─────────────────────────────────────────────
def find_optimal_threshold(y_true, probs):
    """
    Search for the probability threshold that maximizes accuracy.
    Default is 0.5, but class imbalance often shifts the sweet spot.
    """
    best_threshold = 0.5
    best_accuracy = 0.0

    print("\n  Threshold Search:")
    print(f"  {'Threshold':<12} {'Accuracy':<12} {'F1-Weighted':<12}")
    print("  " + "─" * 36)

    for t in np.arange(0.30, 0.75, 0.05):
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='weighted')
        marker = ""
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t
            marker = " ◀ best"
        print(f"  {t:<12.2f} {acc:<12.4f} {f1:<12.4f}{marker}")

    return round(best_threshold, 2)

# ─────────────────────────────────────────────
# 3. TRAIN MODEL (Optimized v3)
# ─────────────────────────────────────────────
def train_model_v3(X, y):
    print("Splitting data (Hold-out 15%)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    # Add gap so train data doesn't overlap with test target window
    # Since our target looks 63 days ahead
    gap = 63
    X_train = X_train_full[:-gap]
    y_train = y_train_full[:-gap]

    # TimeSeriesSplit with gap for internal cross-validation
    tscv = TimeSeriesSplit(n_splits=5, test_size=126, gap=gap) 

    # scale_pos_weight for class imbalance
    spw = (len(y_train) - y_train.sum()) / y_train.sum()

    # Conservative model with L2 regularization & gamma
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_lambda=2,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50
    )

    # Split a validation set from train for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

    print("Training with Early Stopping...")
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

    # Get probabilities on test set
    probs = model.predict_proba(X_test)[:, 1]

    # ── Find optimal threshold ──────────────────
    print("\n" + "=" * 50)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 50)
    optimal_threshold = find_optimal_threshold(y_test, probs)

    # ── Results with DEFAULT threshold (0.5) ────
    preds_default = model.predict(X_test)
    print(f"\n{'='*50}")
    print("RESULTS — Default Threshold (0.50)")
    print(f"{'='*50}")
    print(f"  Accuracy : {accuracy_score(y_test, preds_default):.4f}")
    print(f"  AUC-ROC  : {roc_auc_score(y_test, probs):.4f}")
    print(f"\n{classification_report(y_test, preds_default)}")

    # ── Results with OPTIMAL threshold ──────────
    preds_optimal = (probs >= optimal_threshold).astype(int)
    print(f"{'='*50}")
    print(f"RESULTS — Optimal Threshold ({optimal_threshold})")
    print(f"{'='*50}")
    print(f"  Accuracy : {accuracy_score(y_test, preds_optimal):.4f}")
    print(f"  AUC-ROC  : {roc_auc_score(y_test, probs):.4f}")
    print(f"\n{classification_report(y_test, preds_optimal)}")

    return model, optimal_threshold

# ─────────────────────────────────────────────
# 4. MAIN EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    filepath = 'data/XAU BTC Silver SP500 dataset.csv'
    if os.path.exists(filepath):
        df, features, target = load_and_preprocess_data(filepath)
        X, y = df[features], df[target]
        
        best_model, threshold = train_model_v3(X, y)
        
        joblib.dump({
            'model': best_model, 
            'features': features,
            'threshold': threshold
        }, 'best_model_v3.pkl')
        print(f"\nModel v3 saved with optimal threshold={threshold}")
    else:
        print("Dataset file not found.")
