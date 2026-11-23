"""
EXPERIMENT: Bitcoin vs Historical Depth
=======================================
Question: Is dropping six years of macro history (2008-2014) worth it 
          just to keep Bitcoin as a feature?

Scenario A: Current model (2014-2024, WITH Bitcoin) — 10 years
Scenario B: Full history  (2008-2025, WITHOUT Bitcoin) — 17 years

We compare: Accuracy, AUC-ROC, F1-Score on each scenario's last-15% hold-out (same end date, different start dates).
"""

import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

warnings.filterwarnings('ignore')

def run_experiment(filepath, scenario_name, drop_bitcoin=False, start_year=None):
    """Run the full training pipeline with configurable data scope."""
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'='*60}")

    df = pd.read_csv(filepath, thousands=',')
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # Apply start year filter
    if start_year:
        df = df[df.index >= f'{start_year}-01-01'].copy()
        print(f"  Data range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    
    # Define columns
    price_cols = ['Gold', 'USD_Index', 'Oil', 'Silver', 'SP500', 'Bitcoin', 'KRW', 'KRX']
    macro_cols = ['Interest_Rate', '10Y_Treasury_Yield', 'Inflation_CPI', 'Unemployment']

    # Drop Bitcoin if requested
    if drop_bitcoin:
        price_cols = [c for c in price_cols if c != 'Bitcoin']
        df = df.drop(columns=['Bitcoin'], errors='ignore')
        print("  Bitcoin: DROPPED")
    else:
        print("  Bitcoin: INCLUDED")

    # Point-in-Time shift for CPI & Unemployment
    df['Inflation_CPI'] = df['Inflation_CPI'].shift(30)
    df['Unemployment'] = df['Unemployment'].shift(30)

    # Feature Engineering (same as train_model.py)
    for col in price_cols:
        for window in [5, 10, 20, 60]:
            df[f'{col}_ret_{window}d'] = df[col].pct_change(window)
    
    df['KRW_vol_20d'] = df['KRW'].pct_change().rolling(20).std()
    df['USD_regime'] = (df['USD_Index'] > df['USD_Index'].rolling(200).mean()).astype(int)
    
    for col in macro_cols:
        roll = df[col].rolling(252)
        df[f'{col}_zscore'] = (df[col] - roll.mean()) / (roll.std() + 1e-8)
    
    df['Gold_Silver_ratio'] = df['Gold'] / df['Silver']
    df['KRX_SP500_spread'] = df['KRX'].pct_change(20) - df['SP500'].pct_change(20)

    # Target (rows without a 63-day future are left as NaN and dropped below)
    future_krw = df['KRW'].shift(-63)
    df['Target'] = (future_krw > df['KRW']).astype(float)
    df.loc[future_krw.isna(), 'Target'] = np.nan

    # Clean inf values (can appear when pct_change divides by 0, e.g. Bitcoin=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_clean = df.dropna().copy()
    df_clean['Target'] = df_clean['Target'].astype(int)
    features = [c for c in df_clean.columns if c not in set(price_cols + macro_cols + ['Target'])]

    X = df_clean[features]
    y = df_clean['Target']

    print(f"  Total samples: {len(X)}")
    print(f"  Total features: {len(features)}")
    print(f"  Class balance: {y.mean()*100:.1f}% weaken / {(1-y.mean())*100:.1f}% strengthen")

    # Train/Test Split (same as main pipeline)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    gap = 63
    X_train = X_train_full[:-gap]
    y_train = y_train_full[:-gap]

    spw = (len(y_train) - y_train.sum()) / y_train.sum()

    model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        gamma=1, reg_lambda=2, scale_pos_weight=spw,
        random_state=42, eval_metric='logloss',
        early_stopping_rounds=50
    )

    # Validation window = last 10% of train, with the same 63-day purge gap (as in train_model.py)
    n_val = int(np.ceil(len(X_train) * 0.1))
    X_val, y_val = X_train.iloc[-n_val:], y_train.iloc[-n_val:]
    X_tr, y_tr = X_train.iloc[:-(n_val + gap)], y_train.iloc[:-(n_val + gap)]
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

    val_probs = model.predict_proba(X_val)[:, 1]
    probs = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold on the VALIDATION window (never on test)
    best_threshold = 0.5
    best_accuracy = 0.0
    for t in np.arange(0.30, 0.75, 0.05):
        preds = (val_probs >= t).astype(int)
        acc = accuracy_score(y_val, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = round(t, 2)

    preds = (probs >= best_threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds, average='weighted')
    best_constant = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  RESULTS: {scenario_name:<26}│")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  Accuracy     : {acc*100:>6.2f}%             │")
    print(f"  │  AUC-ROC      : {auc:>6.4f}              │")
    print(f"  │  F1-Weighted  : {f1:>6.4f}              │")
    print(f"  │  Threshold    : {best_threshold:>6.2f}              │")
    print(f"  │  Train samples: {len(X_train):>6}              │")
    print(f"  │  Test samples : {len(X_test):>6}              │")
    print(f"  └─────────────────────────────────────┘")

    return {
        'Scenario': scenario_name,
        'Accuracy': round(acc, 4),
        'AUC-ROC': round(auc, 4),
        'F1-Weighted': round(f1, 4),
        'Best Constant Acc': round(best_constant, 4),
        'Threshold': best_threshold,
        'Data Range': f'{start_year or 2008}-2025',
        'Samples': len(X),
        'Features': len(features),
        'Has Bitcoin': not drop_bitcoin
    }


if __name__ == "__main__":
    filepath = 'data/XAU BTC Silver SP500 dataset.csv'

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   EXPERIMENT: Bitcoin Trade-off Analysis                ║")
    print("║   Does dropping 14 years of history for Bitcoin         ║")
    print("║   actually improve the model?                           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Scenario A: Current approach (2014+, WITH Bitcoin)
    result_a = run_experiment(filepath, "A: 2014+ WITH Bitcoin", 
                              drop_bitcoin=False, start_year=2014)

    # Scenario B: Full history (2008+, WITHOUT Bitcoin)  
    result_b = run_experiment(filepath, "B: 2008+ NO Bitcoin", 
                              drop_bitcoin=True, start_year=None)

    # Scenario C: Hybrid — 2008+ WITH Bitcoin (NaN filled with 0 returns)
    # This tests if we can have both: full history + Bitcoin signal post-2014
    print(f"\n{'='*60}")
    print(f"  SCENARIO: C: 2008+ Bitcoin filled")
    print(f"{'='*60}")
    print("  (Bitcoin returns set to 0 before 2014 — neutral signal)")
    
    df_c = pd.read_csv(filepath, thousands=',')
    if 'Unnamed: 0' in df_c.columns:
        df_c.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df_c['Date'] = pd.to_datetime(df_c['Date'])
    df_c.set_index('Date', inplace=True)
    df_c.sort_index(inplace=True)
    # Fill Bitcoin NaN with forward-fill then 0
    df_c['Bitcoin'] = df_c['Bitcoin'].ffill().fillna(0)
    df_c.to_csv('data/_temp_filled.csv')
    result_c = run_experiment('data/_temp_filled.csv', "C: 2008+ Bitcoin filled",
                              drop_bitcoin=False, start_year=None)
    import os
    os.remove('data/_temp_filled.csv')

    # ── Final Comparison ──────────────────────────
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                    FINAL COMPARISON                            ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  {'Scenario':<26} {'Accuracy':>9} {'Constant':>9} {'AUC-ROC':>8} {'Samples':>8} ║")
    print(f"║  {'─'*26} {'─'*9} {'─'*9} {'─'*8} {'─'*8} ║")
    for r in [result_a, result_b, result_c]:
        acc_str = f"{r['Accuracy']*100:.1f}%"
        const_str = f"{r['Best Constant Acc']*100:.1f}%"
        print(f"║  {r['Scenario']:<26} {acc_str:>9} {const_str:>9} {r['AUC-ROC']:>8.4f} {r['Samples']:>8} ║")
    print("╠══════════════════════════════════════════════════════════════════╣")

    # No automatic "winner": each scenario is scored on its own last 15%,
    # so the test windows differ and one hold-out per scenario is not enough
    # to rank them. Compare each row with its own constant-guess baseline.
    beats = [r['Scenario'].split(':')[0] for r in [result_a, result_b, result_c]
             if r['Accuracy'] > r['Best Constant Acc']]
    print("║  'Constant' = accuracy of always predicting the test window's   ║")
    print("║  majority class. Test windows differ between scenarios.         ║")
    print(f"║  Beat their constant baseline: {', '.join(beats) if beats else 'none':<34}║")
    print("╚══════════════════════════════════════════════════════════════════╝")
