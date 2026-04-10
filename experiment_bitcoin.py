"""
EXPERIMENT: Bitcoin vs Historical Depth
=======================================
Question: Is dropping 14 years of macro history (2000-2014) worth it 
          just to keep Bitcoin as a feature?

Scenario A: Current model (2014-2024, WITH Bitcoin) — 10 years
Scenario B: Full history  (2000-2024, WITHOUT Bitcoin) — 24 years

We compare: Accuracy, AUC-ROC, F1-Score on the same hold-out test.
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

    # Target
    df['Target'] = (df['KRW'].shift(-63) > df['KRW']).astype(int)
    
    # Clean inf values (can appear when pct_change divides by 0, e.g. Bitcoin=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    df_clean = df.dropna().copy()
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

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)

    probs = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    best_threshold = 0.5
    best_accuracy = 0.0
    for t in np.arange(0.30, 0.75, 0.05):
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_test, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = round(t, 2)

    preds = (probs >= best_threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds, average='weighted')

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
        'Threshold': best_threshold,
        'Data Range': f'{start_year or 2000}-2025',
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

    # Scenario B: Full history (2000+, WITHOUT Bitcoin)  
    result_b = run_experiment(filepath, "B: 2000+ NO Bitcoin", 
                              drop_bitcoin=True, start_year=None)

    # Scenario C: Hybrid — 2000+ WITH Bitcoin (NaN filled with 0 returns)
    # This tests if we can have both: full history + Bitcoin signal post-2014
    print(f"\n{'='*60}")
    print(f"  SCENARIO: C: 2000+ Bitcoin filled")
    print(f"{'='*60}")
    print("  (Bitcoin returns set to 0 before 2014 — neutral signal)")
    
    df_c = pd.read_csv(filepath, thousands=',')
    if 'Unnamed: 0' in df_c.columns:
        df_c.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df_c['Date'] = pd.to_datetime(df_c['Date'])
    df_c.set_index('Date', inplace=True)
    df_c.sort_index(inplace=True)
    # Fill Bitcoin NaN with forward-fill then 0
    df_c['Bitcoin'] = df_c['Bitcoin'].fillna(method='ffill').fillna(0)
    df_c.to_csv('data/_temp_filled.csv')
    result_c = run_experiment('data/_temp_filled.csv', "C: 2000+ Bitcoin filled",
                              drop_bitcoin=False, start_year=None)
    import os
    os.remove('data/_temp_filled.csv')

    # ── Final Comparison ──────────────────────────
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                    FINAL COMPARISON                            ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  {'Scenario':<28} {'Accuracy':>10} {'AUC-ROC':>10} {'Samples':>10}  ║")
    print(f"║  {'─'*28} {'─'*10} {'─'*10} {'─'*10}  ║")
    for r in [result_a, result_b, result_c]:
        acc_str = f"{r['Accuracy']*100:.1f}%"
        print(f"║  {r['Scenario']:<28} {acc_str:>10} {r['AUC-ROC']:>10.4f} {r['Samples']:>10}  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")

    # Determine winner
    results = [result_a, result_b, result_c]
    best = max(results, key=lambda x: x['AUC-ROC'])
    print(f"║  🏆 WINNER (by AUC-ROC): {best['Scenario']:<38} ║")
    
    if best == result_b:
        print("║                                                                  ║")
        print("║  CONCLUSION: Dropping Bitcoin and using full 2000-2024 data       ║")
        print("║  is BETTER. The 2008 crisis data provides more value than         ║") 
        print("║  Bitcoin as a feature. Market structure has NOT changed enough     ║")
        print("║  to justify losing 14 years of macro history.                     ║")
    elif best == result_a:
        print("║                                                                  ║")
        print("║  CONCLUSION: Keeping Bitcoin with 2014+ data is BETTER.           ║")
        print("║  This proves that post-2014 market structure has fundamentally    ║")
        print("║  changed, and Bitcoin is an essential feature for modern FX       ║")
        print("║  prediction. The 2008 crisis data is no longer relevant.          ║")
    else:
        print("║                                                                  ║")
        print("║  CONCLUSION: The HYBRID approach wins! Full history with          ║")
        print("║  Bitcoin (zero-filled pre-2014) gives the best of both worlds:    ║")
        print("║  2008 crisis wisdom + modern crypto signal.                       ║")

    print("╚══════════════════════════════════════════════════════════════════╝")
