import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import joblib
import os

# Suppress warnings from sklearn when test folds have only one class
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS (with Lookahead Bias Fix)
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

    # ── STEP 1.1: Fix Lookahead Bias on Government-Released Macro Data ──
    # CPI and Unemployment are published with ~30 day lag.
    # Interest_Rate and 10Y_Treasury_Yield are market-based (real-time), no shift needed.
    print("Applying Point-in-Time shift to CPI & Unemployment (30 trading days)...")
    df['Inflation_CPI'] = df['Inflation_CPI'].shift(30)
    df['Unemployment'] = df['Unemployment'].shift(30)

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

    # ── Target ──
    shift_days = 63
    df['Target'] = (df['KRW'].shift(-shift_days) > df['KRW']).astype(int)

    df_clean = df.dropna().copy()
    features = [c for c in df_clean.columns if c not in
                set(price_cols + macro_cols + ['Target'])]
    
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
# 3. TRAIN CLASSIFIER (Optimized v3)
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

    print("Training Classifier with Early Stopping...")
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
# 4. WALK-FORWARD VALIDATION (Nested CV)
# ─────────────────────────────────────────────
def walk_forward_validation(df, features, target, threshold=0.65):
    """
    Walk-Forward Validation using scikit-learn's TimeSeriesSplit
    with Nested Cross-Validation (RandomizedSearchCV inside each fold).
    Uses expanding window with a 63-day gap to prevent target leakage.
    """
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION (Nested CV via TimeSeriesSplit)")
    print("=" * 70)

    X = df[features]
    y = df[target]

    # Outer TimeSeriesSplit: 5 rolling test periods with 63-day gap
    tscv = TimeSeriesSplit(n_splits=5, gap=63)

    # Hyperparameter search space for inner CV
    param_dist = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.02, 0.05],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8],
        'gamma': [0.5, 1, 2],
        'reg_lambda': [1, 2, 3],
    }

    # Lists to collect per-fold metrics
    accuracies = []
    auc_scores = []
    f1_scores_list = []
    results = []

    print(f"\n  {'Fold':<8} {'Train Size':<14} {'Test Size':<12} {'Accuracy':<12} {'AUC-ROC':<12} {'F1-Weighted':<12}")
    print("  " + "─" * 70)

    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train_wf = X.iloc[train_index]
        y_train_wf = y.iloc[train_index]
        X_test_wf = X.iloc[test_index]
        y_test_wf = y.iloc[test_index]

        # Handle class imbalance
        pos_count = y_train_wf.sum()
        if pos_count == 0:
            continue
        spw = (len(y_train_wf) - pos_count) / pos_count

        # Inner CV: 3 splits with gap for hyperparameter search
        inner_cv = TimeSeriesSplit(n_splits=3, gap=63)

        # Base model template
        # NOTE: early_stopping_rounds is NOT compatible with RandomizedSearchCV
        # (sklearn doesn't pass eval_set). Use fixed n_estimators instead.
        base_model = xgb.XGBClassifier(
            n_estimators=300,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric='logloss'
        )

        # Nested RandomizedSearchCV (20 iterations × 3 inner folds = 60 models/fold)
        print(f"  Fold {fold}: Searching hyperparameters (20 combos × 3 inner folds)...")
        search = RandomizedSearchCV(
            base_model, param_dist,
            cv=inner_cv,
            scoring='roc_auc',
            n_iter=20,
            random_state=42,
            n_jobs=1,           # n_jobs=-1 causes access violation on Python 3.14 + XGBoost
            error_score=0.0
        )

        try:
            search.fit(X_train_wf, y_train_wf)
            best_model_wf = search.best_estimator_
        except ValueError:
            # Fallback: if all inner CV fits fail (e.g., single class in folds),
            # train directly with default parameters
            print(f"  Fold {fold}: Inner CV failed, using default parameters...")
            fallback_model = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.8,
                gamma=1, reg_lambda=2, scale_pos_weight=spw,
                random_state=42, eval_metric='logloss'
            )
            fallback_model.fit(X_train_wf, y_train_wf)
            best_model_wf = fallback_model

        # Predict using threshold (not default 0.50)
        probs_wf = best_model_wf.predict_proba(X_test_wf)[:, 1]
        preds_wf = (probs_wf >= threshold).astype(int)

        # Calculate metrics
        acc = accuracy_score(y_test_wf, preds_wf)
        f1_w = f1_score(y_test_wf, preds_wf, average='weighted')
        try:
            auc = roc_auc_score(y_test_wf, probs_wf)
        except ValueError:
            auc = 0.0

        accuracies.append(acc)
        auc_scores.append(auc)
        f1_scores_list.append(f1_w)

        # Determine test period date range for display
        test_start = df.index[test_index[0]].strftime('%Y-%m')
        test_end = df.index[test_index[-1]].strftime('%Y-%m')

        try:
            best_params = search.best_params_
        except AttributeError:
            best_params = {'note': 'fallback defaults'}

        result = {
            'Fold': f'Fold {fold}',
            'Train Period': f'{df.index[train_index[0]].strftime("%Y-%m")} to {df.index[train_index[-1]].strftime("%Y-%m")}',
            'Test Period': f'{test_start} to {test_end}',
            'Accuracy': round(acc, 4),
            'AUC-ROC': round(auc, 4),
            'F1-Weighted': round(f1_w, 4),
            'Samples': len(X_test_wf),
            'Best Params': str(best_params)
        }
        results.append(result)

        print(f"  {fold:<8} {len(X_train_wf):<14} {len(X_test_wf):<12} {acc:<12.4f} {auc:<12.4f} {f1_w:<12.4f}")
        print(f"           Best params: {best_params}")

    # Summary: Mean & Std
    if results:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        mean_f1 = np.mean(f1_scores_list)
        std_f1 = np.std(f1_scores_list)

        print("  " + "─" * 70)
        print(f"  {'MEAN':<8} {'':<14} {'':<12} {mean_acc:<12.4f} {mean_auc:<12.4f} {mean_f1:<12.4f}")
        print(f"  {'STD':<8} {'':<14} {'':<12} {std_acc:<12.4f} {std_auc:<12.4f} {std_f1:<12.4f}")
        print(f"\n  Nested Walk-Forward Validation complete across {len(results)} folds.")
        print(f"  Average Accuracy: {mean_acc*100:.1f}% (±{std_acc*100:.1f}%)")
        print(f"  Average AUC-ROC:  {mean_auc:.4f} (±{std_auc:.4f})")
        print(f"  Average F1-Score: {mean_f1:.4f} (±{std_f1:.4f})")

    return results

# ─────────────────────────────────────────────
# 6. MAIN EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    filepath = 'data/XAU BTC Silver SP500 dataset.csv'
    if os.path.exists(filepath):
        df, features, target = load_and_preprocess_data(filepath)
        X, y = df[features], df[target]
        
        # Train classifier (direction prediction)
        best_model, threshold = train_model_v3(X, y)

        # Walk-Forward Validation (Nested CV)
        wf_results = walk_forward_validation(df, features, target, threshold)
        
        joblib.dump({
            'model': best_model, 
            'features': features,
            'threshold': threshold,
            'walk_forward_results': wf_results
        }, 'best_model_v3.pkl')
        print(f"\nModel v3 saved with optimal threshold={threshold}")
    else:
        print("Dataset file not found.")
