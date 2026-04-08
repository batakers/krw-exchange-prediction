import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_and_preprocess_data(filepath):
    print("Loading data...")
    # Load dataset
    df = pd.read_csv(filepath)
    # Rename unnamed date column
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Handle Bitcoin missing values by filling with 0 (since it didn't exist/trade actively initially)
    if 'Bitcoin' in df.columns:
        df['Bitcoin'] = df['Bitcoin'].fillna(0)
    
    print("Calculating percentage returns to make features stationary...")
    cols_to_use = ['Gold', 'USD_Index', 'Oil', 'Silver', 'SP500', 'Bitcoin', 
                   'Interest_Rate', '10Y_Treasury_Yield', 'Inflation_CPI', 
                   'Unemployment', 'KRX', 'KRW']
    
    df = df[cols_to_use].copy()
    
    # Calculate daily percentage change
    df_pct = df.pct_change()
    # Replace infinite values with NaN since some initial values might be 0 (e.g. Bitcoin)
    df_pct.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create target variable: Will KRW weaken or strengthen in 126 trading days (~6 months)?
    shift_days = 126
    df_pct['Future_KRW_Price'] = df['KRW'].shift(-shift_days)
    df_pct['Current_KRW_Price'] = df['KRW']
    
    # Target label: 1 if KRW weakened (future KRW per USD > current), 0 if strengthened
    df_pct['Target'] = (df_pct['Future_KRW_Price'] > df_pct['Current_KRW_Price']).astype(int)
    
    # Drop rows where target is NaN (the last 126 days) & the first row (NaN from pct_change)
    df_clean = df_pct.dropna().copy()
    
    features = cols_to_use
    target = 'Target'
    
    return df_clean, features, target

def train_model(X, y):
    print("Beginning Training Pipeline with TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }
    
    grid = GridSearchCV(rf, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    
    print("\nTraining Complete.")
    print("Best parameters found: ", grid.best_params_)
    print("Best cross-validation accuracy: {:.4f}".format(grid.best_score_))
    
    preds = grid.predict(X)
    print("\nClassification Report on Full Dataset:\n", classification_report(y, preds))
    
    return grid.best_estimator_

if __name__ == "__main__":
    filepath = 'XAU BTC Silver SP500 dataset.csv'
    if not os.path.exists(filepath):
        print(f"Error: Dataset {filepath} not found in the current directory.")
    else:
        df, features, target = load_and_preprocess_data(filepath)
        
        X = df[features]
        y = df[target]
        
        best_model = train_model(X, y)
        
        # Save model and feature names for future inference
        model_data = {
            'model': best_model,
            'features': features
        }
        joblib.dump(model_data, 'best_model.pkl')
        print("\nSuccess: Model trained and saved to 'best_model.pkl'. Ready for Streamlit deployment!")
