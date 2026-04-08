import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="KRW Movement Prediction", layout="wide", page_icon="💱")

st.title("💱 KRW Movement Prediction (Korean Won)")
st.write("Will the Korean Won weaken or strengthen over the next 6 months? Predict future movements based on daily returns of global commodities and assets.")

tab1, tab2 = st.tabs(["📈 Prediction Dashboard", "⚙️ Model Insights (Under the Hood)"])

MODEL_PATH = 'best_model.pkl'
DATA_PATH = 'XAU BTC Silver SP500 dataset.csv'

with tab1:
    st.markdown("---")
    st.subheader("Historical Context: USD/KRW Exchange Rate & Macro Events")

    if os.path.exists(DATA_PATH):
        try:
            df_hist = pd.read_csv(DATA_PATH, usecols=['Unnamed: 0', 'KRW'])
            df_hist.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            df_hist['Date'] = pd.to_datetime(df_hist['Date'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['KRW'], mode='lines', name='USD/KRW', line=dict(color='#00ff9d', width=2)))
            
            # Annotations
            fig.add_annotation(x='2008-11-01', y=1350, text="2008 Financial Crisis", showarrow=True, arrowhead=2, ax=-20, ay=-50, arrowcolor="red")
            fig.add_annotation(x='2020-03-20', y=1285, text="COVID-19 Pandemic Shock", showarrow=True, arrowhead=2, ax=-30, ay=-50, arrowcolor="red")
            fig.add_annotation(x='2022-10-01', y=1430, text="Aggressive Fed Rate Hikes", showarrow=True, arrowhead=2, ax=-40, ay=-40, arrowcolor="red")
            
            fig.update_layout(xaxis_title="Time", yaxis_title="KRW per 1 USD",
                              template="plotly_dark", height=450, margin=dict(l=0, r=0, t=30, b=0))
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Silently skipped historical chart loading: {e}")

    st.markdown("---")

    if not os.path.exists(MODEL_PATH):
        st.warning("AI Model not found. Please train the model by running `python train_model.py` in your terminal first.")
    else:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        features = model_data['features']
        
        st.sidebar.header("🕹️ What-If Analysis")
        st.sidebar.write("Slide these bars to simulate the daily percentage return dynamics for each macro indicator.")
        
        input_data = {}
        for i, feature in enumerate(features):
            val = st.sidebar.slider(f"{feature} Return (%)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            input_data[feature] = [val / 100.0]
            
        input_df = pd.DataFrame(input_data)
        
        st.subheader("Simulated Economic Conditions (*Daily Returns*)")
        st.dataframe(input_df.style.format("{:.2%}"))
        
        if st.button("Run Future Prediction Engine", use_container_width=True):
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0]
            
            st.markdown("---")
            
            # --- Speedometer/Gauge Chart ---
            prob_weaken = probability[1] * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_weaken,
                title={'text': "Probability of KRW Weakening (%)", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255, 99, 71, 0.4)"},  # Red
                        {'range': [50, 100], 'color': "rgba(0, 255, 128, 0.3)"} # Green
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prob_weaken
                    }
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), template="plotly_dark")
            st.plotly_chart(fig_gauge, use_container_width=True)
            # -------------------------------
            
            if prediction[0] == 1:
                st.success(f"### 📈 Prediction: KRW Weakens (Increases against USD)")
                st.info("💡 Travel Advice: If you are planning a trip to Korea and hold USD cash, it is highly likely that the Won will become cheaper to buy 6 months from now. It might be a good idea to hold your dollars and exchange them later (Great News!).")
            else:
                st.error(f"### 📉 Prediction: KRW Strengthens (Drops against USD)")
                st.info("💡 Travel Advice: If you are planning a trip using USD, this movement is threatening. The Won is predicted to strengthen, meaning everything will feel more expensive. We advise you to **exchange your cash now before the rate rockets in the coming months.**")

with tab2:
    st.header("⚙️ Under the Hood: Model Insights")
    st.markdown("For technical recruiters: This section details the analytical behavior and performance of the deployed Machine Learning model.")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        # 1. Feature Importance
        st.subheader("1. Feature Importance (Random Forest)")
        model_data = joblib.load(MODEL_PATH)
        rf_model = model_data['model']
        feature_names = model_data['features']
        
        # Check if model has feature_importances_ mapped
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)
            sorted_features = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]
            
            fig_importance = px.bar(
                x=sorted_importances, 
                y=sorted_features, 
                orientation='h',
                title='Which Macro Indicators Drive KRW Predictions Most?',
                labels={'x':'Relative Importance', 'y':''},
                color=sorted_importances,
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(template="plotly_dark")
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Loaded model does not maintain feature importance metrics.")
        
        st.markdown("---")
        
        # 2. Correlation Heatmap
        st.subheader("2. Market Correlation Heatmap")
        st.markdown("Displays the historical correlation of daily percentage returns between global assets.")
        
        # Cache this expensive correlation function 
        @st.cache_data
        def get_correlation_matrix(data_path, cols):
            # Membaca data dengan mempertimbangkan separator ribuan (comma)
            df_full = pd.read_csv(data_path, thousands=',')
            
            # Mengambil list kolom (menggunakan set agar tidak ada duplikasi 'KRW')
            numeric_cols = list(set(list(cols) + ['KRW']))
            df_numeric = df_full[numeric_cols].copy()
            
            # Force convert object (string) to float jika masih ada sisa teks dengan koma
            for c in df_numeric.columns:
                if df_numeric[c].dtype == object:
                    df_numeric[c] = df_numeric[c].astype(str).str.replace(',', '').astype(float)
            
            if 'Bitcoin' in df_numeric.columns:
                df_numeric['Bitcoin'] = df_numeric['Bitcoin'].fillna(0)
                
            df_returns = df_numeric.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            return df_returns.corr()
            
        try:
            corr_matrix = get_correlation_matrix(DATA_PATH, tuple(feature_names))
            
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=".2f", 
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Daily Return Correlation Matrix"
            )
            fig_corr.update_layout(template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        except Exception as e:
            st.error(f"Could not generate heatmap: {str(e)}")
            
    else:
        st.warning("Data or model file is missing, cannot generate insights.")

st.markdown("---")
st.markdown("*KRW Movement Simulator for Travel Decisions. Powered by AI Machine Learning *(Random Forest)*.*")
