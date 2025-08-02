import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import ta

st.set_page_config(page_title="Predicción Financiera", layout="centered")
st.title("📈 IA para Predicción de Activos Financieros")

# Diccionario de activos
activos = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "S&P 500 (SPY)": "SPY",
    "Oro (Gold)": "GC=F"
}

activo_nombre = st.selectbox("Selecciona el activo:", list(activos.keys()))
ticker = activos[activo_nombre]

horizonte = st.radio("Horizonte de predicción:", ["1 Día", "1 Semana"])

if st.button("Ejecutar modelo"):
    try:
        df = yf.download(ticker, start="2020-01-01")

        if "Close" not in df.columns or df.empty:
            st.error("❌ No se pudieron descargar los datos. Verifica el nombre del activo o tu conexión a Internet.")
        else:
            df['Return'] = df['Close'].pct_change()

            if horizonte == "1 Día":
                df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            else:
                df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)

            # Indicadores técnicos con .values.flatten() para aplanar la dimensión
            df['SMA'] = ta.trend.sma_indicator(df['Close'], window=5).values.flatten()
            df['Momentum'] = ta.momentum.roc(df['Close'], window=5).values.flatten()
            df['Volatility'] = ta.volatility.bollinger_hband_width(df['Close'], window=5).values.flatten()

            df.dropna(inplace=True)

            X = df[['SMA', 'Momentum', 'Volatility']]
            y = df['Target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            df['Prediction'] = model.predict(X)
            df['Strategy'] = df['Prediction'].shift(1) * df['Return']

            df['Cumulative Strategy'] = (1 + df['Strategy']).cumprod()
            df['Cumulative Buy & Hold'] = (1 + df['Return']).cumprod()

            st.subheader("📊 Rendimiento Estrategia vs Buy & Hold")
            st.line_chart(df[['Cumulative Strategy', 'Cumulative Buy & Hold']])

            st.subheader("📋 Reporte de clasificación")
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
