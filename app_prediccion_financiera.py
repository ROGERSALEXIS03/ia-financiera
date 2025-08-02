import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import ta

st.title("📈 IA para Predicción de Activos Financieros")

# Diccionario con nombre visible y símbolo real de cada activo
opciones = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "S&P 500 (SPY)": "SPY",
    "Oro (Gold)": "GC=F",
    "Apple Inc. (AAPL)": "AAPL",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Meta (META)": "META"
}

# Menú desplegable que muestra el nombre del activo
nombre_activo = st.selectbox("Selecciona el activo:", list(opciones.keys()))

# Obtiene el símbolo real para descargar los datos
activo = opciones[nombre_activo]

# Selección de horizonte de predicción
horizonte = st.radio("Horizonte de predicción:", ["1 Día", "1 Semana"])

if st.button("Ejecutar modelo:"):
    df = yf.download(activo, start="2020-01-01")

    if 'Close' not in df.columns:
        st.error("⚠️ Error al descargar los datos. Verifica el nombre del activo.")
    else:
        df['Return'] = df['Close'].pct_change()

        if horizonte == "1 Día":
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        else:
            df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)

        df['SMA'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['Momentum'] = ta.momentum.roc(df['Close'], window=3)
        df['Volatility'] = ta.volatility.bollinger_hband_width(df['Close'], window=5)

        df = df.dropna()

        X = df[['SMA', 'Momentum', 'Volatility']]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        df['Prediction'] = model.predict(X)
        df['Strategy'] = df['Prediction'].shift() * df['Return']
        df['Cumulative Strategy'] = (1 + df['Strategy']).cumprod()
        df['Cumulative Buy & Hold'] = (1 + df['Return']).cumprod()

        st.subheader("📊 Rendimiento Estrategia vs Buy & Hold")
        st.line_chart(df[['Cumulative Strategy', 'Cumulative Buy & Hold']])

        st.subheader("📑 Reporte de clasificación")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
