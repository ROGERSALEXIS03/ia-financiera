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

# Diccionario de activos con sus respectivos TICKERS de Yahoo Finance
activo = st.selectbox("Selecciona el activo:", {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "S&P 500 (SPY)": "SPY",
    "Oro (Gold)": "GC=F"
})

# Convertir nombre seleccionado a ticker
ticker = activo

# Horizonte de predicción
horizonte = st.radio("Horizonte de predicción:", ["1 Día", "1 Semana"])

if st.button("Ejecutar modelo"):
    # Descargar los datos
    df = yf.download(ticker, start="2020-01-01")

    if 'Close' not in df.columns:
        st.error("❌ Error al descargar los datos. Verifica el nombre del activo.")
    else:
        # Retorno
        df['Return'] = df['Close'].pct_change()

        # Target
        if horizonte == "1 Día":
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        else:
            df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)

        # Indicadores técnicos
        df['SMA'] = ta.trend.SMAIndicator(close=df['Close'], window=5).sma_indicator()
        df['Momentum'] = ta.momentum.RSIIndicator(close=df['Close'], window=5).rsi()
        df['Volatility'] = ta.volatility.BollingerBands(close=df['Close'], window=5).bollinger_band_width()

        df.dropna(inplace=True)

        # Features y etiquetas
        X = df[['SMA', 'Momentum', 'Volatility']]
        y = df['Target']

        # Entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predicción
        df['Prediction'] = model.predict(X)
        df['Strategy'] = df['Prediction'].shift() * df['Return']
        df['Cumulative Strategy'] = (1 + df['Strategy']).cumprod()
        df['Cumulative Buy & Hold'] = (1 + df['Return']).cumprod()

        # Gráfica de rendimiento
        st.subheader("📊 Rendimiento Estrategia vs Buy & Hold")
        st.line_chart(df[['Cumulative Strategy', 'Cumulative Buy & Hold']])

        # Reporte de clasificación
        st.subheader("📝 Reporte de clasificación")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
