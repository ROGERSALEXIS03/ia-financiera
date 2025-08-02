import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="IA para Predicción de Activos Financieros")
st.title("📉 IA para Predicción de Activos Financieros")

# Menú para seleccionar activo
activo = st.selectbox("Selecciona el activo:", {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "S&P 500 (SPY)": "SPY",
    "Oro (Gold)": "GC=F"
})

# Horizonte de predicción
horizonte = st.radio("Horizonte de predicción:", ["1 Día", "1 Semana"])

# Botón para ejecutar el modelo
if st.button("Ejecutar modelo"):
    df = yf.download(activo, start="2020-01-01")

    if df.empty:
        st.error("❌ No se pudieron descargar los datos. Verifica el nombre del activo o tu conexión a Internet.")
        st.stop()

    if "Close" not in df.columns:
        st.error("❌ El DataFrame no contiene la columna 'Close'.")
        st.dataframe(df.head())
        st.stop()

    df["Return"] = df["Close"].pct_change()

    # Etiqueta del modelo
    if horizonte == "1 Día":
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    else:
        df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    # Indicadores técnicos
    df["SMA"] = ta.trend.sma_indicator(df["Close"], window=5)
    df["Momentum"] = ta.momentum.roc(df["Close"], window=5)
    df["Volatility"] = ta.volatility.bollinger_band_width(df["Close"], window=5)

    df.dropna(inplace=True)

    X = df[["SMA", "Momentum", "Volatility"]]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicciones
    df["Prediction"] = model.predict(X)

    df["Strategy"] = df["Prediction"].shift(1) * df["Return"]
    df["Cumulative Strategy"] = (1 + df["Strategy"]).cumprod()
    df["Cumulative Buy & Hold"] = (1 + df["Return"]).cumprod()

    # Resultados
    st.subheader("📈 Rendimiento Estrategia vs Buy & Hold")
    st.line_chart(df[["Cumulative Strategy", "Cumulative Buy & Hold"]])

    st.subheader("📊 Reporte de clasificación")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
