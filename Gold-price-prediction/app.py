
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import requests
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set page config
st.set_page_config(page_title="Gold Price Prediction", page_icon=":rocket:", layout="wide")


# functions 

def loader_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

home = loader_url("https://lottie.host/f40844f5-57eb-416f-9176-5458ae83236d/gHpQbR3UXd.json")

@st.cache_data
def load_data():
    df = pd.read_csv('./dataset/Gold-price.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_data()
gld_series = df['Price']

with st.sidebar:
    selected = option_menu(
         "Main Menu",
         ["Overview", "Exploratory Data", "Stationarity", "Forecast"],
         icons=["house", "graph-up", "check2-square", "bar-chart", "calendar"],
         menu_icon="cast",
         default_index=0,
    )

if selected == "Overview":
    st.title("Gold Price Prediction using ARIMA")
    st.write("-----")
    st.lottie(home, height=300, key="home")

elif selected == "Exploratory Data":
    st.header("Exploratory Data Analysis")
    st.subheader("Historical Gold Price Data")
    st.line_chart(gld_series)
    
    st.subheader("Data Snapshot")
    st.write(df.head())

    if st.checkbox("Show ACF and PACF Plots (of differenced series)"):
        gld_diff = gld_series.diff().dropna()
        
        st.subheader("ACF Plot")
        fig_acf, ax_acf = plt.subplots(figsize=(10, 5))
        plot_acf(gld_diff, ax=ax_acf)
        st.pyplot(fig_acf)
        
        st.subheader("PACF Plot")
        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 5))
        plot_pacf(gld_diff, ax=ax_pacf)
        st.pyplot(fig_pacf)

elif selected == "Stationarity":
    st.header("Stationarity Test & Differencing")
    result = adfuller(gld_series)
    st.write("**Augmented Dickey-Fuller Test Results**")
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")
    
    if result[1] > 0.05:
        st.write("Data is non-stationary. Differencing the data to achieve stationarity.")
        # First differencing
        gld_diff = gld_series.diff().dropna()
        fig_diff, ax_diff = plt.subplots(figsize=(10, 5))
        ax_diff.plot(gld_diff, marker='o')
        ax_diff.set_title("Differenced Gold Price Series")
        ax_diff.set_xlabel("Date")
        ax_diff.set_ylabel("Differenced Price")
        st.pyplot(fig_diff)
        
        result_diff = adfuller(gld_diff)
        st.write("**After Differencing**")
        st.write(f"ADF Statistic: {result_diff[0]:.4f}")
        st.write(f"p-value: {result_diff[1]:.4f}")
    else:
        st.write("Data is stationary. No differencing is required.")

# Forecast Page
elif selected == "Forecast":
    st.header("Gold Price Forecast")
    # Fit the model if not already done
    model = ARIMA(gld_series, order=(1, 1, 1))
    model_fit = model.fit()
    
    forecast_horizon = st.slider("Select forecast horizon (number of future periods)", 
                                 min_value=1, max_value=30, value=10)
    forecast = model_fit.forecast(steps=forecast_horizon)
    
    st.subheader("Forecasted Gold Prices")
    st.write(forecast)
    
    fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
    ax_forecast.plot(gld_series, label='Historical Gold Price')
    last_date = gld_series.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=forecast_horizon+1)[1:]
    ax_forecast.plot(forecast_index, forecast, label='Forecast', linestyle='--', color='red')
    ax_forecast.set_title("Gold Price Forecast using ARIMA(1,1,1)")
    ax_forecast.set_xlabel("Date")
    ax_forecast.set_ylabel("Gold Price")
    ax_forecast.legend()
    st.pyplot(fig_forecast)
