# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:50:42 2021

@author: Alvin Burhani
"""
import streamlit as st
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use('fivethirtyeight')


from pylab import rcParams
rcParams['figure.figsize'] = 10, 6

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

import math
import yfinance as yf
from datetime import date, timedelta

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
today_date = date.today()
td = timedelta(120)
mulai = today_date - td

from fbprophet import Prophet
m = Prophet(daily_seasonality=False)

#Test for staionarity
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    st.pyplot()
    
    st.write("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    st.write(output)
    
def make_forecast(ticker, periods, hist='max'):
    # pull historical data from yahoo finance
    stock_data = yf.Ticker(ticker)

    hist_data = stock_data.history(hist, auto_adjust=True)

    # create new dataframe to hold dates (ds) & adjusted closing prices (y)
    df = pd.DataFrame()

    df['ds'] = hist_data.index.values
    df['y'] = hist_data['Close'].values

    # create a Prophet model from that data
    m = Prophet(daily_seasonality=False)
    m.fit(df)

    future = m.make_future_dataframe(periods, freq='D')

    forecast = m.predict(future)

    m.plot(forecast)
    plt.xlabel('Tahun')
    plt.ylabel('Harga Saham')
    st.pyplot()

    return forecast

st.header('Prediksi Saham Bursa Efek Jakarta')
st.title('By Alvin Burhani')


KODE = st.text_input("Masukkan Kode Saham", "BBNI.JK")
n = st.slider("Prediksi Berapa Tahun",1,5,value=2)
if(st.button('Submit')):
    
    brk = yf.Ticker(KODE)
    data = yf.download(KODE, start=mulai, end = today_date)
    datam = yf.download(KODE, period="max", auto_adjust=True)
    lb = brk.info['longBusinessSummary']
    
    #plot close price
    plt.figure(figsize=(15,10))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Close Prices')
    plt.plot(datam['Close'])
    plt.title('Grafik Harga Penutupan Saham Perusahaan '+KODE)
    st.pyplot()
    
    lb
    
    
    plt.figure(figsize=(15,10))
    df_close = data['Close']
    df_close.plot(style='k.')
    plt.title(KODE + ' Scatter plot of closing price')
    #st.pyplot()
    
    test_stationarity(df_close)
    
    result = seasonal_decompose(df_close, model='multiplicative', freq = 30)
    fig = plt.figure()  
    fig = result.plot()  
    fig.set_size_inches(16, 9)
    st.pyplot()
    
    from pylab import rcParams
    rcParams['figure.figsize'] = 16, 9
    df_log = np.log(df_close)
    moving_avg = df_log.rolling(12).mean()
    std_dev = df_log.rolling(12).std()
    plt.legend(loc='best')
    plt.title('Moving Average')
    plt.plot(std_dev, color ="black", label = "Standard Deviation")
    plt.plot(moving_avg, color="red", label = "Mean")
    plt.legend(loc='best')
    st.pyplot()
    
    #split data into train and training set
    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
    plt.figure(figsize=(16,9))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Closing Prices')
    plt.plot(df_log, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')
    plt.legend()
    st.pyplot()
    
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                             test='adf',       # use adftest to find optimal 'd'
                             max_p=3, max_q=3, # maximum p and q
                             m=1,              # frequency of series
                             d=None,           # let model determine 'd'
                             seasonal=False,   # No Seasonality
                             start_P=0, 
                             D=0, 
                             trace=True,
                             error_action='ignore',  
                             suppress_warnings=True,
                             stepwise=True)
    
    model_autoARIMA.plot_diagnostics(figsize=(15,8))

    #model = ARIMA(train_data, order=(3, 1, 2))  
    #fitted = model.fit(disp=-1)  
    #st.markdown(fitted.summary())
    
    import statsmodels as sm
    model = sm.tsa.arima_model.ARIMA(train_data, order=(3, 1, 2))
    fitted = model.fit(disp=-1)
    #st.write(fitted.summary())

    # Forecast
    fc, se, conf = fitted.forecast(test_data.shape[0], alpha=0.05)  # 95% confidence
    fc_series = pd.Series(fc, index=test_data.index)
    lower_series = pd.Series(conf[:, 0], index=test_data.index)
    upper_series = pd.Series(conf[:, 1], index=test_data.index)
    
    plt.figure(figsize=(16,9), dpi=100)
    plt.style.use('bmh')
    plt.plot(train_data, label='training')
    plt.plot(test_data, color = 'blue', label='Actual Stock Price')
    plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.10)
    
    plt.title('Prediction Harga Saham' + KODE)
    plt.xlabel('Tanggal Transaksi')
    plt.ylabel('Harga Saham')
    plt.legend(loc='upper left', fontsize=8)

    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    st.pyplot()    


    # report performance
    mse = mean_squared_error(test_data, fc)
    st.success('MSE  : '+str(mse))
    mae = mean_absolute_error(test_data, fc)
    st.info('MAE  : '+str(mae))
    rmse = math.sqrt(mean_squared_error(test_data, fc))
    st.info('RMSE : '+str(rmse))
    mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
    st.info('MAPE : '+str(mape))
    
    dc = make_forecast(KODE, 365 * n).tail()