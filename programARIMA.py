# -*- coding: utf-8 -*-
"""
Created on Fri May 13 18:42:11 2022

@author: Muhammad Burhanuddin
"""
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from pylab import rcParams

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

import math
#from datetime import date, timedelta

st.header("PREDIKSI HARGA SAHAM DENGAN ARIMAX")

mulai = st.sidebar.date_input('Mulai')
today_date = st.sidebar.date_input('Hingga')

KODE = st.text_input('Kode Saham', '^JKLQ45')

import yfinance as yf
df = yf.download(KODE, start=mulai, end = today_date)

st.line_chart(df[['Open','Close','High','Low','Adj Close']])

if st.sidebar.checkbox('Menu',True):
    pilihan = st.sidebar.selectbox(
        'Pilih Tampilan Data',
        ('Volume', 'Open', 'Close', 'High', 'Low')
    )
    
    if pilihan == 'Volume':
        st.write('#### Volume Saham ####')
        st.line_chart(df.Volume)
    
    if pilihan == 'Close':
        st.line_chart(df.Close)
        
    if pilihan == 'Open':
        st.line_chart(df.Open)
    
    if pilihan == 'High':
        st.line_chart(df.High)
    
    if pilihan == 'Low':
        st.line_chart(df.Low)

st.write("#### Grafik Distribusi Harga Rata-Rata ####")
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots()

s = df.Close
avg = s.describe().loc['mean']
plt.axvline(x=avg, color='r', linestyle='--')
ax = s.plot(kind='kde')
st.pyplot(fig)

st.write("#### Tabel Statistik ####")
st.table(s.describe())
    
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

#Test untuk staionarity
def test_stationarity(timeseries):
    #Determinasi rolling statistics
    rolmean = timeseries.rolling(5).mean()
    rolstd = timeseries.rolling(5).std()
    
    #Plot rolling statistics:
    plt.figure(figsize=(15,10))
    plt.style.use('seaborn-whitegrid')
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean dan Standard Deviation')
    plt.show(block=False)
    st.pyplot()
    
    st.write("#### Results of Dickey Fuller Test ####")
    adft = adfuller(timeseries,autolag='AIC')
    
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    st.write(output)

if st.sidebar.checkbox('Rolling Statistik',True):
    st.write("#### Grafik Rolling Statistik ####")
    test_stationarity(s)
    
if st.sidebar.checkbox('Seasonal Decomposed', True):
    st.write("#### Grafik Seasonal Decomposed ####")
    df_close = s
    result = seasonal_decompose(df_close, model='multiplicative', freq = 30)
    plt.style.use('seaborn')
    fig = plt.figure()  
    fig = result.plot()  
    fig.set_size_inches(16, 9)
    st.pyplot(fig)
    
if st.sidebar.checkbox('Moving Avarage', True):
    st.write("#### Grafik Moving Avarage ####")
    rcParams['figure.figsize'] = 16, 9
    df_log = np.log(s)
    moving_avg = df_log.rolling(12).mean()
    std_dev = df_log.rolling(12).std()
    fig, ax = plt.subplots()
    plt.legend(loc='best')
    plt.title('Moving Average')
    plt.plot(std_dev, color ="black", label = "Standard Deviation")
    plt.plot(moving_avg, color="red", label = "Mean")
    plt.legend(loc='best')
    st.pyplot(fig)
    
if st.sidebar.checkbox('Data Train & Data Test', True):
    #split data jadi data train dan data training
    st.write("#### Grafik Data Train & Data Test: ####")
    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
    plt.figure(figsize=(16,9))
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Closing Prices')
    plt.plot(df_log, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')
    plt.legend()
    st.pyplot()

if st.sidebar.checkbox('ARIMA', True):
    st.write("#### Model Auto Arima ####")
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                             test='adf',       # gunakan adf test untuk mencari nilai optimal 'd'
                             max_p=3, max_q=3, # maximum p dan q
                             m=1,              # frequency dari data series
                             d=None,           # Biarkan model untuk 'd' none
                             seasonal=False,   # Tidak ada data musiman
                             start_P=0, 
                             D=0, 
                             trace=True,
                             error_action='ignore',  
                             suppress_warnings=True,
                             stepwise=True)
    st.write(model_autoARIMA.summary())

if st.sidebar.checkbox('Tampilkan Grafik Model', True):
    st.write("#### Model ARIMA ####")
    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    plt.show()
    st.pyplot()
    
if st.sidebar.checkbox('Grafik Prediksi', True):
    #import os
    
    with st.form(key='my_form_to_submit'):
        st.write("#### Input Nilai SARIMAX ####")
        p = st.number_input("Enter number p", 0, 10, 0, 1)
        q = st.number_input("Enter number q", 0, 10, 0, 1)
        d = st.number_input("Enter number d", 0, 10, 0, 1)
        submit_button = st.form_submit_button(label='ARIMA')

    if submit_button:
        
        model = ARIMA(train_data, order=(p, q, d))  
        fitted = model.fit(disp=-1)  
        print(fitted.summary())
        
        st.write("#### Grafik Prediksi ####")
    
        # Forecast
        fc, se, conf = fitted.forecast(test_data.shape[0], alpha=0.05)  # 95% confidence
        fc_series = pd.Series(fc, index=test_data.index)
        lower_series = pd.Series(conf[:, 0], index=test_data.index)
        upper_series = pd.Series(conf[:, 1], index=test_data.index)
        
        #plt.figure(figsize=(16,9), dpi=100)
        plt.style.use('bmh')
        plt.plot(train_data, label='training')
        plt.plot(test_data, color = 'blue', label='Actual Price')
        plt.plot(fc_series, color = 'orange',label='Predicted Price')
        plt.fill_between(lower_series.index, lower_series, upper_series,
                         color='k', alpha=.10)
    
        plt.title('Prediction Harga Saham' + KODE)
        plt.xlabel('Tanggal Transaksi')
        plt.ylabel('Harga')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()
        st.pyplot()
    
        z = pd.Series(fc)
        
        a = z.describe().loc['mean']*1000
        b = z.describe().loc['max']*1000
        c = z.describe().loc['min']*1000
    
        st.write("#### Harga Prediksi ####")
        st.write(f'Estimasi Harga Rata-Rata {a:.2f} ')
        st.write(f'Estimasi Harga Tertinggi {b:.2f} ')
        st.write(f'Estimasi Harga Terendah {c:.2f} ')
        
        st.write("#### Report Performance ####")
        mse = mean_squared_error(test_data, fc)
        st.write('MSE  : '+str(mse))
        mae = mean_absolute_error(test_data, fc)
        st.write('MAE  : '+str(mae))
        rmse = math.sqrt(mean_squared_error(test_data, fc))
        st.write('RMSE : '+str(rmse))
        mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
        st.write('MAPE : '+str(mape))

    