import streamlit as st
import yfinance as yf
import datetime
import time
import pandas as pd
import math

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , LSTM

import warnings
warnings.filterwarnings('ignore')


st.title('Prediksi Saham Perusahaan LQ45')

#KODE = "BBCA.JK"
awal = "2022-01-01"
akhir = "2023-02-10"

KODE = st.sidebar.selectbox(
    'Pilih Tampilan Saham LQ45',
    ('^JKLQ45','ACES.JK', 'ADRO.JK', 'AKRA.JK', 'ANTM.JK','ASII.JK','BBCA.JK','BBNI.JK','BBRI.JK','BBTN.JK',
     'BMRI.JK','BSDE.JK','BTPS.JK','CPIN.JK','CTRA.JK','ERAA.JK','EXCL.JK','GGRM.JK','HMSP.JK',
     'ICBP.JK','INCO.JK','INDF.JK','INKP.JK','INTP.JK','ITMG.JK','JPFA.JK','JSMR.JK','KLBF.JK',
     'MDKA.JK','MIKA.JK','MNCN.JK','PGAS.JK','PTBA.JK','PTPP.JK','PWON.JK','SCMA.JK','SMGR.JK',
     'SMRA.JK','SRIL.JK','TBIG.JK','TKIM.JK','TLKM.JK','TOWR.JK','UNTR.JK','UNVR.JK','WIKA.JK')
)



a=st.sidebar.date_input('Mulai Tanggal', datetime.date(2022,1,2))
b=st.sidebar.date_input('Hingga Tanggal', datetime.date(2023,2,9))
tgl=st.sidebar.date_input('Prediksi untuk Tanggal', datetime.date(2023,2,10))


st.write('Data Saham dari tanggal',a,' hingga tanggal',b)
df = yf.download(KODE, start=a, end=b)

comp = yf.Ticker(KODE)

if KODE == '^JKLQ45':
    st.write("Nama Saham Gabungan", comp.info['longName'], "(",KODE,")")
else:
    
    st.write('Nama Perusahaan :', comp.info['longName'], "(",KODE,")") 

    with st.expander("Lihat Profile"):
        st.write(comp.info['longBusinessSummary'])
        
    st.write('Sektor Usaha :', comp.info['sector'])
    st.write('Sektor Industri :', comp.info['industry'])
    st.write('Website Perusahaan :', comp.info['website'])
    st.write('Alamat Perusahaan :', comp.info['address1'], ' ', 
             comp.info['address2'], ' ',
             comp.info['zip'], ' ',
             comp.info['country'])
    st.write('No. Telephone : ',comp.info['phone'])
    st.write('No. Facsimile : ', comp.info['fax'])


    

jml_data = len(df)
jml_train = jml_data * 0.8

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Chart","Volume","Open","Close","Adj Close","Tabel Data"])

tab1.subheader("Grafik Harga Saham "+str(KODE))
fig = px.line(df,y=['High','Close','Low'])
tab1.plotly_chart(fig,use_container_width=True)

tab2.subheader("Grafik Jumlah Saham "+str(KODE))
tab2.bar_chart(df,y=['Volume'])

tab3.subheader("Grafik Harga Pembukaan Saham "+str(KODE))
tab3.line_chart(df,y=['Open'])

tab4.subheader("Grafik Harga Penutupan Saham "+str(KODE))
tab4.line_chart(df,y=['Close'])

tab5.subheader("Grafik Harga Koreksi Penutupan Saham "+str(KODE))
tab5.line_chart(df,y=['Adj Close'])

tab6.subheader("Tabel Data Saham "+str(KODE))
tab6.dataframe(df, use_container_width = True)


st.sidebar.info('Konfigurasi LSTM')
hari = st.sidebar.slider('Jumlah Hari untuk Prediksi Harga',1,300,60)
epo = st.sidebar.slider('Jumlah Epochs',5,200,10)
Lunit = st.sidebar.slider('Jumlah LSTM Unit',25,100,50)
bat = st.sidebar.slider('Jumlah Batch Size',1,50,15)
#verbo = st.sidebar.slider('Jumlah Verbose',0,3,1)

if st.sidebar.checkbox('Lihat Statistik Data'):
    st.table(df.describe().style.highlight_max(axis=0))
    
if jml_data < hari:
    st.error("Jumlah Hari untuk Prediksi harga seharusnya lebih pendek atau sama dengan tanggal mulai dan tanggal akhir!")
    
    
if st.button('Run LSTM Analisis'):

    #Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    #Converting the dataframe to a numpy array
    dataset = data.values
    #Get /Compute the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) *.8) 
    
    #Scale the all of the data to be values between 0 and 1 
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(dataset)
    
    #Create the scaled training data set 
    train_data = scaled_data[0:training_data_len  , : ]
    #Split the data into x_train and y_train data sets
    x_train=[]
    y_train = []
    for i in range(hari,len(train_data)):
        x_train.append(train_data[i-hari:i,0])
        y_train.append(train_data[i,0])
    
    #Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    #Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    #Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units=Lunit, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=Lunit, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()


   
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(10, text=progress_text)
    
    #Train the model
    history = model.fit(x_train, y_train, batch_size=bat, epochs=epo, verbose=1)
    
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1, text=progress_text)
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write('Grafik Evaluasi Model Loss vs Epoch')
    st.area_chart(history.history['loss'], use_container_width=True)
    
    
    #Test data set
    test_data = scaled_data[training_data_len - hari: , : ]
    #Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_len : , : ] 
    for i in range(hari,len(test_data)):
        x_test.append(test_data[i-hari:i,0])
    
    #Convert x_test to a numpy array 
    x_test = np.array(x_test)
    
    #Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    
    #Getting the models predicted price values
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions) #Undo scaling
    
    # report performance
    st.markdown("""---""")
    st.write('Jumlah Data Saham : '+"{:.0f}".format(jml_data))
    st.write('Jumlah Data Train : '+"{:.0f}".format(jml_train))
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    #rmse = math.sqrt(mean_squared_error(y_test, predictions))
    st.write('RMSE : '+"{:.3f}".format(rmse))
    mape = mean_absolute_percentage_error(y_test, predictions)
    #mape = np.mean(np.abs(predictions - y_test)/np.abs(y_test))*100
    st.write('MAPE : '+"{:.3f}".format(mape),'%')
    st.markdown("""---""")
    
    #Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    
    tab1,tab2 = st.tabs(["Grafik Prediksi","Grafik Prediksi & Aktual"])
    
    tab1.write("Grafik Prediksi Saham")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Aktual'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Prediksi'))
    tab1.plotly_chart(fig, use_container_width=True)
    
    tab2.write("Grafik Nilai Prediksi & Aktual")
    fig = px.line(valid[['Close', 'Predictions']])
    fig.update_layout(
        xaxis_tickformat = '%d %B %Y'
        )
    tab2.plotly_chart(fig, use_container_width=True)
    
    #Grafik sesuai Jumlah Hari
    st.write('Perbandingan Harga Aktuan & Prediksi dalam waktu '+"{:.0f}".format(hari),' hari')
    valid2 = valid.tail(hari)
    st.line_chart(valid2, use_container_width=True)

    st.subheader("Tabel data Prediksi & Aktual")
    valid2['Selisih'] = abs(valid2.Close - valid2.Predictions) 
    valid2['Error'] = valid2.Selisih/valid2.Close
    valid2['Akurasi'] = (1-valid2['Error'])*100

    fig = go.Figure(data=[go.Table(header=dict(values=['Tanggal','Prediksi', 'Valid', 'Deviasi', 'Akurasi']),
                 cells=dict(values=[valid2.index.strftime('%d-%m-%Y'),
                                    valid2.Predictions, 
                                    valid2.Close, round(valid2.Selisih,2), round(valid2.Akurasi,2)]))
                     ])
    st.plotly_chart(fig, use_container_width=True)
    
    writer = pd.ExcelWriter('prediksi.xlsx')
    valid2.to_excel(writer)
    writer.save()
    st.warning('File Prediksi sudah di simpan')
    
    #Get the quote
    saham_quote = yf.download(KODE, start=a, end=b)
    #Create a new dataframe
    new_df = saham_quote.filter(['Close'])
    #Get teh last x day closing price 
    last_x_days = new_df[-hari:].values
    #Scale the data to be values between 0 and 1
    last_x_days_scaled = scaler.transform(last_x_days)
    #Create an empty list
    X_test = []
    #Append teh past x days
    X_test.append(last_x_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
    prediksi_harga = np.array(pred_price).item()  
    with st.expander("Lihat Prediksi"):
        st.write('Prediksi Harga Kedepan adalah : '+"{:.2f}".format(prediksi_harga))
        
    st.info('Table Harga Saham')
    saham_quote2 = yf.download(KODE, start=a, end=tgl)
    harga = saham_quote2['Close']
    harga = harga.tail(5)
    st.dataframe(harga)
    
if st.button("Clear All Data Caches"):
    # Clear values from *all* all in-memory and on-disk data caches:
    # i.e. clear values from both square and cube
    st.cache_data.clear()
    
# running
# streamlit run C:\Users\python\streamlit\apkbaru.py
