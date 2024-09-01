import streamlit as st
import pandas as pd
import requests
import json
import plotly.graph_objs as go

API_URL = "http://127.0.0.1:8000/LSTM_Prediction"


st.title('Stock Price Prediction App')

stocks = ['', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT']
stock_name = st.selectbox('Please choose stock name', stocks)

if stock_name:
    st.write(f"Download historical stock data from: https://www.nasdaq.com/market-activity/stocks/{stock_name.lower()}/historical")

uploaded_file = st.file_uploader("Upload stock data (CSV file)")


last_date = None
predicted_prices = None

if not stock_name:
    st.error('Please select a stock name.')
else:
    if uploaded_file is not None:
        try:
            if uploaded_file.type == 'text/csv':
                stock_data = pd.read_csv(uploaded_file)
                required_columns = ['Date', 'Close/Last', 'Volume', 'Open', 'High', 'Low']
                if all(column in stock_data.columns for column in required_columns):
                    if len(stock_data) >= 61:
                        data = stock_data.copy()
                        data['Date'] = pd.to_datetime(data['Date'])
                        data = data.sort_values(by='Date')

                        last_date = data['Date'].values[-1]

                        columns_to_clean = ['Close/Last', 'Volume', 'Open', 'High', 'Low']
                        stock_data[columns_to_clean] = stock_data[columns_to_clean].replace('[\\$,]', '', regex=True).astype(float)
                        stock_data_str = "\n".join([",".join(map(str, row)) for row in stock_data.values])
                    else:
                        st.error('Please upload a stock data file with at least 61 rows.')
                else:
                    st.error('Uploaded file does not contain all required columns (Date, Close/Last, Volume, Open, High, Low).\n'
                             f'Download stock data from: https://www.nasdaq.com/market-activity/stocks/{stock_name.lower()}/historical to get the required data.')
            else:
                st.error("Please upload a CSV file.")
        except pd.errors.EmptyDataError:
            st.error('The uploaded file is empty. Please upload a valid stock data file.')
    else:
        st.error('Please upload a stock data file.')


if st.button('Predict'):
    if uploaded_file is None:
        st.error('No file uploaded. Please upload a stock data file')
    else:
        payload = {
            "stock_name": str(stock_name),
            "stock_data": stock_data_str
        }

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            predictions = response.json()
            if "prediction" in predictions:
                predicted_prices = predictions["prediction"]

                predicted_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predicted_prices), freq='B')

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=predicted_dates, y=predicted_prices, name='Predicted Price'))
                fig.update_layout(title=f"{stock_name} Predicted Price")
                st.plotly_chart(fig)

            else:
                st.error("Unexpected response format.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error making request: {e}")
