from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from azure.storage.blob import BlobServiceClient


app = FastAPI()

MODEL_CONTAINER_NAME = 'your_model_container_name'
ACCOUNT_NAME = 'your_account_name'
ACCOUNT_KEY = 'your_account_key'

def load_model_from_azure(account_name, account_key, container_name, stock_name):
    """Load LSTM model from Azure Blob Storage based on stock name."""
    try:
        conn_str = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        file_name = f"{stock_name.upper()}.h5"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        with open(file_name, "wb") as data:
            data.write(blob_client.download_blob().readall())
        model = load_model(file_name)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model from Azure Blob Storage: {e}")

class StockRequest(BaseModel):
    stock_name: str
    stock_data: str

def preprocess_stock_data(stock_data_df):
    """Scale stock data and prepare input features for making predictions."""
    features = ['Close/Last', 'Volume', 'Open', 'High', 'Low']

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data_df[features])

    X = []
    y = []
    for i in range(scaled_data.shape[0] - 60, scaled_data.shape[0]):
        X.append(scaled_data[i-60:i, :])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    return X, scaler


@app.post("/LSTM_Prediction")
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name
    stock_data = stock_request.stock_data

    try:
        model = load_model_from_azure(ACCOUNT_NAME, ACCOUNT_KEY, MODEL_CONTAINER_NAME, stock_name)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    try:
        stock_data_df = pd.read_csv(io.StringIO(stock_data))
        stock_data_df.columns = ['Date', 'Close/Last', 'Volume', 'Open', 'High', 'Low']
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load stock data from request: {e}")

    X, scaler = preprocess_stock_data(stock_data_df)

    predictions = model.predict(X)

    predicted_data = np.zeros((60, 5))
    predicted_data[:, 0] = predictions.flatten()

    predicted_data = scaler.inverse_transform(predicted_data)

    predicted_prices = predicted_data[:, 0].tolist()

    return {'prediction': predicted_prices}

