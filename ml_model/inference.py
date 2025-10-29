# inference.py
import pandas as pd
import joblib
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import re
import os

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
model_container = "models"
preprocessor_container = "preprocessors"
output_container = "daily-weather-data"
input_container = "raw-daily-weather-data"

def get_latest_model(blob_service_client, base_name="weather_model"):
    """
    Finds the latest model version in blob storage.
    """
    container_client = blob_service_client.get_container_client(model_container)
    blobs = list(container_client.list_blobs())
    
    pattern = re.compile(f"{base_name}_(\\d+\\.\\d+)\\.pkl")
    versions = []
    for blob in blobs:
        match = pattern.match(blob.name)
        if match:
            versions.append((float(match.group(1)), blob.name))
    
    if not versions:
        raise ValueError("No models found in blob storage.")
    
    # Pick highest version number
    latest_blob = max(versions, key=lambda x: x[0])[1]
    print(f"Using latest model: {latest_blob}")
    
    blob_client = container_client.get_blob_client(latest_blob)
    buffer = BytesIO()
    blob_client.download_blob().readinto(buffer)
    buffer.seek(0)
    
    model = joblib.load(buffer)
    return model

def get_latest_preprocessor(blob_service_client, base_name="preprocessor"):
    """
    Finds and loads the latest version of the preprocessor from blob storage.
    """
    container_client = blob_service_client.get_container_client(preprocessor_container)
    blobs = list(container_client.list_blobs())
    
    # Match files like: preprocessor_1.0.pkl, preprocessor_2.3.pkl, etc.
    pattern = re.compile(f"{base_name}_(\\d+\\.\\d+)\\.pkl")
    versions = []
    for blob in blobs:
        match = pattern.match(blob.name)
        if match:
            versions.append((float(match.group(1)), blob.name))
    
    if not versions:
        raise ValueError("No preprocessors found in blob storage.")
    
    # Pick the one with the highest version number
    latest_blob = max(versions, key=lambda x: x[0])[1]
    print(f"Using latest preprocessor: {latest_blob}")
    
    # Download from blob storage
    blob_client = container_client.get_blob_client(latest_blob)
    buffer = BytesIO()
    blob_client.download_blob().readinto(buffer)
    buffer.seek(0)
    
    preprocessor = joblib.load(buffer)
    return preprocessor

def run_inference_on_blob(blob_name):
    """
    Runs inference on the specified blob in the input container.
    """
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Download new data
    input_blob_client = blob_service_client.get_blob_client(input_container, blob_name)
    data = pd.read_csv(BytesIO(input_blob_client.download_blob().readall()))
    
    # Load latest model
    model = get_latest_model(blob_service_client)
    
    #Preprocessing Data for Inference
    # Creating Month, Day and Season features
    result_df = data.copy()
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['day'] = pd.to_datetime(data['date']).dt.day

    data["season"] = pd.cut(data['month'],
        bins=[0, 2, 5, 8, 11, 12],
        labels=['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'],
        right=True,
        include_lowest=True,
        ordered=False)

    #Dropping date column
    data.drop(columns=['date'], inplace=True)

    # Define Target variable
    target = 'temperature_2m_mean'

    # Split data into X
    X = data.drop(columns=[target])

    # Load latest preprocessor
    preprocessor = get_latest_preprocessor(blob_service_client)
    X = pd.DataFrame(preprocessor.transform(X), columns=preprocessor.get_feature_names_out())

    # Run inference
    predictions = model.predict(X)
    result_df["weather_prediction"] = predictions
    
    # Upload results
    output_blob_client = blob_service_client.get_blob_client(
        output_container, f"predictions_{blob_name}"
    )
    output_buffer = BytesIO()
    result_df.to_csv(output_buffer, index=False)
    output_buffer.seek(0)
    output_blob_client.upload_blob(output_buffer, overwrite=True)
