# inference.py
import pandas as pd
import joblib
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import re
import os

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
model_container = "models"
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
    
    # Run inference
    predictions = model.predict(data)
    result_df = data.copy()
    result_df["weather_prediction"] = predictions
    
    # Upload results
    output_blob_client = blob_service_client.get_blob_client(
        output_container, f"predictions_{blob_name}"
    )
    output_buffer = BytesIO()
    result_df.to_csv(output_buffer, index=False)
    output_buffer.seek(0)
    output_blob_client.upload_blob(output_buffer, overwrite=True)
