# app.py
from flask import Flask, request, jsonify
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import pandas as pd
import os

from inference import run_inference_on_blob
from train_model import retrain_model
from detect_drift import detect_drift
from utils.blob_helpers import get_latest_weather_blob

app = Flask(__name__)

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_container = "weather-data"  # same container stores monthly datasets
models_container = "models"

@app.route("/process", methods=["POST"])
def process_data():
    data = request.get_json()
    new_blob_name = data.get("blob_name")

    if not new_blob_name:
        return jsonify({"error": "Missing blob_name"}), 400

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(blob_container)

        # Download new dataset (this month)
        print(f"⬇️ Downloading new data blob: {new_blob_name}")
        new_blob_client = container_client.get_blob_client(new_blob_name)
        new_df = pd.read_csv(BytesIO(new_blob_client.download_blob().readall()))

        # Get previous (latest existing) training dataset
        latest_blob_name = get_latest_weather_blob(blob_service_client, blob_container)
        if latest_blob_name == new_blob_name:
            return jsonify({"status": "ignored", "message": "Same as latest file"}), 200

        print(f"Downloading reference data blob: {latest_blob_name}")
        ref_blob_client = container_client.get_blob_client(latest_blob_name)
        reference_df = pd.read_csv(BytesIO(ref_blob_client.download_blob().readall()))

        # Drift detection
        drifted = detect_drift(reference_df, new_df)

        if drifted:
            print("Drift detected — Retraining model...")
            retrain_model(new_df)
        else:
            print("No drift detected — Running inference only...")
            run_inference_on_blob(new_blob_name)

        return jsonify({
            "status": "success",
            "drift_detected": drifted,
            "reference_file": latest_blob_name,
            "new_file": new_blob_name
        }), 200

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)