# app.py
from flask import Flask, request, jsonify
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import pandas as pd
import os

from inference import run_inference_on_blob
from train_model import retrain_model
from detect_drift import detect_drift
from utils.blob_helpers import list_weather_blobs

app = Flask(__name__)

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_container = "weather-data"  # same container stores monthly datasets
models_container = "models"

@app.route("/process", methods=["POST"])
def process_data():
    data = request.get_json()
    received_blob_name = data.get("blob_name")

    if not received_blob_name:
        return jsonify({"status": "failed", "error": "Missing blob_name in POST request"}), 400

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(blob_container)

        # List all blobs and sort by date in filename
        all_blobs = list_weather_blobs(container_client)

        if len(all_blobs) == 0:
            return jsonify({"status": "failed", "error": "No blobs found in container"}), 400

        elif len(all_blobs) == 1:
            # Only one blob exists → skip drift detection, run inference on received blob
            print("Only one blob in container — running inference on received blob...")
            run_inference_on_blob(received_blob_name)
            return jsonify({
                "status": "success",
                "message": "Only one blob — inference executed",
                "blob": received_blob_name
            }), 200

        else:
            # Two or more blobs → check drift using the latest two
            latest_blob_name = all_blobs[-1]
            previous_blob_name = all_blobs[-2]

            print(f"Downloading previous blob: {previous_blob_name}")
            prev_blob_client = container_client.get_blob_client(previous_blob_name)
            prev_df = pd.read_csv(BytesIO(prev_blob_client.download_blob().readall()))

            print(f"Downloading latest blob: {latest_blob_name}")
            latest_blob_client = container_client.get_blob_client(latest_blob_name)
            latest_df = pd.read_csv(BytesIO(latest_blob_client.download_blob().readall()))

            drifted = detect_drift(prev_df, latest_df)

            if drifted:
                print("Drift detected — Retraining model on latest blob...")
                retrain_model(latest_df)
            else:
                print("No drift detected — Running inference on received blob...")
                run_inference_on_blob(received_blob_name)

            return jsonify({
                "status": "success",
                "drift_detected": drifted,
                "previous_file": previous_blob_name,
                "latest_file": latest_blob_name,
                "inference_blob": received_blob_name
            }), 200

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)