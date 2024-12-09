from flask import Blueprint, request, jsonify
import pandas as pd
from .classification_logic import  classify_custom, classify_custom_parallel
from sklearn.metrics import classification_report
from llm_app.extensions import csrf_protect
import pdb

import json
from datetime import datetime

LOG_FILE = "/Users/ag/Prophet securit/takehome/classification_log.json"  # Path to the log file

blueprint = Blueprint("classification", __name__)

# Helper function to load dataset from URL
def load_dataset_from_url(url):
    try:
        # Assuming the dataset is in Parquet format
        dataset = pd.read_parquet(url)
        return dataset
    except Exception as e:
        return str(e)

# Helper function for metrics calculation
def calculate_metrics(true_labels, predicted_labels):
    try:

        report = classification_report(true_labels, predicted_labels, output_dict=True)
        return report
    except Exception as e:
        return str(e)


@blueprint.route("/classify", methods=["POST"])
@csrf_protect.exempt
def user_query_classification():
    """
    Live user query classification endpoint
    """
    print("classify end point hit...")
    data = request.json
    query = data.get("query")
    labels = data.get("labels")
    few_shot_examples = data.get("few_shot_examples")

    if not query or not labels:
        return jsonify({"error": "Query and labels are required"}), 400

    # Extract dataset URL from the query (natural language query)
    dataset_url = None
    if "https" in query:
        dataset_url = query.split("https")[1]
        dataset_url = "https" + dataset_url.split()[0]  # Extract URL

    if not dataset_url:
        return jsonify({"error": "Dataset URL not found in query"}), 400

    # Load dataset
    dataset = load_dataset_from_url(dataset_url)
    if isinstance(dataset, str):  # Error occurred while loading dataset
        return jsonify({"error": f"Failed to load dataset: {dataset}"}), 400

    # Check if the dataset contains required columns
    if "text" not in dataset.columns:
        return jsonify({"error": "Dataset must contain a 'text' column"}), 400

    print("Calling classify_custom...")
    
    # Perform classification on the entire dataset in batches
    # predictions = classify_custom(dataset, labels, few_shot_examples)
    
    import time
    start_time = time.time()
    # predictions = classify_custom_parallel(dataset, labels, few_shot_examples)
    predictions = classify_custom(dataset, labels, few_shot_examples)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    pdb.set_trace()
    prediction_labels = [int(prediction["prediction"]) for prediction in predictions]
    
    # Log the results before returning
    log_results(dataset_url, predictions, None)
    
    return jsonify({"predictions": predictions, "time_taken": end_time - start_time})


def log_results(dataset_url, predictions, metrics=None):
    """
    Log the results of the API call to a log file.
    """
    log_entry = {
        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_url": dataset_url,
        "type": "offline" if metrics else "online",  # Determine type based on presence of metrics
        "predictions_count": len(predictions),
        "metrics": metrics
    }

    # Read existing logs
    try:
        with open(LOG_FILE, "r") as log_file:
            logs = json.load(log_file)
    except FileNotFoundError:
        logs = []  # Initialize empty log if file doesn't exist

    # Append the new entry
    logs.append(log_entry)

    # Write back to the log file
    with open(LOG_FILE, "w") as log_file:
        json.dump(logs, log_file, indent=4)
    print(f"Log entry added: {log_entry}")


@blueprint.route("/evaluate", methods=["POST"])
@csrf_protect.exempt
def evaluate_classification():
    """
    Evaluation endpoint with scoring and metrics calculation
    """
    data = request.json
    query = data.get("query")
    labels = data.get("labels")
    few_shot_examples = data.get("few_shot_examples")

    if not query or not labels:
        return jsonify({"error": "Query and labels are required"}), 400

    # Extract dataset URL from the query (natural language query)
    dataset_url = None
    if "https" in query:
        dataset_url = query.split("https")[1]
        dataset_url = "https" + dataset_url.split()[0]  # Extract URL

    if not dataset_url:
        return jsonify({"error": "Dataset URL not found in query"}), 400

    # Load dataset
    dataset = load_dataset_from_url(dataset_url)
    if isinstance(dataset, str):  # Error occurred while loading dataset
        return jsonify({"error": f"Failed to load dataset: {dataset}"}), 400

    # Check if the dataset contains required columns
    if "text" not in dataset.columns or "label" not in dataset.columns:
        return jsonify({"error": "Dataset must contain 'text' and 'label' columns"}), 400

    # Perform classification on the entire dataset in batches
    import time
    start_time = time.time()
    # predictions = classify_custom_parallel(dataset, labels, few_shot_examples)
    predictions = classify_custom(dataset, labels, few_shot_examples)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # predictions = classify_custom(dataset, labels, few_shot_examples)
    # predictions = classify_custom_parallel(dataset, labels, few_shot_examples)
    

    prediction_labels = [int(prediction["prediction"]) for prediction in predictions]
    # Extract true labels from the dataset and adjust to match the number of predictions
    true_labels = dataset["label"].tolist()
    true_labels = true_labels[:len(prediction_labels)]  # Trim true labels to match predictions length

    
    # Calculate evaluation metrics
    metrics = calculate_metrics(true_labels, prediction_labels)

    # Log the results before returning
    log_results(dataset_url, predictions, metrics)
    # pdb.set_trace()
    # return jsonify({"predictions": predictions, "metrics": metrics, "time_taken": end_time - start_time})
    return jsonify({"metrics": metrics, "time_taken": end_time - start_time})



