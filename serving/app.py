"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
from datetime import datetime as dt
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib  
import wandb
import json as json_lib


#import ift6758


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODEL_DIR = "artifacts"


app = Flask(__name__)

# def load_model(model_name):
#     # Initialize model with weights
#     weights = get_model_weights(model_name).DEFAULT
#     model = get_model(model_name, weights=weights)
#     model.eval()
#     return model, weights


# @app.before_first_request
# def before_first_request():
"""
Hook to handle any initialization before the first request (e.g. load model,
setup logging handler, etc.)
"""
# TODO: setup basic logging configuration
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
app.logger.info("Starting Flask app...")

# def load_local_model(model_name: str):
#     """Load model from local disk."""
#     global CURRENT_MODEL, CURRENT_MODEL_NAME

#     model_path = Path(MODEL_DIR) / f"{model_name}" / f"{model_name}.joblib"
#     if not model_path.exists():
#         return None

#     CURRENT_MODEL = joblib.load(model_path)
#     CURRENT_MODEL_NAME = model_name
#     app.logger.info(f"Model loaded: {model_name}")
#     return CURRENT_MODEL


# TODO: any other initialization before the first request (e.g. load default model)
# load_local_model("distance")
current_model = None
current_model_info = None
# model, weights = load_model(os.environ.get("MODEL_NAME"))

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    #Read the log file specified and return the data
    if not os.path.exists(LOG_FILE):
        return jsonify({"error": "Log file not found"}), 404
    
    with open(LOG_FILE, "r") as f:
        data = f.read()

    response = {"logs": data}
    return jsonify(response)  # response must be json serializable!

def _get_wandb_model_path(entity: str, project: str, model_name: str, version: str) -> str:
    """
    Construct W&B model registry path.
    
    Args:
        entity: W&B entity (workspace)
        project: W&B project name
        model_name: Model name in registry
        version: Version (can be 'latest' or specific version like 'v1')
    
    Returns:
        W&B model path string
    """
    return f"{entity}/{project}/{model_name}:{version}"


def _get_cache_path(entity: str, model_name: str, version: str) -> Path:
    """Get local cache file path for a model."""
    cache_dir = Path("model_cache")
    cache_dir.mkdir(exist_ok=True)
    filename = f"{entity}_{model_name}_{version}.joblib"
    return cache_dir / filename


def _load_model_from_wandb(entity: str, project: str, model_name: str, version: str):
    """
    Download and load model from W&B model registry.
    
    Args:
        entity: W&B entity (workspace)
        project: W&B project
        model_name: Model name
        version: Version identifier
    
    Returns:
        Loaded model object
    
    Raises:
        Exception: If model cannot be downloaded
    """
    model_path = _get_wandb_model_path(entity, project, model_name, version)
    
    try:
        app.logger.info(f"Downloading model from W&B: {model_path}")
        
        # Use wandb.Api to download model
        api = wandb.Api()
        model_artifact = api.artifact(model_path)
        
        # Download artifact to temporary directory
        artifact_dir = model_artifact.download()
        
        # Find the model file in the artifact
        model_files = list(Path(artifact_dir).glob("*.joblib"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in artifact: {model_path}")
        
        # Load the first model file found
        model_file = model_files[0]
        app.logger.info(f"Loading model from: {model_file}")
        
        model = joblib.load(model_file)
        app.logger.info(f"Successfully downloaded and loaded model: {model_path}")
        
        return model, artifact_dir
    
    except Exception as e:
        app.logger.error(f"Failed to download model from W&B: {str(e)}")
        raise


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }

        {
            "entity": (required),
            "project": (required),
            "model_name": (required),
            "version": (required, can be "latest")
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    entity = json["entity"]
    project = json["project"]
    model_name = json["model_name"]
    version = json["version"]

    global current_model, current_model_info
    
    model_id = f"{entity}/{project}/{model_name}:{version}"
    
    # Check if model is already cached
    cache_path = _get_cache_path(entity, model_name, version)
    
    global current_model, current_model_info

    # TODO: check to see if the model you are querying for is already downloaded 
    if cache_path.exists():
        app.logger.info(f"Model already cached: {model_id}")
        try:
            # Load cached model
            model = joblib.load(cache_path)
            current_model = model
            current_model_info = {
                "entity": entity,
                "project": project,
                "model_name": model_name,
                "version": version,
                "loaded_at": dt.now().isoformat(),
                "source": "cache"
            }
            app.logger.info(f"Successfully loaded model from cache: {model_id}")
            
            response = {
                "status": "success",
                "message": f"Model loaded from cache: {model_id}",
                "model_info": current_model_info
            }
        
        except Exception as e:
            error_msg = f"Failed to load cached model: {str(e)}"
            app.logger.error(error_msg)
            return jsonify({"error": error_msg}), 500
        

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here
    else:
        try:
            app.logger.info(f"Attempting to download model: {model_id}")
            
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if not wandb_api_key:
                app.logger.warning("WANDB_API_KEY environment variable not set")
            # Download using ift6758 client (you may need to adjust based on your implementation)
            # This is a placeholder - adjust to match your actual client
            model, artifact_dir = _load_model_from_wandb(entity, project, model_name, version)
            
            if model is None:
                raise ValueError(f"Model not found in registry: {model_id}")
            
            # Save to cache
            joblib.dump(model, cache_path)
            
            current_model = model
            current_model_info = {
                "entity": entity,
                "project": project,
                "model_name": model_name,
                "version": version,
                "loaded_at": dt.now().isoformat(),
                "source": "registry"
            }
            app.logger.info(f"Successfully downloaded and loaded model: {model_id}")
            
            response = {
                "status": "success",
                "message": f"Model downloaded and loaded: {model_id}",
                "model_info": current_model_info
            }
        
        except Exception as e:
            error_msg = f"Failed to download model from registry: {str(e)}"
            app.logger.error(error_msg)

            # Keep currently loaded model
            if current_model is not None:
                app.logger.info(f"Keeping current model: {current_model_info}")
                response = {
                    "status": "error",
                    "message": error_msg,
                    "current_model": current_model_info
                }
            else:
                response = {
                    "status": "error",
                    "message": error_msg + " (No model currently loaded)"
                }

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data

    global current_model, current_model_info

    # TODO:
    # raise NotImplementedError("TODO: implement this enpdoint")
    if current_model is None:
        error_msg = "No model currently loaded. Please download a model first."
        app.logger.warning(error_msg)
        response = {"error": error_msg}

    json = request.get_json()
    app.logger.info(json)

    try:
        X = pd.read_json(json_lib.dumps(json))
    except Exception as e:
        error_msg = f"Failed to parse input features: {str(e)}"
        app.logger.error(error_msg)
        response = {"error": error_msg}
        # Make predictions
    try:
        # Try predict_proba first (for probabilistic models like logistic regression)
        if hasattr(current_model, 'predict_proba'):
            predictions_proba = current_model.predict_proba(X)
            predictions_class = current_model.predict(X)
            
            # Format response
            response = {
                "predictions": predictions_proba.tolist(),
                "predicted_class": predictions_class.tolist(),
                "model_info": current_model_info,
                "n_samples": len(X)
            }
            
        else:
            # Fall back to predict if predict_proba not available
            predictions = current_model.predict(X)
            response = {
                "predictions": predictions.tolist(),
                "model_info": current_model_info,
                "n_samples": len(X)
            }
        app.logger.info(f"Successfully generated predictions for {len(X)} samples")
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({"error": error_msg}), 500


    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!
