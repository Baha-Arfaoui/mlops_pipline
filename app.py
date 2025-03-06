from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import pandas as pd
# Import all necessary functions directly from the module
import churn_model_pipeline as cmp

app = Flask(__name__)
Swagger(app)

MODEL_PATH = "models/model.joblib"
FEATURE_NAMES_PATH = "data/feature_names.joblib"

# Load model and feature names
try:
    model = cmp.load_model()  # Load the pre-trained model
    feature_names = joblib.load(FEATURE_NAMES_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

@app.route("/predict", methods=["POST"])
def predict_churn():
    """
    Predict if a customer will churn based on input features.
    ---
    parameters:
      - name: features
        in: body
        required: true
        schema:
          type: object
          properties:
            Account length:
              type: number
            Area code:
              type: number
            Number vmail messages:
              type: number
            Total day minutes:
              type: number
            Total day calls:
              type: number
            Total eve minutes:
              type: number
            Total eve calls:
              type: number
            Total night minutes:
              type: number
            Total night calls:
              type: number
            Total intl minutes:
              type: number
            Total intl calls:
              type: number
            Customer service calls:
              type: number
            International plan_Yes:
              type: number
            Region_Northeast:
              type: number
            Region_South:
              type: number
            Region_West:
              type: number
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: string
            probability:
              type: number
    """
    try:
        # Load model if not already available
        model = joblib.load(MODEL_PATH)
        if model is None:
            return jsonify({"error": "Model not loaded."}), 500
            
        request_data = request.get_json()
        input_df = pd.DataFrame([request_data])
        input_df = input_df[feature_names]  # Ensure correct feature order
        
        prediction_int = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        prediction_label = "churn" if prediction_int == 1 else "non churn"
        
        return jsonify({"prediction": prediction_label, "probability": probability})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

@app.route("/retrain", methods=["POST"])
def retrain_model():
    """
    Retrain the churn prediction model without MLflow.
    ---
    responses:
      200:
        description: Model retrained successfully
      500:
        description: Retraining failed
    """
    try:
        # Prepare data and retrain model using the functions
        cmp.prepare_data()
        model = cmp.train_model()
        
        # No need to separately load model as retrain_model returns the model
        # Save the trained model
        joblib.dump(model, MODEL_PATH)
        
        return jsonify({"message": "✅ Model retrained and loaded successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"❌ Retraining failed: {e}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    ---
    responses:
      200:
        description: API is running
        schema:
          type: object
          properties:
            status:
              type: string
    """
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
