import os
import pickle
from flask import Flask, request, render_template
import pandas as pd
import mlflow.sklearn

# Initialize Flask app
app = Flask(__name__)

# Tracking URI configuration: prefers environment variable, then local sqlite db, or http://127.0.0.1:5000/
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    if os.path.exists("mlflow.db"):
        tracking_uri = "sqlite:///mlflow.db"
    else:
        tracking_uri = "http://127.0.0.1:5000/"

mlflow.set_tracking_uri(uri=tracking_uri)
try:
    mlflow.set_experiment("Rainfall1")
except Exception as exp_err:
    print(f"Notice setting experiment: {exp_err}")

production_model_name = "rainfall-prediction-production"
prod_model_uri = f"models:/{production_model_name}@champion"

# Attempt to load model from MLflow registry, falling back to model.pkl if offline
loaded_model = None
try:
    print(f"Loading champion model from MLflow ({prod_model_uri}) with tracking URI '{tracking_uri}'...")
    loaded_model = mlflow.sklearn.load_model(prod_model_uri)
    print("Model loaded successfully from MLflow registry!")
except Exception as mlflow_err:
    print(f"MLflow load failed ({mlflow_err}). Attempting to load from local 'model.pkl'...")
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
        print("Model loaded successfully from local model.pkl!")
    else:
        print("model.pkl not found. Please run train.py first.")
        raise mlflow_err

# Feature names
feature_names = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

@app.route('/')
def home():
    return render_template('index.html', prediction_result=None, form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        input_data = [
            float(request.form['pressure']),
            float(request.form['dewpoint']),
            float(request.form['humidity']),
            float(request.form['cloud']),
            float(request.form['sunshine']),
            float(request.form['winddirection']),
            float(request.form['windspeed']),
        ]

        # Create DataFrame for the input
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Predict using the loaded model
        prediction = loaded_model.predict(input_df)

        # Convert prediction to human-readable format
        result = "Rainfall" if prediction[0] == 1 else "No Rainfall"

        return render_template('index.html', prediction_result=result, form_data=request.form)
    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {str(e)}", form_data=request.form)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Rainfall Prediction Flask Server on http://127.0.0.1:{port} ...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

