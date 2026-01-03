from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["gender"]),
        float(request.form["age"]),
        float(request.form["smoking"]),
        float(request.form["yellow_fingers"]),
        float(request.form["anxiety"]),
        float(request.form["peer_pressure"]),
        float(request.form["chronic_disease"]),
        float(request.form["fatigue"]),
        float(request.form["allergy"]),
        float(request.form["wheezing"]),
        float(request.form["alcohol"]),
        float(request.form["coughing"]),
        float(request.form["shortness_of_breath"]),
        float(request.form["swallowing_difficulty"]),
        float(request.form["chest_pain"])
    ]

    final_features = scaler.transform([features])
    prediction = model.predict(final_features)

    result = "Lung Cancer Detected" if prediction[0] == 1 else "No Lung Cancer"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
