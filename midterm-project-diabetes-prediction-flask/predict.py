from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# ---- Load model ----
MODEL_PATH = "model.xgb"

print("Loading model...")
model = xgb.Booster()
model.load_model(MODEL_PATH)
print("Model loaded!")

# ---- Features ----
numeric_features = [
    'age', 'bmi', 'systolic_bp', 'hdl_cholesterol', 'ldl_cholesterol',
    'glucose_fasting', 'hba1c', 'physical_activity_minutes_per_week',
    'triglycerides', 'insulin_level', 'diabetes_risk_score',
    'family_history_diabetes', 'hypertension_history', 'cardiovascular_history'
]
cat_features = ['alcohol_group']


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    df = pd.DataFrame([data])

    # Creating categorical feature
    df['alcohol_group'] = pd.cut(
        df['alcohol_consumption_per_week'],
        bins=[-1, 2, 7, 10],
        labels=['Low', 'Medium', 'High']
    )

    # Ensure categoricals
    for col in cat_features:
        df[col] = df[col].astype("category")

    # Select only needed features for model
    model_features = numeric_features + cat_features
    dmatrix = xgb.DMatrix(df[model_features], enable_categorical=True)

    pred = model.predict(dmatrix)[0]

    return jsonify({
        "prediction_proba": float(pred),
        "prediction_label": int(pred > 0.5)
    })


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "Diabetes model API is running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
