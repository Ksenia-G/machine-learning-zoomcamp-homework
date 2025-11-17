# Diabetes Risk Prediction Project

## Project Overview

This project implements a complete **machine learning pipeline** for predicting the likelihood of diabetes using health and lifestyle indicators.  
It is built using:

- **XGBoost** — for high-performance classification  
- **Flask API** — for serving real-time predictions  
- **KaggleHub** — for automatic dataset download  

Dataset used:  
[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset)

---

## Project Structure
diabetes_model/
├── Dockerfile
├── project.ipynb
├── main.py
├── model.xgb
├── train.py
├── predict.py
├── predictions.csv
├── pyproject.toml
└── README.md


### File Descriptions

- **`project.ipynb`** - full EDA and model selection.
- **`train.py`** — loads data, trains an XGBoost model, saves the trained model.
- **`predict.py`** — runs a Flask API that loads the model and provides predictions.
- **`Dockerfile`** — container definition for running the API.
- **`pyproject.toml`** — dependency list and project environment definition.

---

## Dataset Information

The dataset is automatically downloaded through **KaggleHub**.

- **Source:** Diabetes Health Indicators Dataset  
- **File:** `diabetes_dataset.csv`  
- **Features include:**  
  - Age, BMI  
  - Blood pressure, heart rate  
  - Cholesterol levels  
  - Glucose & insulin levels  
  - Lifestyle indicators (sleep, physical activity, smoking, alcohol)  
- **Target:** `diagnosed_diabetes`  
  - `0` → No diabetes  
  - `1` → Diagnosed with diabetes  

---

## Environment Setup

### 1. Clone the repository

```
git clone <repository_url>
cd diabetes_model
```

### 2. Training the Model

Run the training script:

```
python train.py
```


After training, the model will be saved as:

model.xgb

### 3. Running the Prediction API
1. Build the Docker image
```
docker build -t diabetes .
```

2. Run the container
```
docker run -p 8000:8000 diabetes
```

### 🔮 Example Prediction Request

Send a POST request to the API:

```
curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "age": 58,
        "gender": "Male",
        "ethnicity": "Asian",
        "education_level": "Highschool",
        "income_level": "Lower-Middle",
        "employment_status": "Employed",
        "smoking_status": "Never",
        "alcohol_consumption_per_week": 0,
        "physical_activity_minutes_per_week": 215,
        "diet_score": 5.7,
        "sleep_hours_per_day": 7.9,
        "screen_time_hours_per_day": 7.9,
        "family_history_diabetes": 0,
        "hypertension_history": 0,
        "cardiovascular_history": 0,
        "bmi": 30.5,
        "waist_to_hip_ratio": 0.89,
        "systolic_bp": 134,
        "diastolic_bp": 78,
        "heart_rate": 68,
        "cholesterol_total": 239,
        "hdl_cholesterol": 41,
        "ldl_cholesterol": 160,
        "triglycerides": 145,
        "glucose_fasting": 136,
        "glucose_postprandial": 236,
        "insulin_level": 6.36,
        "hba1c": 8.18,
        "diabetes_risk_score": 29.6,
        "diabetes_stage": "Type 2",
        "diagnosed_diabetes": 1
    }'

```
### 📬 Notes

The API returns a probability score: likelihood of diabetes.

The model can be retrained any time by running train.py.
