import pandas as pd
import xgboost as xgb
import kagglehub
from kagglehub import KaggleDatasetAdapter
import joblib

def load_data():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "mohankrishnathalla/diabetes-health-indicators-dataset",
        "diabetes_dataset.csv"
    )
    return df

def main():
    print("Loading dataset...")
    df = load_data()

    # Creating a new feature
    df['alcohol_group'] = pd.cut(df['alcohol_consumption_per_week'], 
                            bins=[-1, 2, 7, 10], 
                            labels=['Low', 'Medium', 'High'])

    # Features
    numeric_features = [
        'age', 'bmi', 'systolic_bp', 'hdl_cholesterol', 'ldl_cholesterol',
        'glucose_fasting', 'hba1c', 'physical_activity_minutes_per_week',
        'triglycerides', 'insulin_level', 'diabetes_risk_score',
        'family_history_diabetes', 'hypertension_history',
        'cardiovascular_history'
    ]

    cat_features = ['alcohol_group']
    features = numeric_features + cat_features
    target = 'diagnosed_diabetes'

    # Split train/test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    # Categoricals
    for col in cat_features:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # XGBoost matrices
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    print("Training model...")
    params = {
        'eta': 0.1,
        'max_depth': 4,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'seed': 1,
        'verbosity': 1,
        'eval_metric': 'auc',
        'tree_method': 'hist'
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=200
    )

    print("Saving model to model.xgb...")
    model.save_model("model.xgb")

    print("Saving test set to test.csv...")
    test_df.to_csv("test.csv", index=False)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
