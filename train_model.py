import json
import pickle
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_FILE = "synthetic_fraud_dataset.csv"
MODEL_FILE = "model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"
CONFIG_FILE = "model_config.json"


def detect_risk_score_scale(df):
    risk_min = float(df["Risk_Score"].min())
    risk_max = float(df["Risk_Score"].max())

    print("Risk_Score min:", risk_min)
    print("Risk_Score max:", risk_max)

    if 0 <= risk_min and risk_max <= 1:
        return "0_to_1"
    return "0_to_100"


def print_results(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred) * 100
    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)
    print("Accuracy:", round(accuracy, 2), "%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return accuracy


def main():
    # 1. Load dataset
    df = pd.read_csv(DATASET_FILE)
    print("Original shape:", df.shape)

    # 2. Clean data
    df = df.drop_duplicates()
    df = df.dropna()
    print("After cleaning:", df.shape)

    # 3. Detect risk score scale
    risk_score_scale = detect_risk_score_scale(df)

    if risk_score_scale == "0_to_1":
        high_risk_cutoff = 0.7
    else:
        high_risk_cutoff = 70.0

    print("Detected Risk Score scale:", risk_score_scale)
    print("High Risk cutoff:", high_risk_cutoff)

    # 4. Feature engineering
    df["Amount_Balance_Ratio"] = df["Transaction_Amount"] / (df["Account_Balance"] + 1)
    df["High_Amount"] = (df["Transaction_Amount"] > 10000).astype(int)
    df["Low_Balance"] = (df["Account_Balance"] < 5000).astype(int)
    df["High_Risk"] = (df["Risk_Score"] > high_risk_cutoff).astype(int)
    df["Amount_Log"] = np.log1p(df["Transaction_Amount"])
    df["Distance_Risk"] = df["Transaction_Distance"] * df["Risk_Score"]

    # 5. Features and target
    feature_columns = [
        "Transaction_Amount",
        "Transaction_Type",
        "Account_Balance",
        "Location",
        "Previous_Fraudulent_Activity",
        "Daily_Transaction_Count",
        "Card_Type",
        "Transaction_Distance",
        "Authentication_Method",
        "Risk_Score",
        "Amount_Balance_Ratio",
        "High_Amount",
        "Low_Balance",
        "High_Risk",
        "Amount_Log",
        "Distance_Risk"
    ]

    target_column = "Fraud_Label"

    X = df[feature_columns]
    y = df[target_column]

    # 6. Numeric and categorical features
    numeric_features = [
        "Transaction_Amount",
        "Account_Balance",
        "Previous_Fraudulent_Activity",
        "Daily_Transaction_Count",
        "Transaction_Distance",
        "Risk_Score",
        "Amount_Balance_Ratio",
        "High_Amount",
        "Low_Balance",
        "High_Risk",
        "Amount_Log",
        "Distance_Risk"
    ]

    categorical_features = [
        "Transaction_Type",
        "Location",
        "Card_Type",
        "Authentication_Method"
    ]

    # 7. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 8. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # 9. Handle imbalance
    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(sampling_strategy=0.9, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # 10. Logistic Regression
    logistic_model = LogisticRegression(
        max_iter=8000,
        solver="liblinear",
        C=2.0,
        penalty="l2",
        random_state=42
    )

    logistic_model.fit(X_train_resampled, y_train_resampled)

    logistic_probs = logistic_model.predict_proba(X_test_processed)[:, 1]
    logistic_threshold = 0.65
    logistic_preds = (logistic_probs >= logistic_threshold).astype(int)

    logistic_accuracy = print_results(
        "Logistic Regression Results",
        y_test,
        logistic_preds
    )

    # 11. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    rf_model.fit(X_train_resampled, y_train_resampled)

    rf_probs = rf_model.predict_proba(X_test_processed)[:, 1]
    rf_threshold = 0.50
    rf_preds = (rf_probs >= rf_threshold).astype(int)

    rf_accuracy = print_results(
        "Random Forest Results",
        y_test,
        rf_preds
    )

    # 12. Save better model
    if rf_accuracy >= logistic_accuracy:
        best_model = rf_model
        saved_model_name = "Random Forest"
        prediction_threshold = rf_threshold
        best_accuracy = rf_accuracy
    else:
        best_model = logistic_model
        saved_model_name = "Logistic Regression"
        prediction_threshold = logistic_threshold
        best_accuracy = logistic_accuracy

    with open(PREPROCESSOR_FILE, "wb") as f:
        pickle.dump(preprocessor, f)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(best_model, f)

    config = {
        "risk_score_scale": risk_score_scale,
        "high_risk_cutoff": high_risk_cutoff,
        "prediction_threshold": prediction_threshold,
        "saved_model_name": saved_model_name,
        "logistic_accuracy_percent": round(logistic_accuracy, 2),
        "random_forest_accuracy_percent": round(rf_accuracy, 2),
        "best_accuracy_percent": round(best_accuracy, 2),
        "feature_columns": feature_columns
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("\n" + "=" * 50)
    print("FINAL SAVED MODEL")
    print("=" * 50)
    print("Saved Model:", saved_model_name)
    print("Best Accuracy:", round(best_accuracy, 2), "%")
    print("Files saved successfully ✅")


if __name__ == "__main__":
    main()