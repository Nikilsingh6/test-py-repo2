import json
import math
import os
import pickle
import sqlite3
from datetime import datetime

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = "atm_fraud_secret_key_123"

DATABASE = "users.db"
MODEL_FILE = "model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"
CONFIG_FILE = "model_config.json"

VALID_TRANSACTION_TYPES = ["POS", "Online", "ATM Withdrawal", "Bank Transfer"]
VALID_LOCATIONS = ["Tokyo", "Mumbai", "London", "Sydney", "New York"]
VALID_CARD_TYPES = ["Mastercard", "Visa", "Amex", "Discover"]
VALID_AUTH_METHODS = ["Biometric", "PIN", "Password", "OTP"]


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            transaction_amount REAL,
            transaction_type TEXT,
            account_balance REAL,
            location TEXT,
            previous_fraudulent_activity INTEGER,
            daily_transaction_count INTEGER,
            card_type TEXT,
            transaction_distance REAL,
            authentication_method TEXT,
            risk_score REAL,
            prediction_label TEXT,
            fraud_probability REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def load_pickle_file(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def load_json_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


preprocessor = load_pickle_file(PREPROCESSOR_FILE)
model = load_pickle_file(MODEL_FILE)
model_config = load_json_file(CONFIG_FILE)


def is_logged_in():
    return "user_id" in session


def current_user():
    if not is_logged_in():
        return None

    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (session["user_id"],)
    ).fetchone()
    conn.close()
    return user


def get_model_meta():
    if not model_config:
        return {
            "risk_score_max": 1,
            "risk_score_placeholder": "Enter risk score between 0 and 1",
            "high_risk_cutoff": 0.7,
            "prediction_threshold": 0.75,
        }

    risk_scale = model_config.get("risk_score_scale", "0_to_1")

    if risk_scale == "0_to_100":
        risk_score_max = 100
        risk_score_placeholder = "Enter risk score between 0 and 100"
    else:
        risk_score_max = 1
        risk_score_placeholder = "Enter risk score between 0 and 1"

    return {
        "risk_score_max": risk_score_max,
        "risk_score_placeholder": risk_score_placeholder,
        "high_risk_cutoff": float(model_config.get("high_risk_cutoff", 0.7)),
        "prediction_threshold": float(model_config.get("prediction_threshold", 0.75)),
    }


@app.route("/")
def home():
    if is_logged_in():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not name or not email or not password:
            flash("Please fill all fields.", "danger")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("register.html")

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        existing_user = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if existing_user:
            conn.close()
            flash("Email already registered. Please sign in.", "warning")
            return redirect(url_for("login"))

        conn.execute("""
            INSERT INTO users (name, email, password, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            name,
            email,
            hashed_password,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()

        flash("Registration successful. Please sign in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "").strip()

        if not identifier or not password:
            flash("Please enter email/username and password.", "danger")
            return render_template("login.html")

        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE email = ? OR name = ?",
            (identifier.lower(), identifier)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            flash("Login successful.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email/username or password.", "danger")

    return render_template("login.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not is_logged_in():
        flash("Please sign in first.", "warning")
        return redirect(url_for("login"))

    model_meta = get_model_meta()

    if model is None or preprocessor is None or model_config is None:
        return render_template(
            "dashboard.html",
            user=current_user(),
            prediction=None,
            probability=None,
            history=[],
            model_missing=True,
            transaction_types=VALID_TRANSACTION_TYPES,
            locations=VALID_LOCATIONS,
            card_types=VALID_CARD_TYPES,
            auth_methods=VALID_AUTH_METHODS,
            risk_score_max=model_meta["risk_score_max"],
            risk_score_placeholder=model_meta["risk_score_placeholder"]
        )

    prediction = None
    probability = None

    if request.method == "POST":
        try:
            transaction_amount = float(request.form["transaction_amount"])
            transaction_type = request.form["transaction_type"]
            account_balance = float(request.form["account_balance"])
            location = request.form["location"]
            previous_fraudulent_activity = int(request.form["previous_fraudulent_activity"])
            daily_transaction_count = int(request.form["daily_transaction_count"])
            card_type = request.form["card_type"]
            transaction_distance = float(request.form["transaction_distance"])
            authentication_method = request.form["authentication_method"]
            risk_score = float(request.form["risk_score"])

            if transaction_amount < 0:
                raise ValueError("Transaction Amount cannot be negative")
            if account_balance < 0:
                raise ValueError("Account Balance cannot be negative")
            if daily_transaction_count < 0:
                raise ValueError("Daily Transaction Count cannot be negative")
            if transaction_distance < 0:
                raise ValueError("Transaction Distance cannot be negative")

            if transaction_type not in VALID_TRANSACTION_TYPES:
                raise ValueError("Invalid Transaction Type")
            if location not in VALID_LOCATIONS:
                raise ValueError("Invalid Location")
            if card_type not in VALID_CARD_TYPES:
                raise ValueError("Invalid Card Type")
            if authentication_method not in VALID_AUTH_METHODS:
                raise ValueError("Invalid Authentication Method")
            if previous_fraudulent_activity not in [0, 1]:
                raise ValueError("Previous Fraudulent Activity must be 0 or 1")

            if not (0 <= risk_score <= model_meta["risk_score_max"]):
                raise ValueError(
                    f"Risk Score must be between 0 and {model_meta['risk_score_max']}"
                )

            amount_balance_ratio = transaction_amount / (account_balance + 1)
            high_amount = int(transaction_amount > 10000)
            low_balance = int(account_balance < 5000)
            high_risk = int(risk_score > model_meta["high_risk_cutoff"])
            amount_log = math.log1p(transaction_amount)
            distance_risk = transaction_distance * risk_score

            input_df = pd.DataFrame({
                "Transaction_Amount": [transaction_amount],
                "Transaction_Type": [transaction_type],
                "Account_Balance": [account_balance],
                "Location": [location],
                "Previous_Fraudulent_Activity": [previous_fraudulent_activity],
                "Daily_Transaction_Count": [daily_transaction_count],
                "Card_Type": [card_type],
                "Transaction_Distance": [transaction_distance],
                "Authentication_Method": [authentication_method],
                "Risk_Score": [risk_score],
                "Amount_Balance_Ratio": [amount_balance_ratio],
                "High_Amount": [high_amount],
                "Low_Balance": [low_balance],
                "High_Risk": [high_risk],
                "Amount_Log": [amount_log],
                "Distance_Risk": [distance_risk]
            })

            transformed_input = preprocessor.transform(input_df)
            if hasattr(transformed_input, "toarray"):
                transformed_input = transformed_input.toarray()

            prob = float(model.predict_proba(transformed_input)[0][1])

            if prob >= model_meta["prediction_threshold"]:
                prediction = "Fraud"
            else:
                prediction = "Not Fraud"

            probability = round(prob * 100, 2)

            conn = get_db_connection()
            conn.execute("""
                INSERT INTO predictions (
                    user_id, transaction_amount, transaction_type, account_balance,
                    location, previous_fraudulent_activity, daily_transaction_count,
                    card_type, transaction_distance, authentication_method,
                    risk_score, prediction_label, fraud_probability, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"],
                transaction_amount,
                transaction_type,
                account_balance,
                location,
                previous_fraudulent_activity,
                daily_transaction_count,
                card_type,
                transaction_distance,
                authentication_method,
                risk_score,
                prediction,
                probability,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            conn.close()

            flash(
                f"Prediction: {prediction} | Fraud Probability: {probability}%",
                "prediction-fraud" if prediction == "Fraud" else "prediction-safe"
            )
            return redirect(url_for("dashboard"))

        except Exception as e:
            flash(f"Input error: {str(e)}", "danger")
            return redirect(url_for("dashboard"))

    conn = get_db_connection()
    history = conn.execute("""
        SELECT * FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 10
    """, (session["user_id"],)).fetchall()
    conn.close()

    return render_template(
        "dashboard.html",
        user=current_user(),
        prediction=prediction,
        probability=probability,
        history=history,
        model_missing=False,
        transaction_types=VALID_TRANSACTION_TYPES,
        locations=VALID_LOCATIONS,
        card_types=VALID_CARD_TYPES,
        auth_methods=VALID_AUTH_METHODS,
        risk_score_max=model_meta["risk_score_max"],
        risk_score_placeholder=model_meta["risk_score_placeholder"]
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)