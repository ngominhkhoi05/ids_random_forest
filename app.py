"""
Flask API - IDS Random Forest Streaming Dashboard
Provides real-time intrusion detection log streaming from NSL-KDD test data.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask import Flask, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ids_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "NSL_ppTest.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "NSL_boosted-1.csv")

app = Flask(__name__, template_folder="templates")

# ── Global model + dataset cache ────────────────────────────────────────────
model = None
df = None
feature_cols = None
label_col = None
# LabelEncoders for categorical columns (must match how the model was trained)
label_encoders = {}

def load_resources():
    """
    Load model + dataset and encode categorical columns exactly as in the notebook.

    Model was trained with:
      1. Drop ['difficulty_level', 'atakcat'] from training data
      2. Rename 'label' -> 'class'
      3. LabelEncode ['protocol_type', 'service', 'flag'] on training data
      4. X_train = all columns except 'class'  (41 columns, in original column order)
    """
    global model, df, feature_cols, label_col, label_encoders

    print("[*] Loading model from:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    # Use model's known feature names as source of truth for column order
    feature_cols = list(model.feature_names_in_)
    print(f"[+] Model expects {len(feature_cols)} features: {feature_cols}")

    # Load training data to derive LabelEncoder classes and ground-truth labels
    print("[*] Loading training data:", TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = train_df.drop(columns=["difficulty_level", "atakcat"], errors="ignore")
    if "label" in train_df.columns:
        train_df = train_df.rename(columns={"label": "class"})
    label_col = "class"  # confirmed by notebook

    # Convert string labels to numeric (0=normal, 1=attack) for display
    train_df[label_col] = train_df[label_col].apply(
        lambda x: 0 if x == "normal" else 1
    )

    # Fit LabelEncoders on training categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        label_encoders[col] = le

    print(f"[+] LabelEncoder classes:")
    for col in categorical_cols:
        classes = list(label_encoders[col].classes_)
        print(f"    {col}: {classes}")

    # Load test data and apply same transformations
    print("[*] Loading test data:", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["atakcat"], errors="ignore")
    if "label" in df.columns:
        df = df.rename(columns={"label": "class"})

    # Encode categorical columns (fit on train, transform on test)
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        # Map unknown values to "<unknown>" so transform never crashes
        df[col] = df[col].where(
            df[col].isin(label_encoders[col].classes_),
            "<unknown>"
        )
        if "<unknown>" not in label_encoders[col].classes_:
            label_encoders[col].classes_ = np.append(
                label_encoders[col].classes_, "<unknown>"
            )
        df[col] = label_encoders[col].transform(df[col])

    # Convert ground-truth label to 0/1
    # Support both object (older pandas) and string (pandas 2+/3.x) dtypes
    if df[label_col].dtype != np.int64 and df[label_col].dtype != np.int32:
        df[label_col] = df[label_col].apply(
            lambda x: 0 if x == "normal" else 1
        )

    # Keep only the 41 feature columns (same order as model expects)
    df = df[feature_cols + [label_col]]

    print(f"[+] Dataset ready: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[+] Features: {feature_cols}")
    print(f"[+] Label: {label_col}")


def get_random_log_row():
    """Pick a random data row, extract features, run model prediction."""
    global feature_cols, label_col

    row = df.sample(n=1).iloc[0]

    # Ground-truth label (for display only — not fed to model)
    true_label = int(row[label_col])

    # All 41 feature columns, already LabelEncoded by load_resources()
    X = row[feature_cols].values.reshape(1, -1).astype(float)

    # Run model prediction
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][prediction])

    # Reverse-encode protocol back to human-readable string
    protocol_encoded = int(row["protocol_type"])
    protocol = label_encoders["protocol_type"].inverse_transform([protocol_encoded])[0]

    src_bytes = int(row.get("src_bytes", 0))
    dst_bytes = int(row.get("dst_bytes", 0))
    packet_id = f"PKT-{int(row.name):06d}"

    return {
        "id": packet_id,
        "protocol": protocol,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "prediction": prediction,
        "confidence": round(probability, 4),
        "true_label": true_label,
    }


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/stream_logs", methods=["GET"])
def stream_logs():
    """Return a single random log entry with model prediction."""
    try:
        entry = get_random_log_row()
        return jsonify(entry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return counts of normal vs attack samples in the dataset (for reference)."""
    try:
        label_counts = df[label_col].value_counts().to_dict()
        return jsonify({
            "total": int(len(df)),
            "normal": int(label_counts.get(0, 0)),
            "attack": int(label_counts.get(1, 0)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_resources()
    print("[*] Starting IDS Flask server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
