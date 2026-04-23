"""
Flask API - IDS Random Forest Streaming Dashboard
Cung cap luong log phat hien xam nhap theo thoi gian thuc tu du lieu kiem tra NSL-KDD.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, jsonify, send_from_directory

# Thu muc goc cua file app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Duong dan den file model da huan luyen
MODEL_PATH = os.path.join(BASE_DIR, "ids_model.pkl")
# Duong dan den file du lieu kiem tra
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "NSL_ppTest.csv")
# Duong dan den file du lieu huan luyen (dung de fit OneHotEncoder)
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "NSL_boosted-0.csv")

# Tao ung dung Flask voi thu muc template la "templates"
app = Flask(__name__, template_folder="templates")

# ── Bien toan cuc cho model va bo dem du lieu ──────────────────────────────────
# Luu model Random Forest da huan luyen
model = None
# DataFrame chua du lieu kiem tra da duoc xu ly
df = None
# Danh sach ten cot feature ma model yeu cau (so cot tuy thuoc vao OHE)
feature_cols = None
# Ten cot nhan (label) trong DataFrame
label_col = None
# Dictionary chua OneHotEncoder cho cac cot phan loai (categorical)
label_encoders = {}
# ── Bo dem thong ke real-time ──────────────────────────────────────────────────
stats_lock = __import__("threading").Lock()
running_stats = {
    "total": 0,
    "correct": 0,
    "wrong": 0,
    "safe": 0,
    "blocked": 0,
}

def load_resources():
    """
    Tai model va du lieu, ma hoa cac cot phan loai bang One-Hot Encoding.

    Model duoc huan luyen voi cac buoc:
      1. Xoa ['difficulty_level', 'atakcat'] khoi du lieu huan luyen
      2. Doi ten 'label' -> 'class'
      3. OneHotEncode ['protocol_type', 'service', 'flag'] tren du lieu huan luyen
      4. X_train = tat ca cac cot tru 'class' (nhieu hon 41 cot do OHE)
    """
    global model, df, feature_cols, label_col, label_encoders

    # Tai model tu file pkl
    print("[*] Loading model from:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    # Tai du lieu huan luyen de lay cac lop cua OneHotEncoder va nhan ground-truth
    print("[*] Loading training data:", TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = train_df.drop(columns=["difficulty_level", "atakcat"], errors="ignore")
    if "label" in train_df.columns:
        train_df = train_df.rename(columns={"label": "class"})
    label_col = "class"

    # Chuyen nhan tu string sang so (0=normal, 1=attack)
    train_df[label_col] = train_df[label_col].apply(
        lambda x: 0 if x == "normal" else 1
    )

    # Fit OneHotEncoder tren cac cot phan loai cua du lieu huan luyen
    categorical_cols = ["protocol_type", "service", "flag"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(train_df[categorical_cols].astype(str))
    label_encoders["ohe"] = ohe

    # Hien thi so luong cot sau OHE
    ohe_cols = ohe.get_feature_names_out(categorical_cols)
    print(f"[+] OneHotEncoder fitted on {len(categorical_cols)} categorical columns")
    print(f"    -> {len(ohe_cols)} one-hot columns created")

    # Transform train data
    train_encoded = ohe.transform(train_df[categorical_cols].astype(str))
    train_ohe_df = pd.DataFrame(train_encoded, columns=ohe_cols, index=train_df.index)
    train_df = pd.concat([train_df.drop(columns=categorical_cols), train_ohe_df], axis=1)

    # Tai du lieu kiem tra va ap dung cung cac phep bien doi
    print("[*] Loading test data:", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["difficulty_level", "atakcat"], errors="ignore")
    if "label" in df.columns:
        df = df.rename(columns={"label": "class"})

    # Transform test data voi OHE (da fit tren train)
    test_encoded = ohe.transform(df[categorical_cols].astype(str))
    test_ohe_df = pd.DataFrame(test_encoded, columns=ohe_cols, index=df.index)
    df = pd.concat([df.drop(columns=categorical_cols), test_ohe_df], axis=1)

    # Chuyen nhan ground-truth sang 0/1
    if df[label_col].dtype != np.int64 and df[label_col].dtype != np.int32:
        df[label_col] = df[label_col].apply(
            lambda x: 0 if x == "normal" else 1
        )

    # Lay feature_cols tu model (dam bao dung thu tu)
    feature_cols = list(model.feature_names_in_)
    # Chi giu lai cac cot feature (cung thu tu nhu model yeu cau) + label
    df = df[feature_cols + [label_col]]

    print(f"[+] Dataset ready: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[+] Features: {feature_cols}")
    print(f"[+] Label: {label_col}")
def get_random_log_row():
    """
    Chon ngau nhien mot dong du lieu, trich xuat features, chay du doan model.
    """
    global feature_cols, label_col

    # Chon ngau nhien 1 dong tu DataFrame
    row = df.sample(n=1).iloc[0]

    # Lay nhan ground-truth (chi de hien thi, khong dua vao model)
    true_label = int(row[label_col])

    # Lay tat ca cac cot feature, da duoc OHE boi load_resources()
    X = row[feature_cols].values.reshape(1, -1).astype(float)

    # Chay du doan bang model
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][prediction])

    # Lay gia tri protocol tu cot OHE tuong ung (nguoc tu one-hot sang ten)
    protocol_col = [c for c in row.index if c.startswith("protocol_type_") and row[c] == 1]
    protocol = protocol_col[0].replace("protocol_type_", "") if protocol_col else "unknown"

    # Lay cac gia tri bytes
    src_bytes = int(row.get("src_bytes", 0))
    dst_bytes = int(row.get("dst_bytes", 0))
    # Tao ID goi tin theo dinh dang PKT-XXXXXX
    packet_id = f"PKT-{int(row.name):06d}"

    # Tra ve dictionary chua thong tin log
    # Cap nhat bo dem thong ke real-time
    is_correct = int(prediction == true_label)
    with stats_lock:
        running_stats["total"] += 1
        running_stats["correct"] += is_correct
        running_stats["wrong"] += 1 - is_correct
        if prediction == 0:
            running_stats["safe"] += 1
        else:
            running_stats["blocked"] += 1

    return {
        "id": packet_id,
        "protocol": protocol,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "prediction": prediction,
        "confidence": round(probability, 4),
        "true_label": true_label,
        "is_correct": is_correct,
    }


# Route trang chu - tra ve file HTML dashboard
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


# Route API tra ve mot log ngau nhien cung voi du doan cua model
@app.route("/api/stream_logs", methods=["GET"])
def stream_logs():
    """
    Tra ve mot entry log ngau nhien cung voi du doan cua model.
    """
    try:
        entry = get_random_log_row()
        return jsonify(entry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route API tra ve thong ke so luong mau normal va attack trong tap du lieu
@app.route("/api/stats", methods=["GET"])
def stats():
    """
    Tra ve thong ke real-time: tong so, so dung, so sai, accuracy cua model.
    """
    try:
        label_counts = df[label_col].value_counts().to_dict()
        with stats_lock:
            s = dict(running_stats)
        total = s["total"]
        accuracy = round(s["correct"] / total * 100, 2) if total > 0 else 0.0
        return jsonify({
            "total": int(len(df)),
            "normal": int(label_counts.get(0, 0)),
            "attack": int(label_counts.get(1, 0)),
            # Real-time session stats
            "session_scanned": s["total"],
            "session_correct": s["correct"],
            "session_wrong": s["wrong"],
            "session_safe": s["safe"],
            "session_blocked": s["blocked"],
            "session_accuracy": accuracy,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route API reset bo dem thong ke
@app.route("/api/reset_stats", methods=["POST"])
def reset_stats():
    """Reset bo dem thong ke real-time ve 0."""
    global running_stats
    with stats_lock:
        running_stats = {
            "total": 0,
            "correct": 0,
            "wrong": 0,
            "safe": 0,
            "blocked": 0,
        }
    return jsonify({"status": "reset", "stats": running_stats})


# Khoi dong Flask server khi chay truc tiep file nay
if __name__ == "__main__":
    # Tai model va du lieu truoc khi server bat dau
    load_resources()
    # In thong bao khoi dong server
    print("[*] Starting IDS Flask server on http://127.0.0.1:5000")
    # Chay server Flask tren port 5000, bat debug de tu dong tai lai khi co thay doi
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
