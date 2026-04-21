"""
Flask API - IDS Random Forest Streaming Dashboard
Cung cap luong log phat hien xam nhap theo thoi gian thuc tu du lieu kiem tra NSL-KDD.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask import Flask, jsonify, send_from_directory

# Thu muc goc cua file app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Duong dan den file model da huan luyen
MODEL_PATH = os.path.join(BASE_DIR, "ids_model.pkl")
# Duong dan den file du lieu kiem tra
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "NSL_ppTest_normal.csv")
# Duong dan den file du lieu huan luyen (dung de fit LabelEncoder)
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "NSL_boosted-1.csv")

# Tao ung dung Flask voi thu muc template la "templates"
app = Flask(__name__, template_folder="templates")

# ── Bien toan cuc cho model va bo dem du lieu ──────────────────────────────────
# Luu model Random Forest da huan luyen
model = None
# DataFrame chua du lieu kiem tra da duoc xu ly
df = None
# Danh sach 41 ten cot feature ma model yeu cau
feature_cols = None
# Ten cot nhan (label) trong DataFrame
label_col = None
# Dictionary chua cac LabelEncoder cho cac cot phan loai (categorical)
# Cac cot nay phai cung cach xu ly nhu khi huan luyen model
label_encoders = {}

def load_resources():
    """
    Tai model va du lieu, ma hoa cac cot phan loai dung cach nhu trong notebook.

    Model duoc huan luyen voi cac buoc:
      1. Xoa ['difficulty_level', 'atakcat'] khoi du lieu huan luyen
      2. Doi ten 'label' -> 'class'
      3. LabelEncode ['protocol_type', 'service', 'flag'] tren du lieu huan luyen
      4. X_train = tat ca cac cot tru 'class' (41 cot, theo dung thu tu cot goc)
    """
    global model, df, feature_cols, label_col, label_encoders

    # Tai model tu file pkl
    print("[*] Loading model from:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    # Lay ten cac cot tu model lam chuan, dam bao dung thu tu
    feature_cols = list(model.feature_names_in_)
    print(f"[+] Model expects {len(feature_cols)} features: {feature_cols}")

    # Tai du lieu huan luyen de lay cac lop cua LabelEncoder va nhan ground-truth
    print("[*] Loading training data:", TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    # Loai bo cac cot khong can thiet
    train_df = train_df.drop(columns=["difficulty_level", "atakcat"], errors="ignore")
    # Doi ten cot label thanh class
    if "label" in train_df.columns:
        train_df = train_df.rename(columns={"label": "class"})
    label_col = "class"  # xac nhan tu notebook

    # Chuyen nhan tu string sang so (0=normal, 1=attack) de hien thi
    train_df[label_col] = train_df[label_col].apply(
        lambda x: 0 if x == "normal" else 1
    )

    # Fit LabelEncoder tren cac cot phan loai cua du lieu huan luyen
    # Cac cot nay la: protocol_type, service, flag
    categorical_cols = ["protocol_type", "service", "flag"]
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        label_encoders[col] = le

    # Hien thi cac lop cua LabelEncoder de kiem tra
    print(f"[+] LabelEncoder classes:")
    for col in categorical_cols:
        classes = list(label_encoders[col].classes_)
        print(f"    {col}: {classes}")

    # Tai du lieu kiem tra va ap dung cung cac phep bien doi
    print("[*] Loading test data:", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["atakcat"], errors="ignore")
    if "label" in df.columns:
        df = df.rename(columns={"label": "class"})

    # Ma hoa cac cot phan loai (fit tren train, transform tren test)
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        # Gan gia tri "<unknown>" cho nhung gia tri khong co trong tap lop
        # De dam bao transform khong bi loi
        df[col] = df[col].where(
            df[col].isin(label_encoders[col].classes_),
            "<unknown>"
        )
        if "<unknown>" not in label_encoders[col].classes_:
            label_encoders[col].classes_ = np.append(
                label_encoders[col].classes_, "<unknown>"
            )
        df[col] = label_encoders[col].transform(df[col])

    # Chuyen nhan ground-truth sang 0/1
    # Ho tro ca dtype object (pandas cu) va string (pandas 2+/3.x)
    if df[label_col].dtype != np.int64 and df[label_col].dtype != np.int32:
        df[label_col] = df[label_col].apply(
            lambda x: 0 if x == "normal" else 1
        )

    # Chi giu lai 41 cot feature (cung thu tu nhu model yeu cau)
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

    # Lay tat ca 41 cot feature, da duoc LabelEncoded boi load_resources()
    X = row[feature_cols].values.reshape(1, -1).astype(float)

    # Chay du doan bang model
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][prediction])

    # Giai ma protocol tra ve chuoi doc duoc (nguoc tu so sang ten)
    protocol_encoded = int(row["protocol_type"])
    protocol = label_encoders["protocol_type"].inverse_transform([protocol_encoded])[0]

    # Lay cac gia tri bytes
    src_bytes = int(row.get("src_bytes", 0))
    dst_bytes = int(row.get("dst_bytes", 0))
    # Tao ID goi tin theo dinh dang PKT-XXXXXX
    packet_id = f"PKT-{int(row.name):06d}"

    # Tra ve dictionary chua thong tin log
    return {
        "id": packet_id,
        "protocol": protocol,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "prediction": prediction,
        "confidence": round(probability, 4),
        "true_label": true_label,
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
    Tra ve so luong mau normal va attack trong tap du lieu (dung tham khao).
    """
    try:
        label_counts = df[label_col].value_counts().to_dict()
        return jsonify({
            "total": int(len(df)),
            "normal": int(label_counts.get(0, 0)),
            "attack": int(label_counts.get(1, 0)),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Khoi dong Flask server khi chay truc tiep file nay
if __name__ == "__main__":
    # Tai model va du lieu truoc khi server bat dau
    load_resources()
    # In thong bao khoi dong server
    print("[*] Starting IDS Flask server on http://127.0.0.1:5000")
    # Chay server Flask tren port 5000, bat debug de tu dong tai lai khi co thay doi
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
