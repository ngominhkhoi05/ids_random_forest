"""
test.py - Kiem tra mo hinh IDS Random Forest tren file test tuong thich voi app.py
Su dung: python test.py dataset/NSL_ppTest_normal.csv
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ids_model.pkl")
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "NSL_boosted-0.csv")


def load_resources(test_path):
    model = joblib.load(MODEL_PATH)
    feature_cols = list(model.feature_names_in_)

    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    train_df = train_df.drop(columns=["difficulty_level", "atakcat"], errors="ignore")
    if "label" in train_df.columns:
        train_df = train_df.rename(columns={"label": "class"})
    label_col = "class"
    train_df[label_col] = train_df[label_col].apply(lambda x: 0 if str(x) == "normal" else 1)

    categorical_cols = ["protocol_type", "service", "flag"]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        label_encoders[col] = le

    print("[*] Loading test data:", test_path)
    df = pd.read_csv(test_path, low_memory=False)
    print(f"    Total rows: {len(df):,}")

    rows_before = len(df)
    df = df.dropna(subset=["label"])
    rows_after = len(df)
    print(f"    Rows dropped (NaN label): {rows_before - rows_after}")

    df = df.fillna(0)
    df = df.drop(columns=["atakcat"], errors="ignore")
    if "label" in df.columns:
        df = df.rename(columns={"label": "class"})

    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].where(df[col].isin(label_encoders[col].classes_), "<unknown>")
        if "<unknown>" not in label_encoders[col].classes_:
            label_encoders[col].classes_ = np.append(label_encoders[col].classes_, "<unknown>")
        df[col] = label_encoders[col].transform(df[col])

    df[label_col] = df[label_col].apply(lambda x: 0 if str(x).strip() == "normal" else 1)

    df = df[[c for c in feature_cols if c in df.columns] + [label_col]]

    return model, df, feature_cols, label_col


def run_evaluation(model, df, feature_cols, label_col):
    X = df[feature_cols].values.astype(float)
    y_true = df[label_col].values
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    correct_normal = np.sum((y_pred == 0) & (y_true == 0))
    wrong_normal = np.sum((y_pred == 1) & (y_true == 0))
    correct_attack = np.sum((y_pred == 1) & (y_true == 1))
    wrong_attack = np.sum((y_pred == 0) & (y_true == 1))

    total = len(df)
    total_correct = int((y_pred == y_true).sum())
    total_wrong = int(total - total_correct)
    accuracy = total_correct / total * 100

    print("\n" + "=" * 52)
    print("          KET QUA DANH GIA MO HINH")
    print("=" * 52)
    print(f"  Tong so mau:             {total:,}")
    print(f"  Mau normal (ground-t): {(y_true == 0).sum():,}")
    print(f"  Mau attack (ground-t): {(y_true == 1).sum():,}")
    print()
    print(f"  Dung:                   {total_correct:,}  ({accuracy:.2f}%)")
    print(f"  Sai:                    {total_wrong:,}  ({100-accuracy:.2f}%)")
    print()
    print("  --- Chi tiet phan loai ---")
    print(f"  Normal  -> Normal (dung):      {correct_normal:,}  (true positive rate: {correct_normal/(y_true==0).sum()*100:.2f}%)")
    print(f"  Normal  -> Attack (sai):       {wrong_normal:,}  (false positive: {wrong_normal/(y_true==0).sum()*100:.2f}%)")
    print(f"  Attack  -> Attack (dung):      {correct_attack:,}  (miss: {wrong_attack/(y_true==1).sum()*100:.2f}%)")
    print(f"  Attack  -> Normal (lo bo):    {wrong_attack:,}  (false negative: {wrong_attack/(y_true==1).sum()*100:.2f}%)")
    print("=" * 52)

    if total_wrong > 0:
        print(f"\n  --- Vi du 5 dong bi du doan sai ---")
        wrong_idx = np.where(y_pred != y_true)[0][:5]
        for i, idx in enumerate(wrong_idx, 1):
            true_str = "NORMAL" if y_true[idx] == 0 else "ATTACK"
            pred_str = "NORMAL" if y_pred[idx] == 0 else "ATTACK"
            conf = float(y_proba[idx][y_pred[idx]])
            print(f"  {i}. Dong {idx}: true={true_str}, pred={pred_str} (conf={conf:.1%})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <path_to_test_csv>")
        sys.exit(1)

    test_path = sys.argv[1]
    if not os.path.exists(test_path):
        print(f"[ERROR] File not found: {test_path}")
        sys.exit(1)

    model, df, feature_cols, label_col = load_resources(test_path)
    run_evaluation(model, df, feature_cols, label_col)


if __name__ == "__main__":
    main()
