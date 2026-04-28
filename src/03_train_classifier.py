#!/usr/bin/env python3
"""
寶山第二水庫 缺水預測分類模型
任務：預測「30天內是否會觸發缺水危機」
閥值: effective_storage < 1,200 萬噸 → 缺水=1
"""

import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

# =====================================================================
# 1. 參數設定
# =====================================================================
DATA_PATH  = "/home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai/data/寶二訓練集_v1.csv"
OUTPUT_DIR = "/home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DROUGHT_THRESHOLD = 1200   # 萬噸（觸發危機的閥值）
HORIZON_DAYS      = 30     # 向前看幾天
SEQ_LEN           = 30     # 輸入序列長度
TRAIN_END         = "2021-12-31"
TEST_START        = "2022-01-01"
EPOCHS            = 200
BATCH_SIZE        = 32
LR                = 1e-3

# =====================================================================
# 2. 載入資料
# =====================================================================
df = pd.read_csv(DATA_PATH)
df["data_date"] = pd.to_datetime(df["data_date"])

drop_cols = ["inflow_cms", "outflow_cms", "is_imputed_rainfall_self",
             "is_imputed_inflow", "is_imputed_outflow", "is_imputed_storage"]
feature_cols = [c for c in df.columns if c not in drop_cols + ["data_date"]]
target_col   = "effective_storage"

df = df.dropna(subset=feature_cols + [target_col])
print(f"使用特徵 ({len(feature_cols)}): {feature_cols}")
print(f"目標: 30天內是否缺水（< {DROUGHT_THRESHOLD} 萬噸）")

# =====================================================================
# 3. 建立分類標籤
# =====================================================================
# 向前看 HORIZON_DAYS 天內是否曾低於閥值
storage = df[target_col].values
labels  = np.zeros(len(df), dtype=int)

for i in range(len(df) - HORIZON_DAYS):
    future_window = storage[i+1 : i+1+HORIZON_DAYS]
    if (future_window < DROUGHT_THRESHOLD).any():
        labels[i] = 1

df["label"] = labels

print(f"\n標籤分佈（整筆資料）:")
print(df["label"].value_counts())
print(f"缺水比例: {df['label'].mean()*100:.1f}%")

# 訓練/測試集（與回歸模型一致）
train_df = df[df["data_date"] <= TRAIN_END].copy()
test_df  = df[df["data_date"] >= TEST_START].copy()

print(f"\n訓練集: {train_df['data_date'].min().date()} ~ {train_df['data_date'].max().date()}, {len(train_df)} 筆")
print(f"測試集: {test_df['data_date'].min().date()} ~ {test_df['data_date'].max().date()}, {len(test_df)} 筆")
print(f"訓練集缺水比例: {train_df['label'].mean()*100:.1f}%")
print(f"測試集缺水比例:  {test_df['label'].mean()*100:.1f}%")

# =====================================================================
# 4. 正規化
# =====================================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X = scaler_X.fit_transform(train_df[feature_cols].values)
train_y_raw = train_df[[target_col]].values  # 拿來算 scaler_y
scaler_y.fit(train_y_raw)

test_X = scaler_X.transform(test_df[feature_cols].values)

# =====================================================================
# 5. 建立序列
# =====================================================================
def create_sequences(X, y_label, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y_label[i + seq_len])
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_X, train_df["label"].values, SEQ_LEN)
X_test,  y_test  = create_sequences(test_X,  test_df["label"].values,  SEQ_LEN)

print(f"\n訓練序列: X_train={X_train.shape}, y_train={y_train.shape}, 缺水={y_train.sum()}")
print(f"測試序列: X_test={X_test.shape},  y_test={y_test.sum()}")

# =====================================================================
# 6. 建立分類模型（Bidirectional LSTM）
# =====================================================================
model = Sequential([
    Bidirectional(LSTM(64, activation="tanh", return_sequences=True),
                  input_shape=(SEQ_LEN, X_train.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(32, activation="tanh")),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auc")]
)
model.summary()

# =====================================================================
# 7. Callbacks
# =====================================================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
]

# =====================================================================
# 8. 訓練
# =====================================================================
print("\n開始訓練...")
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
    class_weight={0: 1.0, 1: (y_train==0).sum() / max((y_train==1).sum(), 1)}  # 對齊類別不平衡
)

# =====================================================================
# 9. 預測與評估
# =====================================================================
y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

# 基本指標
acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred, zero_division=0)
rec   = recall_score(y_test, y_pred, zero_division=0)
f1    = f1_score(y_test, y_pred, zero_division=0)
auc   = roc_auc_score(y_test, y_prob)

print(f"\n========== 測試集分類結果 ==========")
print(f"Accuracy : {acc:.4f} ({accuracy_score(y_test, y_pred)*100:.1f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}")
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print("\n", classification_report(y_test, y_pred, target_names=["正常","缺水"]))

# =====================================================================
# 10. 繪圖
# =====================================================================
# --- 10a. 訓練曲線 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train Acc")
axes[1].plot(history.history["val_accuracy"], label="Val Acc")
axes[1].plot(history.history["val_auc"], label="Val AUC")
axes[1].set_title("Accuracy & AUC")
axes[1].legend()
fig.suptitle("Classification LSTM Training", fontsize=14)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/classifier_training.png", dpi=150)
print(f"訓練曲線已存: {OUTPUT_DIR}/classifier_training.png")

# --- 10b. Confusion Matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["正常","缺水"]); ax.set_yticklabels(["正常","缺水"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

# 填入數字
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=20)
plt.colorbar(im)
fig.savefig(f"{OUTPUT_DIR}/classifier_cm.png", dpi=150)
print(f"Confusion Matrix已存: {OUTPUT_DIR}/classifier_cm.png")

# --- 10c. ROC Curve ---
fpr_, tpr_, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_, tpr_, label=f"LSTM (AUC={auc:.3f})", linewidth=2)
ax.plot([0,1],[0,1], "--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — 30天內缺水預測")
ax.legend()
fig.savefig(f"{OUTPUT_DIR}/classifier_roc.png", dpi=150)
print(f"ROC Curve已存: {OUTPUT_DIR}/classifier_roc.png")

# --- 10d. 預測時序圖（拿掉前30筆因為沒有label）---
test_dates = test_df["data_date"].values[SEQ_LEN:]
pred_df = pd.DataFrame({"date": test_dates, "actual": y_test, "prob": y_prob, "pred": y_pred})

fig, ax = plt.subplots(figsize=(16, 5))
ax2 = ax.twinx()
ax.plot(pred_df["date"], pred_df["actual"], "b-", alpha=0.6, label="實際是否缺水", linewidth=1)
ax2.plot(pred_df["date"], pred_df["prob"], "r-", alpha=0.7, label="缺水機率", linewidth=1)
ax.axhline(0, color="b", linewidth=0.5)
ax2.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="決策閾值0.5")
ax.set_ylabel("實際缺水（0/1）"); ax2.set_ylabel("缺水機率")
ax.set_xlabel("日期")
ax.set_title(f"30天內缺水預測（2022-2023）| AUC={auc:.3f}")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc="upper right")
fig.savefig(f"{OUTPUT_DIR}/classifier_prediction_timeline.png", dpi=150)
print(f"預測時序圖已存: {OUTPUT_DIR}/classifier_prediction_timeline.png")

# --- 10e. 閥值敏感度分析 ---
print("\n========== 閾值敏感度分析 ==========")
print(f"{'閾值':>10} {'精確度':>8} {'召回率':>8} {'F1':>8} {'AUC':>8}")
print("-" * 46)
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    yp = (y_prob >= thresh).astype(int)
    p  = precision_score(y_test, yp, zero_division=0)
    r  = recall_score(y_test, yp, zero_division=0)
    f  = f1_score(y_test, yp, zero_division=0)
    print(f"{thresh:>10.1f} {p:>8.3f} {r:>8.3f} {f:>8.3f} {auc:>8.3f}")

# =====================================================================
# 11. 儲存
# =====================================================================
model.save(f"{OUTPUT_DIR}/bao2_classifier.keras")
np.save(f"{OUTPUT_DIR}/classifier_scaler_X_scale.npy", scaler_X.scale_)
np.save(f"{OUTPUT_DIR}/classifier_scaler_X_min.npy",   scaler_X.min_)
np.save(f"{OUTPUT_DIR}/classifier_feature_cols.npy",   np.array(feature_cols))

# 儲存測試預測結果
pred_df.to_csv(f"{OUTPUT_DIR}/classifier_daily_predictions.csv", index=False)
print(f"\n每日預測已存: {OUTPUT_DIR}/classifier_daily_predictions.csv")
print(f"模型已存: {OUTPUT_DIR}/bao2_classifier.keras")
print("\n========== 分類模型訓練完成 ==========")
print(f"最佳 Val Loss: {min(history.history['val_loss']):.6f}")
print(f"Val AUC: {max(history.history['val_auc']):.4f}")
