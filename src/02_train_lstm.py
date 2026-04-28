#!/usr/bin/env python3
"""
寶山第二水庫 LSTM 蓄水量預測模型
Dataset: 寶二訓練集_v1.csv (2016-01-01 ~ 2023-12-31, 2922 days)
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# =====================================================================
# 1. 設定參數
# =====================================================================
DATA_PATH = "/home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai/data/寶二訓練集_v1.csv"
OUTPUT_DIR = "/home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 實驗參數
SEQ_LEN = 30          # 輸入序列長度（天）
TRAIN_END = "2021-12-31"  # 訓練集截止日（2022-2023 留作測試）
TEST_START = "2022-01-01"

EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3

# =====================================================================
# 2. 載入與篩選特徵（排除大量缺失的 inflow/outflow）
# =====================================================================
df = pd.read_csv(DATA_PATH)
df["data_date"] = pd.to_datetime(df["data_date"])

# 排除 inflow_cms / outflow_cms（有 97% 缺失）以及 is_* 標註欄位
drop_cols = ["inflow_cms", "outflow_cms", "is_imputed_rainfall_self",
             "is_imputed_inflow", "is_imputed_outflow", "is_imputed_storage"]
feature_cols = [c for c in df.columns if c not in drop_cols + ["data_date"]]
target_col = "effective_storage"

# 刪除有 NaN 的列（只有 basin_rainfall_self_mm 有缺失）
df = df.dropna(subset=feature_cols + [target_col])
print(f"刪除 NaN 後: {len(df)} 筆")

print(f"使用特徵 ({len(feature_cols)}): {feature_cols}")
print(f"目標變數: {target_col}")

# =====================================================================
# 3. 資料集切分
# =====================================================================
train_df = df[df["data_date"] <= TRAIN_END].copy()
test_df  = df[df["data_date"] >= TEST_START].copy()

print(f"\n訓練集: {train_df['data_date'].min().date()} ~ {train_df['data_date'].max().date()}, {len(train_df)} 筆")
print(f"測試集: {test_df['data_date'].min().date()} ~ {test_df['data_date'].max().date()}, {len(test_df)} 筆")

# =====================================================================
# 4. 正規化（只 fit 訓練集，避免 leakage）
# =====================================================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X = scaler_X.fit_transform(train_df[feature_cols].values)
train_y = scaler_y.fit_transform(train_df[[target_col]].values)

test_X = scaler_X.transform(test_df[feature_cols].values)
test_y = scaler_y.transform(test_df[[target_col]].values)

# =====================================================================
# 5. 建立序列資料集
# =====================================================================
def create_sequences(X, y, seq_len):
    """切成 (sample, seq_len, features)"""
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_X, train_y, SEQ_LEN)
X_test,  y_test  = create_sequences(test_X,  test_y,  SEQ_LEN)

print(f"\n訓練序列: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"測試序列: X_test={X_test.shape},  y_test={y_test.shape}")

# =====================================================================
# 6. 建立 LSTM 模型
# =====================================================================
model = Sequential([
    LSTM(64, activation="tanh", return_sequences=True, input_shape=(SEQ_LEN, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, activation="tanh"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="mse",
    metrics=["mae"]
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
)

# =====================================================================
# 9. 評估
# =====================================================================
y_pred = model.predict(X_test, verbose=0)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

mae  = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2   = r2_score(y_test_inv, y_pred_inv)

print(f"\n========== 測試集評估結果 ==========")
print(f"MAE : {mae:,.2f} 萬噸")
print(f"RMSE: {rmse:,.2f} 萬噸")
print(f"R²  : {r2:.4f}")

# =====================================================================
# 10. 繪圖
# =====================================================================
# --- 10a. 訓練曲線 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["mae"], label="Train MAE")
axes[1].plot(history.history["val_mae"], label="Val MAE")
axes[1].set_title("MAE Curve")
axes[1].set_xlabel("Epoch")
axes[1].legend()
fig.suptitle("LSTM Training History", fontsize=14)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/training_history.png", dpi=150)
print(f"\n訓練曲線已存: {OUTPUT_DIR}/training_history.png")

# --- 10b. 預測 vs 實際 ---
test_dates = test_df["data_date"].values[SEQ_LEN:]

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(test_dates, y_test_inv, label="實際蓄水量", color="steelblue", linewidth=1.5)
ax.plot(test_dates, y_pred_inv, label="LSTM 預測", color="orangered", linewidth=1.5, alpha=0.8)
ax.set_title(f"寶二水庫蓄水量預測 (2022-2023) | MAE={mae:,.0f} | RMSE={rmse:,.0f} | R²={r2:.3f}")
ax.set_xlabel("日期")
ax.set_ylabel("有效蓄水量（萬噸）")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/prediction_vs_actual.png", dpi=150)
print(f"預測圖已存: {OUTPUT_DIR}/prediction_vs_actual.png")

# --- 10c. 殘差分佈 ---
residuals = y_test_inv.flatten() - y_pred_inv.flatten()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=50, edgecolor="white", color="gray")
ax.axvline(0, color="red", linestyle="--")
ax.set_title("預測殘差分佈（萬噸）")
ax.set_xlabel("實際值 - 預測值")
ax.set_ylabel("天數")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/residual_distribution.png", dpi=150)
print(f"殘差圖已存: {OUTPUT_DIR}/residual_distribution.png")

# =====================================================================
# 11. 儲存模型與 scaler
# =====================================================================
model.save(f"{OUTPUT_DIR}/bao2_lstm.keras")
print(f"模型已存: {OUTPUT_DIR}/bao2_lstm.keras")

# 儲存 scaler 參數（讓之後預測時可以還原）
np.save(f"{OUTPUT_DIR}/scaler_X_scale.npy", scaler_X.scale_)
np.save(f"{OUTPUT_DIR}/scaler_X_min.npy",   scaler_X.min_)
np.save(f"{OUTPUT_DIR}/scaler_y_scale.npy", scaler_y.scale_)
np.save(f"{OUTPUT_DIR}/scaler_y_min.npy",   scaler_y.min_)
np.save(f"{OUTPUT_DIR}/feature_cols.npy",   np.array(feature_cols))
print("Scaler 參數已存")

# =====================================================================
# 12. 每日預測誤差摘要
# =====================================================================
result_df = pd.DataFrame({
    "date":        test_dates,
    "actual":      y_test_inv.flatten(),
    "predicted":   y_pred_inv.flatten(),
    "residual":    residuals,
    "abs_error":   np.abs(residuals),
})
result_df["date"] = pd.to_datetime(result_df["date"])
result_df.to_csv(f"{OUTPUT_DIR}/daily_predictions.csv", index=False)
print(f"每日預測已存: {OUTPUT_DIR}/daily_predictions.csv")

print("\n========== 訓練完成 ==========")
print(f"最佳 Val Loss: {min(history.history['val_loss']):.6f}")
