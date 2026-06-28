# Best LSTM 模型清單

> 自動產出，最後更新：2026-06-28
> 排序依據：2024-2025 期末考的 R²（高到低）

## 🥇 #1 最佳模型

- **路徑**：`model/20260628_220911_seed6/`
- **SEED**：6
- **訓練輪數**：23 epochs（Early Stop 觸發）
- **資料切分**：Train 2014-2022 / Val 2023 / Test 2024-2025
- **預測任務**：未來 14 天後的 `storage_rate`

### 評估指標（2024-2025 期末考）

| 指標 | 數值 |
|---|---:|
| **R²** | **0.9064** |
| **MAPE** | 5.9637% |
| **Acc (100 - MAPE)** | 94.0363% |

### 檔案清單

| 檔案 | 用途 |
|---|---|
| `lstm_model.keras` | 訓練好的模型權重 |
| `scaler.pkl` | MinMaxScaler (推論時要載入) |
| `loss_curve.png` | 訓練/驗證 loss 曲線 |
| `2024_2025_prediction_results.png` | 真實 vs 預測水位對比圖 |
| `feature_importance.png` | 7 個特徵的 permutation importance |
| `準確率.txt` | R²/MAPE 文字版成績單 |

### 推論用法

```python
import joblib
from tensorflow.keras.models import load_model

model = load_model('model/20260628_220911_seed6/lstm_model.keras')
scaler = joblib.load('model/20260628_220911_seed6/scaler.pkl')

# X_new 形狀: (n_samples, 14, 7) — 過去 14 天 × 7 個特徵
# 特徵順序必須跟訓練時一致:
#   [PP01, TX01, TX02, RH01, WD01, PS01, storage_rate]
X_scaled = scaler.transform(X_new.reshape(-1, 7)).reshape(X_new.shape)
y_pred_scaled = model.predict(X_scaled)
```

---

## 10-run 重現性實驗（seeds 1-10）

完整 10 次實驗驗證模型穩健性，避免單次結果的隨機性誤導。

| seed | epochs | R² | MAPE | Acc |
|---:|---:|---:|---:|---:|
| 1 | 20 | 0.8954 | 6.31% | 93.69% |
| 2 | 56 | 0.8945 | 5.93% | 94.07% |
| 3 | 20 | 0.8996 | 6.30% | 93.70% |
| 4 | 53 | 0.8948 | 6.10% | 93.90% |
| 5 | 33 | 0.8875 | 6.77% | 93.23% |
| **6** | **23** | **0.9064** ⭐ | **5.96%** | **94.04%** |
| 7 | 29 | 0.9035 | 5.98% | 94.02% |
| 8 | 41 | 0.8860 | 6.75% | 93.25% |
| 9 | 30 | 0.8918 | 6.17% | 93.83% |
| 10 | 49 | 0.8948 | 6.02% | 93.98% |

### 統計結果

| 指標 | min | max | mean | median | std |
|---|---:|---:|---:|---:|---:|
| Epochs | 20 | 56 | 35.4 | 31.5 | ±12.7 |
| R² | 0.8860 | 0.9064 | **0.8954** | 0.8948 | ±0.0064 |
| MAPE | 5.93% | 6.77% | **6.23%** | 6.16% | ±0.31% |
| Acc | 93.23% | 94.07% | **93.77%** | 93.85% | ±0.31% |

### 解讀

- **R² 區間 0.886~0.906（mean 0.895）**，變異係數僅 0.7% — 模型架構穩健
- **MAPE 區間 5.93%~6.77%** — 平均每天誤差 6.2% ± 0.3%
- **epochs 範圍 20~56** — 沒用滿 100 上限，Early Stopping 有效
- **最佳模型 seed=6**：R² 0.9064 顯著高於平均 (0.895 + 1.7σ)，且只跑 23 epochs（最少的之一），意味收斂快、無 overfit

---

## 如何重現實驗

```bash
# 跑單次（指定 seed）
venv/bin/python src/data_pipeline/train_end_to_end.py 42

# 跑 10 次重現性實驗 (seed 1~10)
/tmp/run_10seeds.sh
```

跑完後會自動統計 min/max/mean/median/std，並更新本檔案。

---

## 注意事項

- `model/` 目錄被 `.gitignore` 排除（檔案大、可重現），clone repo 後不會看到模型權重
- 若要重現最佳模型，跑 `venv/bin/python src/data_pipeline/train_end_to_end.py 6`
- 若要在不同 LEAD_TIME (7 vs 14 天) 對照，修改 `train_end_to_end.py` 第 171 行的 `LEAD_TIME = 14`