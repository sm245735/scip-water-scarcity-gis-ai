# SCIP (Science Park) Water Scarcity GIS-AI

> 基於 GIS 空間分析與 LSTM 時間序列預測之水資源風險評估系統
>
> **目標水庫**：寶山第二水庫（reservoir_id=23，新竹科學園區供水關鍵，設計蓄水量 3,100 萬噸）
>
> **研究問題**：極端旱象下，新竹科學園區供水是否會觸發缺水危機？

---

## 📋 專案背景

台灣水資源面臨空間分布不均與季節性乾旱挑戰，半導體產業作為用水大戶，需有效管理供水風險。本系統整合：

| 核心技術 | 應用場景 |
|---------|---------|
| **GIS 空間分析** | 集水區識別、網格化降雨與水庫集水區交集計算 |
| **LSTM 時間序列預測** | 水庫蓄水量預測，提前掌握缺水風險 |
| **缺水風險分類器** | 預測「30天內是否觸發缺水危機」 |
| **PostgreSQL + PostGIS** | 空間資料儲存與查詢 |

### 研究目標

1. 建立新竹地區水庫缺水預警模型（寶山第二水庫）
2. 串聯 GIS 空間資料（集水區、降雨網格）與 LSTM 時間序列
3. 提供視覺化風險地圖（ArcGIS JS 前端）

### 為何選擇寶山第二水庫？

| 原因 | 說明 |
|------|------|
| **供水主力** | 有效蓄水量 3,100 萬噸，是寶山（500萬噸）的 6 倍 |
| **聯合運用** | 水利署將「寶山-寶二聯合運用系統」視為單一供水單元 |
| **無調度變數** | 寶二為離槽水庫，降雨→蓄水映射純粹自然水文，適合 LSTM 建模 |
| **學長建議** | 研究範圍界定，避免多水庫維度過高與人為調度複雜化 |

---

## 🗂️ 資料來源

| 資料 | 來源 | 時間範圍 | 備註 |
|------|------|---------|------|
| 水庫日水情 | WRA Statistics.aspx（Selenium）| 2016-2026 | 全台 112 水庫，每日凌晨 01:00 更新 |
| 頭前溪集水區降雨 | TCCIP × PostGIS ST_Intersects | 2016-2023 | 43 格點/天，面積加權平均 |
| 集水區邊界 | 水利署 110 年度 shapefile | — | EPSG:3826 → EPSG:4326 |
| 水庫邊界 | ressub.shp | — | MultiPolygon（需 ST_Force2D）|

---

## 🗄️ 資料庫結構

### reservoir_daily（水庫每日觀測，LSTM 訓練資料）

| 欄位 | 類型 | 說明 |
|------|------|------|
| `id` | SERIAL | Primary Key |
| `data_date` | DATE | 資料日期 |
| `reservoir_id` | INTEGER | FK → reservoirs（流水號 1~112） |
| `observation_time` | TIMESTAMP | 水情觀測時間（UTC+8） |
| `basin_rainfall_mm` | NUMERIC(8,2) | 集水區降雨（mm） |
| `inflow_cms` | NUMERIC(10,3) | 進水量（cms）⚠️ 97.8% 缺值 |
| `effective_storage` | NUMERIC(12,2) | **Label**：有效蓄水量（萬立方公尺） |
| `outflow_cms` | NUMERIC(10,3) | 出水量（cms）⚠️ 97.8% 缺值 |
| `water_level_m` | NUMERIC(10,3) | 水位（公尺） |
| `full_water_level_m` | NUMERIC(10,3) | 滿水位（公尺） |
| `storage_rate` | NUMERIC(7,2) | 蓄水率（%） |

### reservoirs（MDM 水庫主數據表）

| 欄位 | 類型 | 說明 |
|------|------|------|
| `reservoir_id` | INTEGER | Primary Key（流水號 1~112） |
| `reservoir_name` | VARCHAR(100) | 水庫名稱 |
| `location` | VARCHAR(50) | 行政區 |
| `capacity_10k_m3` | NUMERIC(12,2) | 設計有效蓄水量（萬立方公尺） |
| `lon`, `lat`, `geom` | — | 空間欄位（待座標檔補填） |
| `statistics_url_id` | VARCHAR(20) | ReservoirChart.aspx?key= 的數字 |

### 重要水庫對照

| reservoir_id | reservoir_name | statistics_url_id |
|--------------|---------------|------------------|
| **23** | **寶山第二水庫（研究目標）** | 10405 |
| 22 | 寶山水庫 | 10404 |

---

## 🔬 LSTM 模型設計

### §3.2 特徵選擇（已驗證資料限制）

本研究原規劃使用進水量（inflow）與出水量（outflow）作為 LSTM 輸入特徵，以符合傳統水平衡模型的概念。然而資料品質分析顯示，寶山第二水庫 2016-2023 期間，`inflow_cms` 與 `outflow_cms` 僅各有 25 天之有效觀測值（觀測率 0.9%）。此現象反映寶二作為離槽水庫之運作特性——其流率並非每日例行記錄。

本研究改採「降雨–蓄水直接映射」之特徵策略：

```python
FEATURE_COLS = [
    "basin_rainfall_tccip_mm",  # 當日集水區降雨（TCCIP 空間加權）
    "rainfall_7d_sum",           # 7 天累積降雨
    "rainfall_30d_sum",         # 30 天累積降雨
    "effective_storage",         # 當前蓄水量
    "storage_diff_1d",           # 前日蓄水變化（一階差分）
    "doy_sin",                   # 年週期 sin 編碼
    "doy_cos",                  # 年週期 cos 編碼
]
LABEL_COL = "effective_storage"  # shift -1，預測明日蓄水量
```

### 訓練集建置流程（v2）

```
PostgreSQL（髒資料，有缺值、斷日）
  ↓ 撈取寶二（id=23）2016-2023
  ↓ 日期對齊 → 2,922 天完整時間序列
  ↓ 產生 is_imputed 旗標（補值前標記）
  ↓ 補值（線性內插 ≤7 天，超過保留 NaN）
  ↓ 加衍生特徵（rainfall_7d_sum, rainfall_30d_sum, storage_diff_1d）
  ↓ 加時間編碼（doy_sin, doy_cos）
  ↓ 輸出 CSV
  → data/寶二訓練集_v1.csv
```

### 模型架構

**LSTM 回歸模型**（`src/02_train_lstm.py`）
- SEQ_LEN=30（輸入 30 天序列）
- 訓練：2016-01-01 ~ 2021-12-31（6年）
- 測試：2022-01-01 ~ 2023-12-31（2年）
- 架構：LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)
- Callbacks：EarlyStopping(patience=15) + ReduceLROnPlateau(factor=0.5)

**缺水風險分類器**（`src/03_train_classifier.py`）
- 任務：預測「30天內是否觸發缺水危機（蓄水量 < 1,200 萬噸）」
- 架構：Bidirectional LSTM(64) → Dropout(0.3) → Bidirectional LSTM(32) → Dropout(0.3) → Dense(16) → Dense(1, sigmoid)
- 類別不平衡處理：class_weight

---

## 📊 模型表現

### LSTM 回歸模型（測試集 638 天，2022-02 ~ 2023-12）

| 指標 | 數值 | 說明 |
|------|------|------|
| **MAE** | **211.8 萬噸** | 約佔總容量 6.8% |
| **RMSE** | — | 待補 |
| **R²** | — | 待補 |
| Max Error | 528.8 萬噸 | 極端值 |

### 缺水風險分類器（測試集 638 天）

| 指標 | 數值 | 說明 |
|------|------|------|
| **AUC-ROC** | **0.981** ✅ | 區分能力極強 |
| Accuracy | 77.6% | 整體正確率 |
| **Recall** | **100%** ✅ | 所有缺水事件無漏網 |
| Precision | 17.3% | 誤報較多（類別不平衡）|
| F1 | 0.296 | |
| 缺水事件 | 30 天 / 638 天（4.7%）| |

> **口試亮點**：Recall=100% 代表「模型從未漏掉任何一次缺水事件」，適合在口試時拿出來說。

---

## 📁 目錄結構

```
scip-water-scarcity-gis-ai/
├── data/                      # 原始資料（不進 Git）
│   └── 寶二訓練集_v1.csv       # LSTM 訓練集（2,922 天）
├── data_samples/              # 展示用樣本資料（進 Git）
│   ├── 水庫ID對照表_自研版.csv
│   └── 新竹頭前溪集水區每日降雨量_2016_2023.csv
├── database/
│   └── schema.sql
├── models/                    # 訓練好的模型（進 Git）
│   ├── bao2_lstm.keras        # LSTM 回歸模型
│   ├── bao2_classifier.keras  # 缺水風險分類器
│   ├── utils_metrics.py        # 評估指標工具
│   ├── *.png                  # 訓練曲線、預測圖、ROC、CM
│   └── *.npy                  # Scaler 參數（供 inference）
├── src/
│   ├── 02_train_lstm.py       # LSTM 訓練腳本
│   ├── 03_train_classifier.py  # 分類器訓練腳本
│   ├── data_pipeline/          # 資料收集腳本
│   └── gis_analysis/           # GIS 空間分析腳本
├── doc/
│   ├── 資料來源.md
│   ├── 技術筆記.md
│   ├── 待辦事項.md
│   └── 論文日誌.md
└── README.md
```

---

## 🏗️ 系統架構

```
┌──────────────────────────────────────────────────────────┐
│                    Docker Compose                          │
├────────────────────┬─────────────────────────────────────┤
│  PostgreSQL+PostGIS│     Python 開發環境                   │
│  (db:5432)        │     thesis_python_dev                 │
│  Port: 9235 (host)│     TensorFlow / GeoPandas / Jupyter  │
└────────────────────┴─────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ 水庫日資料    │   │  TCCIP 降雨   │   │  GIS 邊界    │
  │ Selenium 收集 │   │  PostGIS 查詢 │   │  Shapefile   │
  └──────────────┘   └──────────────┘   └──────────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │   01_build_training_set  │
                │   寶二訓練集 v1/v2 CSV   │
                └────────────┬────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
  ┌─────────────────┐                 ┌─────────────────────┐
  │ 02_train_lstm   │                 │ 03_train_classifier │
  │ 蓄水量回歸       │                 │ 缺水風險二元分類     │
  │ MAE=211.8萬噸   │                 │ AUC=0.981          │
  └────────┬────────┘                 └──────────┬──────────┘
           │                                     │
           ▼                                     ▼
  ┌─────────────────┐                 ┌─────────────────────┐
  │ bao2_lstm.keras │                 │ bao2_classifier.keras│
  └─────────────────┘                 └─────────────────────┘
```

---

## 🚀 快速開始

### 1. 啟動環境

```bash
docker compose up -d
```

### 2. 進入 Python 容器

```bash
docker exec -it thesis_python_dev bash
```

### 3. 重新訓練 LSTM

```bash
docker exec thesis_python_dev python /app/src/02_train_lstm.py
```

### 4. 重新訓練分類器

```bash
docker exec thesis_python_dev python /app/src/03_train_classifier.py
```

---

## 🔑 環境變數

敏感資訊使用環境變數，請勿 commit `.env`：

```bash
DB_PASSWORD=your_password_here
```

---

## 📝 Commit 規範

```
feat: 新功能說明
fix: 修正問題
docs: 文件更新
refactor: 重構程式碼
clean: 清理不需要的檔案
```

---

## 📚 相關文件

- [資料來源](./doc/資料來源.md) — API URL、資料格式說明
- [技術筆記](./doc/技術筆記.md) — 爬蟲過程與問題解法
- [待辦事項](./doc/待辦事項.md) — 論文專案待辦追蹤
- [論文日誌](./doc/論文日誌.md) — 開發日誌與環境狀態

---

## ✅ 已完成項目

| 日期 | 項目 |
|------|------|
| 2026-04-12 | 水庫統計資料收集（2016-2026，467,732 行）|
| 2026-04-12 | TCCIP 降雨資料匯入 PostGIS（頭前溪 43 格點，2016-2023）|
| 2026-04-13 | 頭前溪集水區降雨 CSV 建置（2,922 天）|
| 2026-04-14 | reservoir_daily 資料庫匯入（409,248 行，112 水庫）|
| 2026-04-19 | 訓練集 v2 建置（rainfall_7d/30d_sum, storage_diff_1d）|
| 2026-04-23 | LSTM 模型訓練完成（MAE=211.8 萬噸）|
| 2026-04-23 | 缺水風險分類器訓練完成（AUC=0.981）|
| 2026-04-28 | 模型 commit + README 更新 |

---

## ⚠️ 待完成項目

- [ ] 補填 `reservoirs.lon/lat/geom` 空間座標（待康康提供座標檔）
- [ ] 取得科學園區實際用水量（月資料 → 線性內插為日資料）
- [ ] 加入氣象資料（蒸發量、氣溫）強化 LSTM 特徵
- [ ] ArcGIS JS 前端視覺化風險地圖
- [ ] LSTM 對照組實驗（SVR、ARIMA、Random Forest）
- [ ] R²、NSE、KGE 等完整水資源指標補測

---

*最後更新：2026-04-28*
