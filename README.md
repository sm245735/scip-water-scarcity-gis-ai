# SCIP Water Scarcity GIS-AI

> 基於 GIS 空間分析与 LSTM 時間序列預測之水資源風險評估系統
>
> 目標水庫：寶山第二水庫（新竹科學園區供水關鍵）

---

## 📋 專案背景

台灣水資源長期面臨空間分布不均與季節性乾旱挑戰，半導體產業作為用水大戶，需有效管理供水風險。本系統整合：

| 核心技術 | 應用場景 |
|---------|---------|
| **GIS 空間分析** | 集水區識別、網格化降雨與水庫集水區交集計算 |
| **LSTM 時間序列預測** | 水庫入流量預測，提前掌握缺水風險 |
| **PostgreSQL + PostGIS** | 空間資料儲存與查詢 |

### 研究目標

1. 建立新竹地區水庫缺水預警模型
2. 串聯 GIS 空間資料（集水區、降雨網格）與 LSTM 時間序列
3. 提供視覺化風險地圖（ArcGIS JS 前端）

---

## 🗂️ 資料來源

| 資料 | 來源 | 時間範圍 | 備註 |
|------|------|---------|------|
| 水庫日資料 | WRA Comparison API | 2025-2026 | 18 個防汛重點水庫 |
| 網格化降雨 | TCCIP（中央氣象署）| 1960-2023 | 新竹市 147 網格點 |
| 竹科定點降雨 | TCCIP 預處理 | 2019-2023 | 竹科園區座標（121.01, 24.78）|
| 集水區邊界 | 水利署 110 年度 shapefile | — | EPSG:3826 → EPSG:4326 |
| 水庫邊界 | ressub.shp | — | PolygonZ（需 ST_Force3DZ）|

### 水庫 ID × 名稱對照（部分）

| ID | 名稱 | 行政區 |
|----|------|--------|
| **10405** | **寶山第二水庫** | **新竹** |
| 10601 | 明德水庫 | 苗栗 |
| 20101 | 鯉魚潭水庫 | 苗栗/台中 |
| 30502 | 曾文水庫 | 台南/嘉義 |
| 30503 | 南化水庫 | 台南 |

> 完整 103 個水庫 ID 對照見 `data/reservoir_id_map.csv`

---

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                        使用者端                             │
│   ArcGIS JS Frontend（風險地圖 + 預測結果視覺化）            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      API 層                                 │
│   FastAPI（待建）                                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────┐  ┌─────────────────────────────────┐
│   LSTM 模型訓練      │  │   資料收集系統                    │
│   (Jupyter/Model)   │  │   - WRA Comparison API（歷史）    │
│   - Scikit-learn    │  │   - WRA SOAP API（即時）          │
│   - TensorFlow/Keras│  │   - TCCIP 降雨（年度 CSV）        │
└─────────────────────┘  └─────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    資料庫（PostgreSQL）                      │
│   rainfall_grid_data  │  reservoir_daily                   │
│   reservoir_catchments│  reservoir_boundaries              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 目錄結構

```
scip-water-scarcity-gis-ai/
├── data/                      # 原始資料（不進 Git）
│   ├── 03. 資料/
│   │   └── 01. 氣候（TCCIP）/   # 降雨 CSV
│   ├── reservoir_id_map.csv    # 水庫 ID × 名稱對照表
│   └── 水庫歷史日資料_ComparisonAPI.csv
├── database/
│   └── schema.sql              # 資料庫結構定義
├── doc/                        # 文件（進 Git）
│   ├── 資料來源.md
│   ├── 技術筆記.md
│   └── 論文開發環境.md
├── frontend/                   # ArcGIS JS 前端
├── models/                     # LSTM 模型訓練腳本
├── notebooks/                  # Jupyter 分析 notebook
├── src/
│   ├── data_pipeline/          # 資料收集腳本
│   │   ├── 水庫歷史日資料收集_v1_20260412.py
│   │   └── 水庫日資料收集_v1_20260412.py
│   └── gis_analysis/           # PostGIS 空間查詢
├── requirements.txt            # Python 依賴
└── README.md                   # 本檔案
```

---

## 🔧 環境需求

### Docker 環境

```bash
docker compose up -d
```

| Container | 用途 |
|-----------|------|
| `thesis_python_dev` | Python 3 + Jupyter |
| `thesis_postgres` | PostgreSQL 16 + PostGIS |

### 主要 Python 套件

```
psycopg2-binary>=2.9
sqlalchemy>=2.0
geopandas>=0.14
shapely>=2.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
tensorflow>=2.15
```

---

## 🚀 快速開始

### 1. 啟動環境

```bash
cd scip-water-scarcity-gis-ai
docker compose up -d
```

### 2. 資料庫初始化

```bash
docker exec -it thesis_postgres psql -U sm245735 -d thesis_analysis -f /app/database/schema.sql
```

### 3. 收集水庫日資料

```bash
# 即時收集（每小時 cron）
docker exec thesis_python_dev python /app/src/data_pipeline/水庫日資料收集_v1_20260412.py

# 歷史收集（Comparison API，需數小時）
docker exec thesis_python_dev python /app/src/data_pipeline/水庫歷史日資料收集_v1_20260412.py \
  --start 2016/01/01 --end 2026/04/12 --days-per-batch 10
```

### 4. LSTM 模型訓練

```bash
cd notebooks
jupyter notebook
# 開啟 LSTM_水庫入流量預測.ipynb
```

---

## 📊 資料庫結構

### `reservoir_daily`（水庫每日觀測）

| 欄位 | 類型 | 說明 |
|------|------|------|
| `id` | SERIAL | Primary Key |
| `reservoir_id` | VARCHAR(10) | 水庫 ID |
| `reservoir_name` | VARCHAR(50) | 水庫名稱 |
| `observation_time` | TIMESTAMP | 觀測時間 |
| `effective_storage_萬噸` | FLOAT | 有效蓄水量 |
| `water_level_m` | FLOAT | 水位（公尺）|
| `inflow_cms` | FLOAT | 入流量（cms）|
| `total_outflow_cms` | FLOAT | 出流量（cms）|
| `catchment_rainfall_mm` | FLOAT | 集水區降雨（mm）|
| `water_draw_cms` | FLOAT | 放水量（cms）|

### `rainfall_grid_data`（TCCIP 網格降雨）

| 欄位 | 類型 | 說明 |
|------|------|------|
| `id` | SERIAL | Primary Key |
| `data_date` | DATE | 日期 |
| `lon` | FLOAT | 經度 |
| `lat` | FLOAT | 緯度 |
| `rainfall_mm` | FLOAT | 日降雨量（mm）|

---

## 🔑 環境變數

敏感資訊使用環境變數，請勿 commit `.env` 或 `.env.local`：

```bash
DB_PASSWORD=your_password_here
```

---

## 📝 Commit 規範

```
feat: 新功能說明（中文补充）
fix: 修正問題（中文补充）
docs: 文件更新（中文补充）
refactor: 重構程式碼（中文补充）
```

---

## 📚 相關文件

- [資料來源](./doc/資料來源.md) — API URL、資料格式說明
- [技術筆記](./doc/技術筆記.md) — 爬蟲過程與問題解法
- [論文開發環境](./doc/論文開發環境.md) — 開發日誌與環境狀態

---

*最後更新：2026-04-12*