# SCIP (Science Park) Water Scarcity GIS-AI

> 基於 GIS 空間分析與 LSTM 時間序列預測之水資源風險評估系統
>
> 目標水庫：寶山第二水庫（新竹科學園區供水關鍵）

---

## 📋 專案背景

台灣水資源長期面臨空間分布不均與季節性乾旱挑戰，半導體產業作為用水大戶，需有效管理供水風險。本系統整合：

| 核心技術 | 應用場景 |
|---------|---------|
| **GIS 空間分析** | 集水區識別、網格化降雨與水庫集水區交集計算 |
| **LSTM 時間序列預測** | 水庫蓄水量預測，提前掌握缺水風險 |
| **PostgreSQL + PostGIS** | 空間資料儲存與查詢 |

### 研究目標

1. 建立新竹地區水庫缺水預警模型（寶山第二水庫）
2. 串聯 GIS 空間資料（集水區、降雨網格）與 LSTM 時間序列
3. 提供視覺化風險地圖（ArcGIS JS 前端）

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
| `data_date` | DATE | 資料日期（避免 SQL 保留字） |
| `reservoir_id` | INTEGER | FK → reservoirs（流水號 1~112） |
| `observation_time` | TIMESTAMP | 水情觀測時間（UTC+8） |
| `basin_rainfall_mm` | NUMERIC(8,2) | Feature：集水區降雨（mm） |
| `inflow_cms` | NUMERIC(10,3) | Feature：進水量（cms） |
| `effective_storage` | NUMERIC(12,2) | **Label**：有效蓄水量（萬立方公尺） |
| `outflow_cms` | NUMERIC(10,3) | Feature：消耗量/出水量（cms） |
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
| **23** | **寶山第二水庫**（研究目標） | 10405 |
| 1 | 七美水庫 | — |
| 22 | 寶山水庫 | 10404 |

> 完整 112 個水庫對照見 `data_samples/水庫ID對照表_自研版.csv`

---

## 📊 現有資料

| 資料 | 筆數 | 日期範圍 |
|------|------|---------|
| reservoir_daily | 413,000+ 筆 | 2016-01-01 ~ 2026-04-14 |
| rainfall_grid_data（TCCIP） | — | 2016-2023（頭前溪集水區 43 格點） |
| 頭前溪集水區降雨 | 2,922 天 | 2016-01-01 ~ 2023-12-31 |

---

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────┐
│              ArcGIS JS Frontend              │
│     （風險地圖 + 預測結果視覺化）               │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│              LSTM 模型訓練                   │
│   (Jupyter Notebook / Python)              │
└────────────────────┬──────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐  ┌──────────────────────┐
│  資料庫 PostGIS   │  │  資料收集系統          │
│  reservoir_daily  │  │  Statistics.aspx      │
│  rainfall_grid   │  │  每日 01:00 更新      │
└──────────────────┘  └──────────────────────┘
```

---

## 📂 目錄結構

```
scip-water-scarcity-gis-ai/
├── data/                      # 原始資料（不進 Git）
│   └── 水庫統計_gap_2026.csv   # gap fill 產出
├── data_samples/              # 展示用樣本資料（進 Git）
│   ├── 水庫ID對照表_自研版.csv  # 112 水庫 MDM 對照表
│   └── 新竹頭前溪集水區每日降雨量_2016_2023.csv
├── database/
│   └── schema.sql              # 資料庫結構定義
├── doc/
│   ├── 資料來源.md
│   ├── 技術筆記.md
│   ├── 待辦事項.md
│   └── 論文開發環境.md
├── logs/                      # 執行日誌（不進 Git）
├── src/
│   ├── data_pipeline/
│   │   ├── 水庫Statistics_每日收集_host.py   # 每日 01:00 收集
│   │   ├── 建立水庫資料表.py                 # DB 建表 + 匯入
│   │   ├── TCCIP降雨資料匯入_v5_20260412.py  # TCCIP 降雨匯入
│   │   ├── check_todos.py                    # 待辦事項檢查工具
│   │   └── spawn_coder_for_review.py         # 程式碼審查發包腳本
│   └── gis_analysis/
│       ├── 集水區匯入_v1_20260412.py          # 集水區邊界匯入
│       ├── 集水區每日降雨查詢_v1_20260412.py  # 集水區每日降雨查詢
│       ├── 新竹集水區每日降雨查詢_v1_20260412.py  # 新竹集水區降雨查詢
│       ├── 水庫蓄水範圍匯入_v1_20260412.py    # 水庫蓄水範圍匯入
│       └── 竹科降雨資料匯出_v1_20260412.py   # 竹科降雨資料匯出
└── README.md
```

---

## 🔧 環境需求

### Docker 環境

```bash
docker compose up -d
```

| Container | 用途 |
|-----------|------|
| `thesis_python_dev` | Python 3 + Jupyter + Selenium |
| `thesis_postgres` | PostgreSQL 16 + PostGIS（port 9235） |

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
selenium>=4.0
```

---

## 🚀 快速開始

### 1. 啟動環境

```bash
docker compose up -d
```

### 2. 初始化資料庫

```bash
docker exec thesis_python_dev python /app/src/data_pipeline/建立水庫資料表.py
```

### 3. 每日水庫資料收集（凌晨 01:00 自動執行）

```bash
# 每天 01:00 自動執行（需先啟動 Chrome Remote Debugging）
nohup google-chrome --remote-debugging-port=18800 &
/usr/bin/python3 src/data_pipeline/水庫Statistics_每日收集_host.py
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
feat: 新功能說明（中文補充）
fix: 修正問題（中文補充）
docs: 文件更新（中文補充）
refactor: 重構程式碼（中文補充）
clean: 清理不需要的檔案
```

---

## 📚 相關文件

- [資料來源](./doc/資料來源.md) — API URL、資料格式說明
- [技術筆記](./doc/技術筆記.md) — 爬蟲過程與問題解法
- [待辦事項](./doc/待辦事項.md) — 論文專案待辦追蹤
- [論文開發環境](./doc/論文開發環境.md) — 開發日誌與環境狀態

---

*最後更新：2026-04-15*
*主要更新：廢除 Comparison API，全面改用 Statistics.aspx，ID mapping 問題已杜絕*