-- =============================================
--  scip-water-scarcity-gis-ai
--  資料庫 Schema 初始化腳本
--  日期：2026-04-12（更新：2026-04-14 學長 review 版）
-- =============================================

-- rainfall_grid_data 表
-- 用途：儲存 TCCIP 網格化降雨資料，支援空間查詢
CREATE TABLE IF NOT EXISTS rainfall_grid_data (
    id SERIAL PRIMARY KEY,
    area_name VARCHAR(50) NOT NULL,          -- '竹科', '中科', '南科'
    lon DOUBLE PRECISION NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(Point, 4326),             -- 空間欄位，GIS 核心
    data_date DATE NOT NULL,                -- 資料日期 (如 2023-01-01)
    rainfall_mm NUMERIC(10, 2),             -- 降雨量 (mm)
    created_at TIMESTAMP DEFAULT '2026-04-11',
    acquired_at TIMESTAMP DEFAULT '2026-04-11'
);

-- 索引：加速特定日期/區域/空間查詢
CREATE INDEX IF NOT EXISTS idx_rainfall_date ON rainfall_grid_data(data_date);
CREATE INDEX IF NOT EXISTS idx_rainfall_area ON rainfall_grid_data(area_name);
CREATE INDEX IF NOT EXISTS idx_rainfall_geom ON rainfall_grid_data USING GIST(geom);

-- reservoir_boundaries 表
-- 用途：儲存台灣各水庫蓄水範圍 shapefile
CREATE TABLE IF NOT EXISTS reservoir_boundaries (
    id SERIAL PRIMARY KEY,
    res_name VARCHAR(100),
    area_description TEXT,
    source VARCHAR(200),
    build_date VARCHAR(20),
    geom GEOMETRY(GeometryZ, 4326),
    created_at TIMESTAMP DEFAULT '2026-04-12'
);
CREATE INDEX IF NOT EXISTS idx_reservoir_geom ON reservoir_boundaries USING GIST(geom);

-- reservoir_catchments 表
-- 用途：台灣 839 個子集水區範圍
-- 來源：水利相關單位（110年度）
-- CRS：EPSG:4326（WGS84，由 EPSG:3826 轉換）
CREATE TABLE IF NOT EXISTS reservoir_catchments (
    id SERIAL PRIMARY KEY,
    basin_id INTEGER,
    basin_name VARCHAR(100),
    ws_id INTEGER,
    ws_name VARCHAR(100),
    branch VARCHAR(100),
    area_m2 NUMERIC,
    geom GEOMETRY(Geometry, 4326),
    created_at TIMESTAMP DEFAULT '2026-04-12'
);
CREATE INDEX IF NOT EXISTS idx_catchments_geom ON reservoir_catchments USING GIST(geom);

-- =============================================
-- 2026-04-14 新增：學長 review 版 schema
-- =============================================

-- reservoir_id_self（MDM 主數據表）
-- 用途：水庫唯一識別對照表，串接所有資料來源
-- 設計：學長建議 MDM 概念，避免「A 系統叫 1202，B 叫寶山第二水庫」崩潰問題
CREATE TABLE IF NOT EXISTS reservoir_id_self (
    reservoir_id       INTEGER PRIMARY KEY,         -- 流水號 1~112（INT，不是001）
    reservoir_name    VARCHAR(100) NOT NULL,
    location          VARCHAR(50),                 -- 行政區（可留空）
    capacity_10k_m3   NUMERIC(12,2),               -- 設計有效蓄水量（萬立方公尺）

    -- 空間欄位（GIS，未來由康康提供座標後填入）
    lon               DOUBLE PRECISION,             -- 水庫中心經度（留空，等座標檔）
    lat               DOUBLE PRECISION,            -- 水庫中心緯度（留空，等座標檔）
    geom              GEOMETRY(Point, 4326),       -- 水庫位置（留空，等座標檔）

    -- 多來源 ID 對照（避免各系統 ID 不同造成混乱）
    soap_id           VARCHAR(20),                 -- SOAP API 的 ST_NO
    opendata_id       VARCHAR(20),                 -- opendata.wra.gov.tw 的 ID
    comparison_api_id VARCHAR(20),                 -- Comparison API 的 ST_NO
    statistics_url_id VARCHAR(20),                -- ReservoirChart.aspx?key= 的數字

    note              TEXT                         -- 備註
);

-- reservoir_daily（LSTM 訓練資料表）
-- 用途：水庫每日水情，10年歷史資料（2016-2026）
-- 來源：Statistics.aspx 爬蟲收集（467,732 行，3654 天，112 水庫）
CREATE TABLE IF NOT EXISTS reservoir_daily (
    id                   SERIAL PRIMARY KEY,
    date                 DATE NOT NULL,             -- 爬蟲執行日期
    reservoir_id         INTEGER NOT NULL,        -- FK → reservoir_id_self(reservoir_id)

    observation_time      TIMESTAMP,                -- 水情時間（UTC+8）

    -- LSTM Golden Features（ML 模型核心欄位）
    basin_rainfall_mm     NUMERIC(8,2),             -- Feature：集水區降雨（mm）
    inflow_cms            NUMERIC(10,3),           -- Feature：進水量（cms）
    effective_storage     NUMERIC(12,2),           -- Label：要預測的目標（萬立方公尺）
    outflow_cms           NUMERIC(10,3),           -- Feature：消耗量/出水量（cms）

    -- 額外保留欄位
    water_level_m         NUMERIC(10,3),            -- 水位（公尺）
    full_water_level_m   NUMERIC(10,3),           -- 滿水位（公尺）
    storage_rate         NUMERIC(5,2),            -- 蓄水率（%）

    -- 唯一約束（同一水庫同一天只有一筆記錄）
    UNIQUE(date, reservoir_id)
);

CREATE INDEX IF NOT EXISTS idx_reservoir_daily_date       ON reservoir_daily(date);
CREATE INDEX IF NOT EXISTS idx_reservoir_daily_reservoir ON reservoir_daily(reservoir_id);
CREATE INDEX IF NOT EXISTS idx_reservoir_daily_dateres   ON reservoir_daily(date, reservoir_id);