-- =============================================
--  scip-water-scarcity-gis-ai
--  資料庫 Schema 初始化腳本
--  日期：2026-04-12（更新：2026-04-15 學長 review 版）
-- =============================================

-- =============================================
-- 模組一：空間與觀測基礎資料表（GIS 疊圖與特徵來源）
-- =============================================

-- 1. rainfall_grid_data（TCCIP 網格化降雨資料）
-- 用途：儲存 TCCIP 網格化降雨資料，支援空間查詢
-- 調整（2026-04-15 學長）：
--   - date → data_date（避免 SQL 保留字，Python ORM 不會報錯）
--   - created_at / acquired_at → CURRENT_TIMESTAMP（資料庫自動記錄）
CREATE TABLE IF NOT EXISTS rainfall_grid_data (
    id SERIAL PRIMARY KEY,
    area_name VARCHAR(50) NOT NULL,          -- 區域名稱（如「新竹頭前溪」）
    lon DOUBLE PRECISION NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(Point, 4326),             -- 空間欄位，GIS 核心
    data_date DATE NOT NULL,                -- 資料日期（避免保留字，改用 data_date）
    rainfall_mm NUMERIC(10, 2),             -- 降雨量（mm）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 匯入時間（資料庫自動記錄）
    acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- 資料取得時間（資料庫自動記錄）
);
CREATE INDEX IF NOT EXISTS idx_rainfall_date ON rainfall_grid_data(data_date);
CREATE INDEX IF NOT EXISTS idx_rainfall_area ON rainfall_grid_data(area_name);
CREATE INDEX IF NOT EXISTS idx_rainfall_geom ON rainfall_grid_data USING GIST(geom);

-- 2. reservoir_boundaries（水庫實體蓄水範圍 Shapefile）
-- 用途：儲存台灣各水庫蓄水範圍 shapefile
-- 調整（2026-04-15 學長）：
--   - res_name → reservoir_name（統一命名，方便與 reservoirs 表 JOIN）
--   - GeometryZ → MultiPolygon（明確指定，降維提升空間查詢效能）
--   - created_at → CURRENT_TIMESTAMP
CREATE TABLE IF NOT EXISTS reservoir_boundaries (
    id SERIAL PRIMARY KEY,
    reservoir_name VARCHAR(100),            -- 水庫名稱（統一命名，方便 JOIN）
    area_description TEXT,                   -- 區域描述
    source VARCHAR(200),                    -- 資料來源
    build_date VARCHAR(20),                 -- 興建日期
    geom GEOMETRY(MultiPolygon, 4326),      -- 蓄水範圍（GeometryZ 降維，明確指定 MultiPolygon）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_reservoir_geom ON reservoir_boundaries USING GIST(geom);

-- 3. reservoir_catchments（台灣 839 個子集水區範圍）
-- 用途：台灣 839 個子集水區範圍
-- 來源：水利相關單位（110 年度）
-- CRS：EPSG:4326（WGS84）
-- 調整（2026-04-15 學長）：
--   - geom 明確指定為 MultiPolygon，確保 CRS 為 4326
--   - created_at → CURRENT_TIMESTAMP
CREATE TABLE IF NOT EXISTS reservoir_catchments (
    id SERIAL PRIMARY KEY,
    basin_id INTEGER,
    basin_name VARCHAR(100),               -- 所屬流域（如「頭前溪」）
    ws_id INTEGER,
    ws_name VARCHAR(100),
    branch VARCHAR(100),
    area_m2 NUMERIC,                       -- 集水區面積（平方公尺）
    geom GEOMETRY(MultiPolygon, 4326),     -- 明確指定 MultiPolygon，確保 CRS 為 4326
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_catchments_geom ON reservoir_catchments USING GIST(geom);

-- =============================================
-- 模組二：機器學習 LSTM 核心資料表（主數據與時間序列）
-- =============================================

-- 4. reservoirs（MDM 水庫主數據表）
-- 用途：水庫唯一識別對照表，串接所有資料來源
-- 調整（2026-04-15 學長）：
--   - reservoir_id_self → reservoirs（表名用複數，符合資料庫慣例）
--   - 空間欄位（lon/lat/geom）預留給未來前端 Dashboard 打點使用
--   - created_at → CURRENT_TIMESTAMP
CREATE TABLE IF NOT EXISTS reservoirs (
    reservoir_id       INTEGER PRIMARY KEY,         -- 流水號 1~112（INT，不是 "001"）
    reservoir_name    VARCHAR(100) NOT NULL,       -- 水庫名稱
    location          VARCHAR(50),                 -- 行政區（可留空）
    capacity_10k_m3   NUMERIC(12,2),               -- 設計有效蓄水量（萬立方公尺）

    -- 空間欄位（預留給未來前端 Dashboard 打點使用）
    lon               DOUBLE PRECISION,            -- 水庫中心經度（可留空，等座標檔）
    lat               DOUBLE PRECISION,            -- 水庫中心緯度（可留空，等座標檔）
    geom              GEOMETRY(Point, 4326),       -- 水庫位置 point（可留空，等座標檔）

    -- 多來源 ID 對照（解決政府各單位代碼不一致的痛點）
    soap_id           VARCHAR(20),                 -- SOAP API 的 ST_NO
    opendata_id       VARCHAR(20),                 -- opendata.wra.gov.tw 的 ID
    comparison_api_id VARCHAR(20),                 -- Comparison API 的 ST_NO
    statistics_url_id VARCHAR(20),                -- ReservoirChart.aspx?key= 的數字

    note              TEXT                          -- 備註
);

-- 5. reservoir_daily（LSTM 訓練資料表）
-- 用途：水庫每日水情，10年歷史資料（2016-2026）
-- 來源：Statistics.aspx 爬蟲收集（467,732 行，3654 天，112 水庫）
-- 調整（2026-04-15 學長）：
--   - date → data_date（避免 SQL 保留字，Python ORM 不會報錯）
--   - storage_rate → NUMERIC(7,2)（有些水庫蓄水率超過 999.99%）
--   - created_at → CURRENT_TIMESTAMP
CREATE TABLE IF NOT EXISTS reservoir_daily (
    id                   SERIAL PRIMARY KEY,
    data_date            DATE NOT NULL,             -- 資料日期（避免 SQL 保留字）
    reservoir_id         INTEGER NOT NULL,         -- FK → reservoirs(reservoir_id)
    observation_time      TIMESTAMP,                -- 水情觀測時間（UTC+8）

    -- LSTM Golden Features（ML 模型核心欄位）
    basin_rainfall_mm     NUMERIC(8,2),             -- Feature：集水區降雨（mm）
    inflow_cms            NUMERIC(10,3),           -- Feature：進水量（cms）
    effective_storage     NUMERIC(12,2),            -- Label：要預測的目標（萬立方公尺）
    outflow_cms           NUMERIC(10,3),           -- Feature：消耗量/出水量（cms）

    -- 額外保留欄位
    water_level_m         NUMERIC(10,3),            -- 水位（公尺）
    full_water_level_m   NUMERIC(10,3),           -- 滿水位（公尺）
    storage_rate         NUMERIC(7,2),            -- 蓄水率（%）（放超過 999.99% 的值）

    -- 唯一約束（同一水庫同一天只有一筆記錄）
    CONSTRAINT fk_reservoir FOREIGN KEY (reservoir_id) REFERENCES reservoirs(reservoir_id),
    UNIQUE(data_date, reservoir_id)
);

CREATE INDEX IF NOT EXISTS idx_reservoir_daily_date       ON reservoir_daily(data_date);
CREATE INDEX IF NOT EXISTS idx_reservoir_daily_reservoir ON reservoir_daily(reservoir_id);
CREATE INDEX IF NOT EXISTS idx_reservoir_daily_dateres   ON reservoir_daily(data_date, reservoir_id);