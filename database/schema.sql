-- =============================================
--  scip-water-scarcity-gis-ai
--  資料庫 Schema 初始化腳本
--  日期：2026-04-12
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
    res_name VARCHAR(100),                  -- 水庫名稱
    area_description TEXT,                 -- 範圍描述
    source VARCHAR(200),                   -- 資料來源
    build_date VARCHAR(20),                -- 建置日期
    geom GEOMETRY(GeometryZ, 4326),        -- 3D 幾何（Polygon Z）
    created_at TIMESTAMP DEFAULT '2026-04-12'
);

-- 空間索引
CREATE INDEX IF NOT EXISTS idx_reservoir_geom ON reservoir_boundaries USING GIST(geom);

-- 範例查詢：確認資料已匯入
-- SELECT area_name, COUNT(*) FROM rainfall_grid_data GROUP BY area_name;
-- SELECT ST_AsText(geom), data_date, rainfall_mm FROM rainfall_grid_data LIMIT 3;
-- SELECT res_name, ST_GeometryType(geom) FROM reservoir_boundaries LIMIT 5;

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
