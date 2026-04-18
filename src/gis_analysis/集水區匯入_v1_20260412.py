#!/usr/bin/env python3
"""
=============================================================================
集水區（Catchment Area）Shapefile 匯入腳本
=============================================================================
功能說明：
    讀取 110 年度全臺 839 子集水區 shapefile，
    轉換座標系統 (EPSG:3826 → EPSG:4326) 後匯入 PostgreSQL/PostGIS。

資料來源：
    data/05_集水區/（110年度全臺839子集水區範圍圖_UTF8.shp）
    - 來源：台灣水利相關單位（110年度）
    - 筆數：839 個子集水區
    - CRS：EPSG:3826 (TWD97)

資料庫目的地：
    Table: reservoir_catchments
    欄位：basin_id, basin_name, ws_id, ws_name, branch, area_m2, geom, created_at

執行方式：
    docker exec thesis_python_dev python /app/src/gis_analysis/集水區匯入_v1_20260412.py
    # 本地環境：PROJECT_ROOT=/path/to/repo python src/gis_analysis/集水區匯入_v1_20260412.py

歷程：
    2026-04-12 v1：初始版本（EPSG:3826 → 4326）
"""

import os
import sys
import glob
import geopandas as gpd
import psycopg2
from urllib.parse import quote_plus

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from path_utils import CATCHMENT_DATA_DIR

BASE_PATH = CATCHMENT_DATA_DIR  # data/05_集水區
DB_USER = os.getenv('DB_USER', 'sm245735')
DB_HOST = os.getenv('DB_HOST', 'db')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'thesis_analysis')
TARGET_TABLE = "reservoir_catchments"


def main():
    print("=" * 50)
    print("集水區 Shapefile → PostgreSQL/PostGIS")
    print("=" * 50)

    # 用 wildcard 找 shapefile（避免檔名亂碼問題）
    shp_files = glob.glob(os.path.join(BASE_PATH, "*.shp"))
    if not shp_files:
        print("找不到 shapefile！")
        return
    shp_path = shp_files[0]
    print(f"讀取：{shp_path}")

    # 讀取 shapefile（來源為 EPSG:3826）
    gdf = gpd.read_file(shp_path)
    print(f"  原始筆數：{len(gdf)} 筆")
    print(f"  原始 CRS：{gdf.crs}")

    # 轉換座標系統至 WGS84（EPSG:4326）
    gdf = gdf.to_crs(epsg=4326)
    print(f"  轉換後 CRS：{gdf.crs}")

    # 建立資料庫連線
    raw_conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=os.getenv('DB_PASSWORD', ''),
        database=DB_NAME
    )
    cur = raw_conn.cursor()

    # 建立 Table（如不存在）
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            id SERIAL PRIMARY KEY,
            basin_id INTEGER,
            basin_name VARCHAR(100),
            ws_id INTEGER,
            ws_name VARCHAR(100),
            branch VARCHAR(100),
            area_m2 NUMERIC,
            geom GEOMETRY(MultiPolygon, 4326),
            created_at TIMESTAMP DEFAULT '2026-04-12'
        );
    """)
    raw_conn.commit()
    print("  Table 建立完成")

    # 批次寫入
    inserted = 0
    geom_col = gdf.geometry.name
    for _, row in gdf.iterrows():
        geom_wkb = row[geom_col].wkb

        cur.execute(
            f"INSERT INTO {TARGET_TABLE} "
            f"(basin_id, basin_name, ws_id, ws_name, branch, area_m2, geom) "
            f"VALUES (%s, %s, %s, %s, %s, %s, ST_SetSRID(ST_GeomFromWKB(%s), 4326));",
            (
                int(row.get("BASIN_ID", 0)),
                str(row.get("BASIN_NAME", ""))[:100],
                int(row.get("WS_ID", 0)) if row.get("WS_ID") else None,
                str(row.get("WS_NAME", ""))[:100],
                str(row.get("BRANCH", ""))[:100],
                float(row.get("AREA_M2", 0)) if row.get("AREA_M2") else None,
                psycopg2.Binary(geom_wkb)
            )
        )
        inserted += 1

    raw_conn.commit()
    cur.close()
    raw_conn.close()
    print(f"  已寫入：{inserted} 筆")
    print(f"\n✅ 完成！共匯入 {inserted} 個子集水區")


if __name__ == "__main__":
    main()
