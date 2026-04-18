#!/usr/bin/env python3
"""
=============================================================================
水庫蓄水範圍 Shapefile 匯入腳本
=============================================================================
功能說明：
    讀取水庫蓄水範圍 shapefile (ressub.shp)，
    轉換座標系統 (EPSG:3826 → EPSG:4326) 後，
    使用 ST_SetSRID + ST_Force2D + ST_GeomFromWKB 寫入 PostgreSQL/PostGIS。
    注意：原始 shapefile 幾何為 3D (PolygonZ)，需用 ST_Force2D 移除 Z 維度以符合 schema 約束。

資料來源：
    data/水資源（水庫蓄水）/ressub.shp
    - 來源：水利地理資訊服務平台
    - 建置日期：2024-06-17
    - 筆數：129 個水庫

資料庫目的地：
    Table: reservoir_boundaries
    欄位：id, reservoir_name, area_description, source, build_date, geom(MultiPolygon, 4326), created_at

執行方式：
    # Docker 環境（容器內）
    docker exec thesis_python_dev python /app/src/gis_analysis/水庫蓄水範圍匯入_v1_20260412.py
    # 本地環境（使用 PROJECT_ROOT 或相對路徑）
    PROJECT_ROOT=/path/to/repo python src/gis_analysis/水庫蓄水範圍匯入_v1_20260412.py

歷程：
    2026-04-12 v1：初始版本，解決 Z dimension 問題（ST_Force3DZ）
    2026-04-18 v2：改用 ST_Force2D（schema 為 2D MultiPolygon，強制保留 Z 會被 PostGIS 拒絕）
"""

import os
import geopandas as gpd
import psycopg2
from urllib.parse import quote_plus

# =============================================
# 設定區
# =============================================
import os
import sys

# 將 src/utils 加入路徑，確保無論從哪個位置執行都能找到 path_utils
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from path_utils import RESERVOIR_DATA_DIR

BASE_PATH = RESERVOIR_DATA_DIR  # data/水資源（水庫蓄水）
DB_USER = os.getenv('DB_USER', 'sm245735')
DB_HOST = os.getenv('DB_HOST', 'db')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'thesis_analysis')
TARGET_TABLE = "reservoir_boundaries"


def main():
    print("=" * 50)
    print("水庫蓄水範圍 → PostgreSQL/PostGIS")
    print("=" * 50)

    shp_path = os.path.join(BASE_PATH, "ressub.shp")
    print(f"讀取：{shp_path}")

    # 讀取 shapefile（來源為 TWD97 / EPSG:3826）
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

    # 建立 Table（如不存在，MultiPolygon 明確指定）
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            id SERIAL PRIMARY KEY,
            reservoir_name VARCHAR(100),
            area_description TEXT,
            source VARCHAR(200),
            build_date VARCHAR(20),
            geom GEOMETRY(MultiPolygon, 4326),
            created_at TIMESTAMP DEFAULT '2026-04-12'
        );
    """)
    raw_conn.commit()
    print("  Table 建立完成")

    # 批次寫入（ST_Force3DZ 保留 Z 維度）
    inserted = 0
    for _, row in gdf.iterrows():
        res_name = str(row.get("RES_NAME", row.get("NAME", "未知")))[:100]
        area_desc = str(row.get("蓄水範p", ""))[:500]
        geom_wkb = row.geometry.wkb  # Shapely 自動產出 WKB

        cur.execute(
            f"INSERT INTO {TARGET_TABLE} "
            f"(reservoir_name, area_description, source, build_date, geom) "
            f"VALUES (%s, %s, %s, %s, ST_SetSRID(ST_Force2D(ST_GeomFromWKB(%s)), 4326));",
            (res_name, area_desc, "水利地理資訊服務平台", "2024-06-17", psycopg2.Binary(geom_wkb))
        )
        inserted += 1

    raw_conn.commit()
    cur.close()
    raw_conn.close()

    print(f"  已寫入：{inserted} 筆")
    print(f"\n✅ 完成！共匯入 {inserted} 個水庫範圍")


if __name__ == "__main__":
    main()
