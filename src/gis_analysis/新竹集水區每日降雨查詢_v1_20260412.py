#!/usr/bin/env python3
"""
=============================================================================
水庫集水區每日平均降雨量查詢（LSTM 特徵工程）- 新竹縣市限定版
=============================================================================
功能說明：
    1. 先用縣市界線 shapefile 限縮新竹縣市範圍內的集水區
    2. 只對這些集水區做 Spatial Join，大幅減少查詢資料量
    3. 計算每個集水區每天的平均降雨量

前置資料：
    - 縣市界線：/app/data/03_GIS/COUNTY_MOI_1140318.shp（EPSG: TWD97）
    - 集水區：reservoir_catchments（EPSG: 4326）
    - 降雨點：rainfall_grid_data（EPSG: 4326）

執行方式：
    docker exec thesis_python_dev python /app/src/gis_analysis/新竹集水區每日降雨查詢.py

歷程：
    2026-04-12 v1：新增縣市界線限縮，解決查詢過慢問題
"""

import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# =============================================
# 設定區
# =============================================
GIS_PATH = "/app/data/03_GIS/COUNTY_MOI_1140318.shp"
DB_USER = os.getenv('DB_USER', 'sm245735')
DB_PASS = quote_plus(os.getenv('DB_PASSWORD', ''))
DB_HOST = os.getenv('DB_HOST', 'db')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'thesis_analysis')

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_hsinchu_basins():
    """
    用縣市界線找出新竹縣市內的集水區（減少查詢範圍）
    Returns list of basin names within Hsinchu County/City.
    """
    print("讀取縣市界線，找出新規縮小範圍...")

    # 讀取縣市 shapefile（TWD97）
    county_gdf = gpd.read_file(GIS_PATH)

    # 取出新竹縣 + 新竹市的 geometry
    hsinchu_names = ['新竹縣', '新竹市']
    hsinchu_gdf = county_gdf[county_gdf['COUNTYNAME'].isin(hsinchu_names)]
    print(f"  找到 {len(hsinchu_gdf)} 個新竹行政區")

    # 轉換至 EPSG:4326
    hsinchu_gdf = hsinchu_gdf.to_crs(epsg=4326)
    hsinchu_geom = hsinchu_gdf.geometry.union_all()  # 合併成一個 MultiPolygon（GeoPandas 1.0+）
    print(f"  合併後範圍：{hsinchu_geom}")

    # 查詢所有集水區，過濾出落在新竹縣市內的
    engine = create_engine(DB_URL)
    sql = """
        SELECT basin_id, basin_name, ws_name, geom
        FROM reservoir_catchments;
    """
    all_basins = gpd.GeoDataFrame.from_postgis(sql, engine, geom_col='geom')
    engine.dispose()

    # 空間過濾：集水區的 centroid 或 geometry 有在新竹縣市內
    # 使用 intersects 檢查（只要有交集就算）
    in_hsinchu = all_basins[all_basins.intersects(hsinchu_geom)]
    print(f"  新竹縣市內的集水區數量：{len(in_hsinchu)}")
    print(f"  集水區名稱：{in_hsinchu['basin_name'].tolist()[:10]}...")

    return in_hsinchu['basin_name'].tolist()


def query_rainfall(basin_names: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    查詢指定集水區的每日平均降雨量
    """
    if not basin_names:
        print("沒有集水區，請確認縣市界線資料")
        return pd.DataFrame()

    params = {
        "start_date": start_date,
        "end_date": end_date,
    }
    in_parts = []
    for i, bn in enumerate(basin_names):
        params[f"bn_{i}"] = bn
        in_parts.append(f":bn_{i}")
    in_clause = ", ".join(in_parts)

    sql = text(f"""
        SELECT
            c.basin_name,
            r.data_date,
            AVG(r.rainfall_mm) AS avg_catchment_rainfall
        FROM
            rainfall_grid_data r
        JOIN
            reservoir_catchments c
        ON
            ST_Intersects(r.geom, c.geom)
        WHERE
            r.data_date BETWEEN :start_date AND :end_date
            AND c.basin_name IN ({in_clause})
        GROUP BY
            c.basin_name,
            r.data_date
        ORDER BY
            c.basin_name,
            r.data_date;
    """)
    engine = create_engine(DB_URL)
    df = pd.read_sql(sql, engine, params=params)
    engine.dispose()
    return df


def main():
    print("=" * 50)
    print("新竹縣市集水區每日降雨量查詢")
    print("=" * 50)

    # Step 1：找出新竹縣市內的集水區
    hsinchu_basins = get_hsinchu_basins()

    if not hsinchu_basins:
        print("⚠️ 找不到新竹縣市內的集水區")
        return

    # Step 2：查詢 2023 年當作示範
    print("\n查詢 2023 年每日降雨量...")
    df = query_rainfall(
        basin_names=hsinchu_basins,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    print(f"\n結果：{len(df)} 筆")
    if len(df) > 0:
        print(f"涵蓋集水區：{df['basin_name'].nunique()} 個")
        print(df.head(10))
        # 匯出 CSV
        output = "/app/data/新竹縣市_集水區每日降雨量_2023.csv"
        df.to_csv(output, index=False, encoding='utf-8-sig')
        print(f"\n已匯出：{output}")
    else:
        print("⚠️ 查詢結果為空")


if __name__ == "__main__":
    main()
