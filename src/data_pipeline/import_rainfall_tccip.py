#!/usr/bin/env python3
"""
TCCIP 降雨資料批次匯入腳本 v4
修正：
1. DB URL 密碼需 URL encode（密碼包含 @ 字元）
2. 使用 gdf.rename_geometry('geom') 而非 rename(columns) 來改 geometry 欄位名稱
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os
import glob
from datetime import datetime

# ============ 設定區 ============
BASE_PATH = "/app/03. 資料/01. 氣候（TCCIP）"
DB_USER = "sm245735"
DB_PASS = "DB_PASSWORD_PLACEHOLDER"  # 密碼含 @，需 URL encode
DB_HOST = "db"
DB_PORT = "5432"
DB_NAME = "thesis_analysis"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

CSV_ALL = sorted(glob.glob(f"{BASE_PATH}/觀測_日資料_新竹市_降雨量/觀測_日資料_新竹市_降雨量_*.csv"))


def process_and_load(csv_path: str, area_name: str, engine):
    print(f"\n處理：{area_name} - {os.path.basename(csv_path)}")

    # 讀取並刪除空白欄（結尾逗號會產生 Unnamed 欄）
    df = pd.read_csv(csv_path, usecols=lambda c: not c.startswith("Unnamed"))
    print(f"  原始：{df.shape[0]} 列 x {df.shape[1]} 欄")

    # melt：LON/LAT 以外全部變 Date + Rainfall
    date_cols = [c for c in df.columns if c not in ("LON", "LAT")]
    df_long = df.melt(id_vars=["LON", "LAT"], value_vars=date_cols,
                      var_name="Date", value_name="Rainfall")

    # 日期解析（mixed format 應對 20230101 格式）
    df_long["Date"] = pd.to_datetime(df_long["Date"], format="mixed")

    # 固定欄位
    df_long["area_name"] = area_name
    df_long["created_at"] = datetime(2026, 4, 11)
    df_long["acquired_at"] = datetime(2026, 4, 11)

    # 建 Point geometry（GeoDataFrame 預設欄位名稱為 'geometry'）
    gdf = gpd.GeoDataFrame(
        df_long,
        geometry=[Point(x, y) for x, y in zip(df_long["LON"], df_long["LAT"])],
        crs="EPSG:4326"
    )

    # 對齊欄位名稱
    gdf = gdf.rename(columns={
        "LON": "lon",
        "LAT": "lat",
        "Date": "data_date",
        "Rainfall": "rainfall_mm"
    })

    # 重要：PostGIS 欄位名稱是 'geom'，不是 'geometry'
    # 使用 rename_geometry 而非 rename(columns=...) 才能正確改 geometry 欄位名稱
    gdf = gdf.rename_geometry("geom")

    cols = ["area_name", "lon", "lat", "geom", "data_date", "rainfall_mm", "created_at", "acquired_at"]
    gdf[cols].to_postgis("rainfall_grid_data", engine, if_exists="append", index=False)
    print(f"  已寫入：{len(gdf)} 列")


def main():
    print("=" * 50)
    print("TCCIP 降雨資料 v4 → PostgreSQL")
    print("=" * 50)

    engine = create_engine(DB_URL)
    total = 0

    for csv_path in CSV_ALL:
        try:
            process_and_load(csv_path, "竹科", engine)
            total += 1
        except Exception as e:
            print(f"  ❌ 錯誤：{e}")

    print(f"\n✅ 完成！共處理 {total} 個檔案")
    engine.dispose()


if __name__ == "__main__":
    main()
