#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
TCCIP 降雨資料批次匯入腳本 v5（記憶體優化版）
=============================================================================
功能說明：
    批次讀取 TCCIP 網格化觀測日降雨資料 CSV 檔案（新竹市、臺中市、臺南市），
    轉換為 GeoDataFrame（Point geometry）後分塊匯入 PostgreSQL/PostGIS 資料庫。

適用資料：
    - TCCIP 網格化觀測日資料（1960-2023 年）
    - CSV 寬表格式：LON/LAT + 日期欄位
    - 園區：竹科（新竹市）、中科（臺中市）、南科（臺南市）

作者：AI Assistant（OpenClaw Coder）
日期：2026 年 4 月 12 日
版本：v5（分塊 to_postgis，chunk_size=10000，記憶體優化）
=============================================================================
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os
import glob
import gc
from datetime import datetime

# ===========================================================================
# 設定區
# ===========================================================================
BASE_PATH = "/app/03. 資料/01. 氣候（TCCIP）"

DB_USER = "sm245735"
DB_PASS = "DB_PASSWORD_PLACEHOLDER"
DB_HOST = "db"
DB_PORT = "5432"
DB_NAME = "thesis_analysis"
DB_URL = f"postgresql+psycopg2://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

CSV_MAP = {
    "竹科": f"{BASE_PATH}/觀測_日資料_新竹市_降雨量/觀測_日資料_新竹市_降雨量_*.csv",
    "中科": f"{BASE_PATH}/觀測_日資料_臺中市_降雨量/觀測_日資料_臺中市_降雨量_*.csv",
    "南科": f"{BASE_PATH}/觀測_日資料_臺南市_降雨量/觀測_日資料_臺南市_降雨量_*.csv",
}

TARGET_TABLE = "rainfall_grid_data"
IMPORT_TIMESTAMP = datetime(2026, 4, 12)
CHUNK_SIZE = 10000  # 每批寫入列數（降低以避免記憶體不足）


# ===========================================================================
# 函式：process_and_load
# ===========================================================================
def process_and_load(csv_path: str, area_name: str, engine) -> int:
    """
    處理單一 CSV 檔案並分塊匯入資料庫。

    流程：
        1. 讀取 CSV（刪除空白欄）
        2. melt 寬表轉長表
        3. 解析日期、填入 Metadata
        4. 建立 Point geometry → GeoDataFrame
        5. 對齊欄位名稱（rename_geometry）
        6. 分塊 to_postgis（每塊 CHUNK_SIZE 列）
    """
    print(f"\n處理：{area_name} - {os.path.basename(csv_path)}")

    # Step 1：讀取 CSV
    df = pd.read_csv(csv_path, usecols=lambda c: not c.startswith("Unnamed"))
    print(f"  原始：{df.shape[0]} 列 × {df.shape[1]} 欄")

    # Step 2：寬表轉長表
    date_cols = [c for c in df.columns if c not in ("LON", "LAT")]
    df_long = df.melt(
        id_vars=["LON", "LAT"],
        value_vars=date_cols,
        var_name="Date",
        value_name="Rainfall"
    )
    del df
    gc.collect()

    # Step 3：解析日期與 Metadata
    df_long["Date"] = pd.to_datetime(df_long["Date"], format="mixed")
    df_long["area_name"] = area_name
    df_long["created_at"] = IMPORT_TIMESTAMP
    df_long["acquired_at"] = IMPORT_TIMESTAMP

    # Step 4：建立 Point geometry → GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_long,
        geometry=[Point(x, y) for x, y in zip(df_long["LON"], df_long["LAT"])],
        crs="EPSG:4326"
    )
    del df_long
    gc.collect()

    # Step 5：對齊欄位名稱
    gdf = gdf.rename(columns={
        "LON": "lon",
        "LAT": "lat",
        "Date": "data_date",
        "Rainfall": "rainfall_mm"
    }).rename_geometry("geom")

    cols = [
        "area_name", "lon", "lat", "geom",
        "data_date", "rainfall_mm",
        "created_at", "acquired_at"
    ]

    # Step 6：分塊寫入 PostGIS
    total_rows = len(gdf)
    written = 0
    for start in range(0, total_rows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_rows)
        chunk = gdf[cols].iloc[start:end]
        chunk.to_postgis(TARGET_TABLE, engine, if_exists="append", index=False)
        written += len(chunk)
        print(f"  已寫入 {written:,} / {total_rows:,} 列", end="\r")

    print(f"  已寫入：{written:,} 列")
    del gdf
    gc.collect()
    return written


# ===========================================================================
# 函式：main
# ===========================================================================
def main():
    print("=" * 60)
    print("TCCIP 降雨資料批次匯入 v5")
    print("目標資料表：rainfall_grid_data")
    print("日期：2026 年 4 月 12 日")
    print("分塊大小：{:,} 列".format(CHUNK_SIZE))
    print("=" * 60)

    engine = create_engine(DB_URL)

    total_files = 0
    total_rows = 0

    for area_name, glob_pattern in CSV_MAP.items():
        csv_files = sorted(glob.glob(glob_pattern))
        print(f"\n【{area_name}】共找到 {len(csv_files)} 個 CSV 檔案")

        for csv_path in csv_files:
            try:
                rows = process_and_load(csv_path, area_name, engine)
                total_files += 1
                total_rows += rows
            except Exception as e:
                print(f"\n  ❌ 錯誤：{e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"✅ 匯入完成！共處理 {total_files} 個檔案，{total_rows:,} 列資料")
    print(f"{'=' * 60}")

    engine.dispose()
    gc.collect()


if __name__ == "__main__":
    main()
