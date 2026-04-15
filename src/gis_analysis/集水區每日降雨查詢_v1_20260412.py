#!/usr/bin/env python3
"""
=============================================================================
水庫集水區每日平均降雨量查詢（LSTM 特徵工程）
=============================================================================
功能說明：
    使用 PostGIS Spatial Join（ST_Intersects）計算每個水庫集水區
    每天的平均降雨量。
    結果可直接作為 LSTM 模型的輸入特徵（Feature）。

前置條件：
    1. rainfall_grid_data 表已有 TCCIP 網格降雨資料（EPSG:4326）
    2. reservoir_catchments 表已有集水區資料（EPSG:4326，已由 EPSG:3826 轉換）

SQL 邏輯（學長提供）：
    - rainfall_grid_data：降雨網格點（Point, EPSG:4326）
    - reservoir_catchments：集水區多邊形（Polygon, EPSG:4326）
    - ST_Intersects：判斷降雨網格點是否落在集水區多邊形內
    - AVG(rainfall_mm)：計算落在集水區內所有網格點的平均降雨量

資料庫：thesis_analysis

執行方式：
    docker exec thesis_python_dev python /app/src/gis_analysis/集水區每日降雨查詢.py

歷程：
    2026-04-12 v1：初始版本
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# =============================================
# 設定區
# =============================================
DB_USER = os.getenv('DB_USER', 'sm245735')
DB_PASS = quote_plus(os.getenv('DB_PASSWORD', ''))
DB_HOST = os.getenv('DB_HOST', 'db')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'thesis_analysis')

DB_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def query_catchment_rainfall(
    basin_names: list[str] | None = None,
    area_name: str = "竹科",
    start_date: str | None = None,
    end_date: str | None = None
) -> pd.DataFrame:
    """
    查詢指定集水區的每日平均降雨量

    Parameters
    ----------
    basin_names : list[str] | None
        集水區名稱列表（如 ['寶山水庫', '寶山第二水庫']）
        若為 None，則查詢所有集水區
    area_name : str
        降雨資料的來源地區（預設 '竹科'）
    start_date : str | None
        起始日期（格式：'YYYY-MM-DD'）
    end_date : str | None
        結束日期（格式：'YYYY-MM-DD'）

    Returns
    -------
    pd.DataFrame
        含 basin_name, data_date, avg_catchment_rainfall 的 DataFrame
    """
    # 組合參數
    params = {"area_name": area_name}
    if basin_names:
        params["basin_names"] = basin_names
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    # 動態 WHERE 子句（避免 SQL injection）
    where_parts = ["r.area_name = :area_name"]
    if basin_names:
        for i, bn in enumerate(basin_names):
            params[f"bn_{i}"] = bn
        in_clause = ", ".join([f":bn_{i}" for i in range(len(basin_names))])
        where_parts.append(f"c.basin_name IN ({in_clause})")
    if start_date:
        where_parts.append("r.data_date >= :start_date")
    if end_date:
        where_parts.append("r.data_date <= :end_date")

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
            {" AND ".join(where_parts)}
        GROUP BY
            c.basin_name,
            r.data_date
        ORDER BY
            c.basin_name,
            r.data_date
    """)

    engine = create_engine(DB_URL)
    df = pd.read_sql(sql, engine, params=params)
    engine.dispose()
    return df


def main():
    print("=" * 50)
    print("水庫集水區每日平均降雨量查詢")
    print("=" * 50)

    # 查詢新竹地區 2023 年資料當作示範
    df = query_catchment_rainfall(
        area_name="竹科",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    print(f"\n查詢結果：{len(df)} 筆")
    if len(df) > 0:
        print(f"集水區數量：{df['basin_name'].nunique()}")
        print(df.head(10))
        # 匯出 CSV（可餵給 LSTM 模型）
        output_path = "/app/data/集水區每日降雨量_2023.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n已匯出：{output_path}")
    else:
        print("⚠️ 查詢結果為空，請確認：")
        print("  1. rainfall_grid_data 是否有該 area_name 的資料")
        print("  2. reservoir_catchments 是否有對應集水區")
        print("  3. 座標系統是否一致（兩表都應為 EPSG:4326）")


if __name__ == "__main__":
    main()
