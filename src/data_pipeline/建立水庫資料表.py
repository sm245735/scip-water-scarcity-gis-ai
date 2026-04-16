#!/usr/bin/env python3
"""
=============================================================================
水庫資料庫建立腳本
=============================================================================
用途：初始化資料庫（建立 reservoirs + reservoir_daily 表）

使用：
    # 一般初始化（有確認提示）
    docker exec thesis_python_dev python /app/src/data_pipeline/建立水庫資料表.py

    # 強制執行（不詢問，適用於 CI/自動化）
    docker exec thesis_python_dev python /app/src/data_pipeline/建立水庫資料表.py --force

⚠️  注意：本腳本會 DROP 舊有 tables！帶 --force 才會實際刪除。
"""

import psycopg2, os, sys, argparse

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = "thesis_analysis"
DB_USER = os.environ.get("DB_USER", "sm245735")
DB_PASSWORD = os.environ["DB_PASSWORD"]  # 從環境變數讀取

def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )

def create_tables(conn, force=False):
    cur = conn.cursor()

    # 檢查 tables 是否已有資料
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename IN ('reservoirs', 'reservoir_daily')
    """)
    existing = [r[0] for r in cur.fetchall()]

    if existing and not force:
        cur.execute("SELECT COUNT(*) FROM reservoir_daily")
        row_count = cur.fetchone()[0]
        print(f"⚠️  reservoir_daily 已有 {row_count} 筆資料！")
        print("如要繼續，請加上 --force 參數：")
        print("  python 建立水庫資料表.py --force")
        print("不放棄，請按 Ctrl+C。")
        sys.exit(1)

    if force and existing:
        print("⚠️  --force 已設定，開始刪除舊 tables...")
        cur.execute("DROP TABLE IF EXISTS reservoir_daily CASCADE")
        cur.execute("DROP TABLE IF EXISTS reservoirs CASCADE")
        print("  已刪除舊 tables")

    # 1. reservoirs（MDM 水庫主數據表）
    cur.execute("""
        CREATE TABLE reservoirs (
            reservoir_id       INTEGER PRIMARY KEY,
            reservoir_name    VARCHAR(100) NOT NULL,
            location          VARCHAR(50),
            capacity_10k_m3   NUMERIC(12,2),
            lon               DOUBLE PRECISION,
            lat               DOUBLE PRECISION,
            geom              GEOMETRY(Point, 4326),
            soap_id           VARCHAR(20),
            opendata_id       VARCHAR(20),
            comparison_api_id VARCHAR(20),
            statistics_url_id VARCHAR(20),
            note              TEXT
        )
    """)
    print("✅ reservoirs 建立完成")

    # 2. reservoir_daily（LSTM 訓練資料）
    cur.execute("""
        CREATE TABLE reservoir_daily (
            id                   SERIAL PRIMARY KEY,
            data_date            DATE NOT NULL,
            reservoir_id         INTEGER NOT NULL,
            observation_time      TIMESTAMP,
            basin_rainfall_mm    NUMERIC(8,2),
            inflow_cms           NUMERIC(10,3),
            effective_storage    NUMERIC(12,2),
            outflow_cms          NUMERIC(10,3),
            water_level_m        NUMERIC(10,3),
            full_water_level_m   NUMERIC(10,3),
            storage_rate         NUMERIC(7,2),
            UNIQUE(data_date, reservoir_id),
            FOREIGN KEY (reservoir_id) REFERENCES reservoirs(reservoir_id)
        )
    """)
    print("✅ reservoir_daily 建立完成（data_date + FK）")

    # 索引
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reservoir_daily_date       ON reservoir_daily(data_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reservoir_daily_reservoir ON reservoir_daily(reservoir_id)")
    print("✅ 索引建立完成")

    cur.close()
    conn.commit()

def populate_geom(conn):
    """用 lon/lat 填補 geom 幾何欄位"""
    cur = conn.cursor()
    cur.execute("""
        UPDATE reservoirs
        SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326)
        WHERE lon IS NOT NULL AND lat IS NOT NULL
    """)
    updated = cur.rowcount
    cur.close()
    conn.commit()
    print(f"✅ geom 填補完成：{updated} 筆")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="水庫資料庫建立腳本")
    parser.add_argument("--force", action="store_true",
                        help="強制刪除舊 tables 後重建（無確認提示）")
    args = parser.parse_args()

    print("=" * 60)
    print("水庫資料庫建立腳本")
    print("=" * 60)
    if args.force:
        print("⚠️  --force 已設定，將刪除舊 tables！")
    print()

    conn = get_conn()
    print("已連線到資料庫\n")

    print("📦 建立資料表...")
    create_tables(conn, force=args.force)

    print("\n📍 填補 geom...")
    populate_geom(conn)

    conn.close()
    print("\n✅ 全部完成！")
