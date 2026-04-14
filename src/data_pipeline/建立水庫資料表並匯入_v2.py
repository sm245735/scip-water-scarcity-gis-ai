#!/usr/bin/env python3
"""
=============================================================================
水庫資料庫建立與匯入腳本 v2（修正版）
=============================================================================
修正：
    - process_row 加入 exception handling，避免 transaction abort
    - storage_rate 用 NUMERIC(7,2) 放超過 1000% 的值
"""

import psycopg2, csv, os, time, sys
from datetime import datetime

DB_HOST = "db"
DB_PORT = "5432"
DB_NAME = "thesis_analysis"
DB_USER = "sm245735"
DB_PASS = "1qaz@WSX"

PROJECT_DIR = "/home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai"
RESERVOIR_CSV = "/app/data_samples/水庫ID對照表_自研版.csv"
RESERVOIR_DAILY_CSV = "/app/data/水庫歷史水情/水庫統計表_2016_2026.csv"

def get_conn():
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                             user=DB_USER, password=DB_PASS)

def create_tables(conn):
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS reservoir_daily CASCADE")
    cur.execute("DROP TABLE IF EXISTS reservoir_id_self CASCADE")
    print("已刪除舊有 tables")
    
    cur.execute("""
        CREATE TABLE reservoir_id_self (
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
    print("✅ reservoir_id_self 建立完成")
    
    cur.execute("""
        CREATE TABLE reservoir_daily (
            id                   SERIAL PRIMARY KEY,
            date                 DATE NOT NULL,
            reservoir_id         INTEGER NOT NULL,
            observation_time      TIMESTAMP,
            basin_rainfall_mm    NUMERIC(8,2),
            inflow_cms           NUMERIC(10,3),
            effective_storage     NUMERIC(12,2),
            outflow_cms          NUMERIC(10,3),
            water_level_m        NUMERIC(10,3),
            full_water_level_m   NUMERIC(10,3),
            storage_rate         NUMERIC(7,2),  -- 修正：放超過 999.99 的值
            UNIQUE(date, reservoir_id)
        )
    """)
    print("✅ reservoir_daily 建立完成（storage_rate 改為 NUMERIC(7,2)）")
    
    cur.execute("CREATE INDEX idx_reservoir_daily_date ON reservoir_daily(date)")
    cur.execute("CREATE INDEX idx_reservoir_daily_reservoir ON reservoir_daily(reservoir_id)")
    cur.execute("CREATE INDEX idx_reservoir_daily_dateres ON reservoir_daily(date, reservoir_id)")
    print("✅ 索引建立完成")
    
    cur.close()
    conn.commit()

def import_reservoir_id(conn):
    cur = conn.cursor()
    count = 0
    with open(RESERVOIR_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur.execute("""
                INSERT INTO reservoir_id_self (
                    reservoir_id, reservoir_name, location, capacity_10k_m3,
                    lon, lat, geom, soap_id, opendata_id, comparison_api_id, statistics_url_id, note
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                int(row['reservoir_id']),
                row['reservoir_name'],
                row['location'] or None,
                float(row['capacity_10k_m3']) if row['capacity_10k_m3'] else None,
                float(row['lon']) if row['lon'] else None,
                float(row['lat']) if row['lat'] else None,
                None,
                row['soap_id'] or None,
                row['opendata_id'] or None,
                row['comparison_api_id'] or None,
                row['statistics_url_id'] or None,
                row['note'] or None
            ))
            count += 1
    print(f"✅ reservoir_id_self 匯入完成：{count} 筆")
    cur.close()
    conn.commit()

def parse_num(val, remove_comma=False, remove_percent=False):
    if not val or val.strip() in ('--', '', '-'):
        return None
    v = val.strip()
    if remove_comma:
        v = v.replace(',', '')
    if remove_percent:
        v = v.replace('%', '').strip()
    try:
        return float(v)
    except:
        return None

def process_row_safe(cur, row, name_to_id):
    """處理單一 CSV row，加入錯誤處理"""
    if len(row) < 10:
        return False, 'skip_short_row'
    
    date_str = row[0].strip()
    name = row[1].strip()
    obs_time = row[2].strip() if len(row) > 2 else ''
    
    basin_rainfall = parse_num(row[3])
    inflow_cms = parse_num(row[4])
    water_level = parse_num(row[5])
    full_water_level = parse_num(row[6])
    effective_storage = parse_num(row[7], remove_comma=True)
    storage_rate = parse_num(row[8], remove_percent=True)
    outflow_cms = parse_num(row[9])
    
    if name not in name_to_id:
        return False, 'skip_unknown_reservoir'
    
    reservoir_id = name_to_id[name]
    
    obs_ts = None
    if obs_time and obs_time != '--':
        try:
            obs_ts = datetime.strptime(obs_time, '%Y-%m-%d %H:%M:%S')
        except:
            obs_ts = None
    
    cur.execute("""
        INSERT INTO reservoir_daily (
            date, reservoir_id, observation_time,
            basin_rainfall_mm, inflow_cms, effective_storage, outflow_cms,
            water_level_m, full_water_level_m, storage_rate
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date, reservoir_id) DO NOTHING
    """, (
        date_str, reservoir_id, obs_ts,
        basin_rainfall, inflow_cms, effective_storage, outflow_cms,
        water_level, full_water_level, storage_rate
    ))
    return True, 'ok'

def import_reservoir_daily(conn):
    cur = conn.cursor()
    
    cur.execute("SELECT reservoir_id, reservoir_name FROM reservoir_id_self")
    name_to_id = {name: rid for rid, name in cur.fetchall()}
    print(f"水庫名稱對照：{len(name_to_id)} 筆")
    
    total = 0
    errors = 0
    skipped = 0
    t0 = time.time()
    
    with open(RESERVOIR_DAILY_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        first = next(reader)
        # 確認是否為 header
        if first[0] == 'date':
            pass  # header，跳過
        else:
            ok, msg = process_row_safe(cur, first, name_to_id)
            if ok:
                total += 1
            elif msg == 'skip_short_row' or msg == 'skip_unknown_reservoir':
                skipped += 1
            else:
                errors += 1
                conn.rollback()  # 重要：rollback 後繼續
        
        batch = 0
        for row in reader:
            batch += 1
            ok, msg = process_row_safe(cur, row, name_to_id)
            if ok:
                total += 1
            elif msg == 'skip_short_row' or msg == 'skip_unknown_reservoir':
                skipped += 1
            else:
                errors += 1
                conn.rollback()  # 重要：abort 後 rollback 並繼續
            
            if batch % 50000 == 0:
                conn.commit()
                elapsed = time.time() - t0
                print(f"  進度：{total:,} 筆匯入，{skipped:,} 筆跳過，{errors} 筆錯誤 | {batch:,} rows processed | {elapsed:.0f}s")
    
    conn.commit()
    cur.close()
    
    elapsed = time.time() - t0
    print(f"\n✅ reservoir_daily 匯入完成")
    print(f"   成功：{total:,} 筆 | 跳過：{skipped:,} 筆 | 錯誤：{errors} 筆")
    print(f"   總耗時：{elapsed:.0f}s，平均 {total/elapsed:.0f} 筆/秒")

if __name__ == "__main__":
    print("=" * 60)
    print("水庫資料庫建立與匯入腳本 v2")
    print("=" * 60)
    
    conn = get_conn()
    print("已連線到資料庫\n")
    
    print("📦 Step 1: 建立資料表...")
    create_tables(conn)
    
    print("\n📥 Step 2: 匯入 reservoir_id_self（112 筆）...")
    import_reservoir_id(conn)
    
    print("\n📥 Step 3: 匯入 reservoir_daily（467,732 行）...")
    import_reservoir_daily(conn)
    
    conn.close()
    print("\n✅ 全部完成！")
