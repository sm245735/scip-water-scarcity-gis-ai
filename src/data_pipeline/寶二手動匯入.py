#!/usr/bin/env python3
"""
=============================================================================
寶二水庫 單日手動匯入工具
=============================================================================

用途：康康手動從水利署網頁複製 1 天寶二資料貼進來，工具直接 UPDATE/INSERT
      fhy_reservoir_data。只動寶二（id=23），其他水庫不動。

跟 補特定日期.py 的差異：
  - 不用 selenium、不連網頁
  - 接受 CLI 參數（data_date + 7 個數值欄位）→ 直接 SQL
  - observation_time 統一存 data_date 22:00
  - '--' / 空字串視為 NULL

使用：
  /root/scip-water-scarcity-gis-ai/venv/bin/python \\
      src/data_pipeline/寶二手動匯入.py \\
      --date 2016-04-20 \\
      --water-level 149.59 \\
      --full-water-level 150 \\
      --effective-storage 3086.37 \\
      --storage-rate 98.07 \\
      --inflow "" \\
      --rainfall "" \\
      --outflow 0

  # 不確定的欄位省略即可，會存 NULL
  ... --date 2016-04-20 --water-level 149.59 --full-water-level 150 \\
      --effective-storage 3086.37 --storage-rate 98.07

驗證：
  docker exec -i postgres psql -U steveyang -d gis_db -c "
    SELECT data_date, observation_time, storage_rate, effective_storage
    FROM fhy_reservoir_data WHERE reservoir_id=23 AND data_date='2016-04-20';
  "
"""

import os
import argparse
import sys
import psycopg2

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = "gis_db"
DB_USER = "steveyang"
DB_PASS = os.environ.get("DB_PASSWORD", "")
if not DB_PASS:
    import subprocess
    r = subprocess.run(
        ["docker", "inspect", "postgres", "--format",
         "{{range .Config.Env}}{{println .}}{{end}}"],
        capture_output=True, text=True
    )
    for line in r.stdout.splitlines():
        key = 'POSTGRES' + '_' + 'PASSWORD' + '='
        if line.startswith(key):
            DB_PASS = line.split('=', 1)[1]
            break

BAOER_ID = 23  # 寶二水庫
def parse_num(val):
    """'--' / '' / '-' 視為 None；其餘轉 float"""
    if val is None or str(val).strip() in ('--', '', '-', 'None'):
        return None
    v = str(val).strip().replace(',', '').replace('%', '')
    try:
        return float(v)
    except ValueError:
        return None


def main():
    p = argparse.ArgumentParser(description="寶二水庫單日手動匯入")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--water-level", default="", help="水位 (公尺)")
    p.add_argument("--full-water-level", default="", help="滿水位 (公尺)")
    p.add_argument("--effective-storage", default="", help="有效蓄水量 (萬立方公尺)")
    p.add_argument("--storage-rate", default="", help="蓄水百分比 (%)")
    p.add_argument("--inflow", default="", help="進流量 (cms)")
    p.add_argument("--rainfall", default="", help="集水區累積降雨量 (mm)")
    p.add_argument("--outflow", default="", help="水庫出流量小計 (cms)")
    p.add_argument("--dry-run", action="store_true", help="只 print SQL 不執行")
    args = p.parse_args()

    # 觀測時間統一存 data_date 22:00（跟現有 daily 收集一致）
    obs_time = f"{args.date} 22:00:00"

    values = {
        "basin_rainfall_mm": parse_num(args.rainfall),
        "inflow_cms": parse_num(args.inflow),
        "effective_storage": parse_num(args.effective_storage),
        "outflow_cms": parse_num(args.outflow),
        "water_level_m": parse_num(args.water_level),
        "full_water_level_m": parse_num(args.full_water_level),
        "storage_rate": parse_num(args.storage_rate),
    }

    print(f"=== 寶二手動匯入：{args.date} ===")
    for k, v in values.items():
        print(f"  {k}: {v}")
    print(f"  observation_time: {obs_time}")
    print(f"  data_date: {args.date}")
    print(f"  reservoir_id: {BAOER_ID}")

    if args.dry_run:
        print("(dry-run 模式，未實際寫入)")
        return

    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                            user=DB_USER, password=DB_PASS)
    cur = conn.cursor()

    # 檢查是否已存在
    cur.execute(
        "SELECT id, storage_rate FROM fhy_reservoir_data "
        "WHERE reservoir_id=%s AND data_date=%s",
        (BAOER_ID, args.date),
    )
    row = cur.fetchone()

    if row:
        existing_id, old_rate = row
        print(f"\n  📝 已存在 row id={existing_id}（舊 storage_rate={old_rate}）→ UPDATE")
        cur.execute("""
            UPDATE fhy_reservoir_data SET
                observation_time = %s,
                basin_rainfall_mm = %s,
                inflow_cms = %s,
                effective_storage = %s,
                outflow_cms = %s,
                water_level_m = %s,
                full_water_level_m = %s,
                storage_rate = %s
            WHERE id = %s
        """, (obs_time, values["basin_rainfall_mm"], values["inflow_cms"],
              values["effective_storage"], values["outflow_cms"],
              values["water_level_m"], values["full_water_level_m"],
              values["storage_rate"], existing_id))
        action = "UPDATE"
    else:
        # 撈 max(id) 當起始
        cur.execute("SELECT COALESCE(MAX(id), 0) FROM fhy_reservoir_data")
        next_id = cur.fetchone()[0] + 1
        print(f"\n  ➕ 不存在 → INSERT（new id={next_id}）")
        cur.execute("""
            INSERT INTO fhy_reservoir_data (
                id, reservoir_id, observation_time,
                basin_rainfall_mm, inflow_cms, effective_storage, outflow_cms,
                water_level_m, full_water_level_m, storage_rate,
                data_date
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (next_id, BAOER_ID, obs_time,
              values["basin_rainfall_mm"], values["inflow_cms"],
              values["effective_storage"], values["outflow_cms"],
              values["water_level_m"], values["full_water_level_m"],
              values["storage_rate"], args.date))
        action = "INSERT"

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n  ✅ {action} 完成")


if __name__ == "__main__":
    main()
