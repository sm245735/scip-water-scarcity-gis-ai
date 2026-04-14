#!/usr/bin/env python3
"""
水庫歷史日資料收集_每日.py（Comparison API）
資料來源：水利署 fhy.wra.gov.tw Disaster Comparison API
URL: POST https://fhy.wra.gov.tw/Disaster/api/ReservoirHistoryApi/GetComparision
Params: ST_NO=id1,id2,... (最多9個), StartDate=YYYY/MM/DD, EndDate=YYYY/MM/DD

重要發現：
- API 只在 StartDate 和 EndDate 范圍的「頭尾」各回傳一筆記錄
- 策略：每次查詢區間=1天，逐日查詢，即可取得完整日資料
- API 似乎只提供近期滾動資料（約2週以內），不包含完整的歷史記錄

使用方式（每日 Cron）：
  python 水庫歷史日資料收集_每日.py

指定日期範圍：
  python 水庫歷史日資料收集_每日.py --start 2026/01/02 --end 2026/04/14

預計時間：
  - 20個水庫 × 1天 = 20次查詢（每日）
  - 每批 9 個水庫 = 3批請求
"""

import requests
import pandas as pd
import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta, date
from datetime import date as date_type

# === 組態 ===
API_URL = "https://fhy.wra.gov.tw/Disaster/api/ReservoirHistoryApi/GetComparision"
DATA_DIR = "/app/data"
OUTPUT_CSV = os.path.join(DATA_DIR, "水庫歷史日資料_ComparisonAPI.csv")
PROGRESS_FILE = os.path.join(DATA_DIR, "水庫收集進度.json")

# Comparison API 的 20 個水庫（ST_NO: 名稱）
RESERVOIRS = {
    "10201": "石門水庫",
    "10205": "翡翠水庫",
    "10405": "寶山第二水庫",  # 主要！
    "10501": "永和山水庫",
    "10601": "明德水庫",
    "20101": "鯉魚潭水庫",
    "20201": "德勒水庫",
    "20202": "石岡壩",
    "20501": "霧社水庫",
    "20502": "日月潭水庫",
    "20503": "集集攔河堰",
    "20509": "湖山水庫",
    "30301": "仁義潭水庫",
    "30401": "白河水庫",
    "30501": "烏山頭水庫",
    "30502": "曾文水庫",
    "30503": "南化水庫",
    "30802": "阿公店水庫",
    "30901": "高屏溪攔河堰",
    "31201": "牡丹水庫",
}

# 分組（每組最多 9 個）
def chunk_dict(d, size=9):
    items = list(d.items())
    return [dict(items[i:i+size]) for i in range(0, len(items), size)]

RESERVOIR_GROUPS = chunk_dict(RESERVOIRS, 9)


def load_progress():
    """載入進度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"last_date": None, "completed_days": 0, "total_records": 0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def fetch_day(group_ids, day_date, max_retries=3):
    """抓取單日單組水庫資料"""
    date_str = day_date.strftime("%Y/%m/%d")
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL,
                data={
                    "ST_NO": ",".join(group_ids.keys()),
                    "StartDate": date_str,
                    "EndDate": date_str,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            if resp.status_code == 200 and resp.text.strip() and "Request Rejected" not in resp.text:
                return resp.json()
            time.sleep(1)
        except Exception as e:
            time.sleep(2)
    return []


def main():
    parser = argparse.ArgumentParser(description="水庫歷史日資料收集（每日模式）")
    parser.add_argument("--start", default=None, help="開始日期 YYYY/MM/DD（預設：昨天）")
    parser.add_argument("--end", default=None, help="結束日期 YYYY/MM/DD（預設：昨天）")
    args = parser.parse_args()

    # 預設：收集昨天的資料
    yesterday = (datetime.now() - timedelta(days=1)).date()
    if args.start is None:
        start_date = yesterday
    else:
        start_date = datetime.strptime(args.start, "%Y/%m/%d").date()

    if args.end is None:
        end_date = yesterday
    else:
        end_date = datetime.strptime(args.end, "%Y/%m/%d").date()

    print("=" * 60)
    print("水庫歷史日資料收集（每日模式）")
    print(f"日期範圍：{start_date} ~ {end_date}")
    print(f"水庫數量：{len(RESERVOIRS)}（分 {len(RESERVOIR_GROUPS)} 組，每組最多9個）")
    print("=" * 60)

    # 讀取現有資料（如果有）
    existing_df = None
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
        print(f"已讀取現有資料：{len(existing_df)} 筆")
        if "date" in existing_df.columns:
            existing_dates = set(pd.to_datetime(existing_df["date"]).dt.date)
            print(f"現有日期數：{len(existing_dates)} 天")
    else:
        existing_df = pd.DataFrame()
        existing_dates = set()

    # 生成日期列表
    all_dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date not in existing_dates:
            all_dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"待收集天數：{len(all_dates)} 天")
    if not all_dates:
        print("所有日期都已收集完成！")
        return

    all_records = []
    total_queries = len(all_dates) * len(RESERVOIR_GROUPS)
    query_count = 0
    no_data_count = 0

    for d in all_dates:
        day_records = []
        for group in RESERVOIR_GROUPS:
            query_count += 1
            data = fetch_day(group, d)
            for item in data:
                st_no = str(item.get("ST_NO", ""))
                if st_no in group:
                    day_records.append({
                        "st_no": st_no,
                        "reservoir_name": item.get("RES_NAME", group.get(st_no, "?")),
                        "date": item.get("DATE", "")[:10],
                        "water_level_m": item.get("WaterLine"),
                        "capacity_萬噸": item.get("Capacity"),
                        "capacity_rate": item.get("CapacityRate"),
                        "inflow_cms": item.get("Inflow_Total"),
                        "outflow_cms": item.get("Outflow_Total"),
                        "basin_rainfall_mm": item.get("Basin_Rain"),
                        "use_outflow": item.get("UseOutflow"),
                        "spillway_outflow": item.get("SpillwayOutflow"),
                        "api_source": "ComparisonAPI",
                        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })

        all_records.extend(day_records)
        if day_records:
            no_data_count += 1

        # 每20天顯示進度
        if query_count % (len(RESERVOIR_GROUPS) * 20) == 0:
            print(f"  [{d}] 累計 {len(all_records)} 筆...")

        # 避免請求太密集
        time.sleep(0.5)

    if not all_records:
        print("沒有取得任何新資料（API 可能無資料或已達有效期）")
        return

    # 合併新舊資料
    new_df = pd.DataFrame(all_records)
    combined = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None and not existing_df.empty else new_df

    # 去除重複
    before = len(combined)
    combined = combined.drop_duplicates(subset=["st_no", "date"], keep="last")
    after = len(combined)
    print(f"去除重複：{before} → {after} 筆")

    # 排序並儲存
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["st_no", "date"])
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已匯出：{OUTPUT_CSV}")

    # 摘要
    print(f"\n收集摘要：")
    summary = combined.groupby(["st_no", "reservoir_name"]).agg(
        筆數=("date", "count"),
        起始日=("date", "min"),
        結束日=("date", "max")
    ).reset_index()
    for _, row in summary.iterrows():
        print(f"  {row['st_no']} {row['reservoir_name']}: {row['筆數']} 筆 ({row['起始日']} ~ {row['結束日']})")

    print(f"\n總筆數：{len(combined)}")


if __name__ == "__main__":
    main()