#!/usr/bin/env python3
"""
收集_2026增量.py
收集 2026-01-02 ~ 2026-04-14 的水庫日資料（Comparison API）
用途：填補 reservoir_daily 的 gap

使用方式：
  python 收集_2026增量.py
"""

import requests
import pandas as pd
import os
import sys
import time
import json
from datetime import datetime, timedelta

# === 組態 ===
API_URL = "https://fhy.wra.gov.tw/Disaster/api/ReservoirHistoryApi/GetComparision"
DATA_DIR = "/app/data"
OUTPUT_CSV = os.path.join(DATA_DIR, "水庫歷史日資料_ComparisonAPI_2026.csv")

# Comparison API 的水庫 ID 對照（ST_NO 兩碼）
RESERVOIRS = {
    "01": "石門水庫",
    "02": "翡翠水庫",
    "03": "寶山水庫",
    "04": "寶山第二水庫",
    "05": "永和山湖水庫",
    "06": "鯉魚潭水庫",
    "08": "大明湖水庫",
    "10": "谷關水庫",
    "11": "珈居高義水庫",
    "12": "湖山水庫",
    "13": "蘭潭水庫",
    "14": "仁義潭水庫",
    "15": "阿更加水庫",
    "16": "士林水庫",
    "17": "榮華水庫",
    "18": "老天山村...",
    "19": "瑞峰水庫",
    "21": "集集攔河堰",
    "23": "Civil...",
    "25": "牡丹水庫",
}

# 分組（每組最多 9 個）
def chunk_dict(d, size=9):
    items = list(d.items())
    return [dict(items[i:i+size]) for i in range(0, len(items), size)]

RESERVOIR_GROUPS = chunk_dict(RESERVOIRS, 9)

START_DATE = datetime(2026, 1, 2)
END_DATE = datetime(2026, 4, 14)


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
    print("=" * 60)
    print("水庫歷史日資料收集（2026 增量）")
    print(f"日期範圍：{START_DATE.date()} ~ {END_DATE.date()}")
    print(f"水庫數量：{len(RESERVOIRS)}（分 {len(RESERVOIR_GROUPS)} 組，每組最多9個）")
    print("=" * 60)

    # 讀取現有資料（去重）
    existing_df = None
    existing_dates = set()
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
        if "date" in existing_df.columns and len(existing_df) > 0:
            existing_dates = set(pd.to_datetime(existing_df["date"]).dt.date)
            print(f"已讀取現有資料：{len(existing_df)} 筆，{len(existing_dates)} 天")

    # 生成日期列表
    all_dates = []
    current = START_DATE
    while current <= END_DATE:
        if current.date() not in existing_dates:
            all_dates.append(current)
        current += timedelta(days=1)

    print(f"待收集天數：{len(all_dates)} 天")
    if not all_dates:
        print("所有日期都已收集完成！")
        return

    all_records = []
    total_queries = len(all_dates) * len(RESERVOIR_GROUPS)
    query_count = 0
    collected_days = 0

    for day in all_dates:
        day_records = []
        for group in RESERVOIR_GROUPS:
            query_count += 1
            data = fetch_day(group, day)
            for item in data:
                st_no = str(item.get("ST_NO", "")).zfill(2)
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
                        "api_source": "ComparisonAPI_2026",
                        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
            # 每次請求間隔
            time.sleep(0.5)

        if day_records:
            collected_days += 1

        # 每20天顯示進度
        if query_count % (len(RESERVOIR_GROUPS) * 20) == 0:
            print(f"  進度：{query_count}/{total_queries} ({100*query_count/total_queries:.1f}%)，已收集 {len(all_records)} 筆...")

    print(f"\n完成：收集了 {collected_days} 天，共 {len(all_records)} 筆")

    if not all_records:
        print("沒有取得任何新資料")
        return

    # 合併新舊資料
    new_df = pd.DataFrame(all_records)
    combined = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None and not existing_df.empty else new_df

    # 去重
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