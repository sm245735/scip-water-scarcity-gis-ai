#!/usr/bin/env python3
"""
水庫日資料收集腳本
資料來源：水利署 opendata API（即時每小時更新）
用途：每日定時收集，append 到 CSV 檔案，逐步建立歷史資料庫

使用方式：
  python 水庫日資料收集_v1_20260412.py

建議使用 cron job 每日 12:00 執行：
  0 12 * * * docker exec thesis_python_dev python /app/src/data_pipeline/水庫日資料收集_v1_20260412.py
"""

import requests
import pandas as pd
import os
import json
from datetime import datetime

# === 組態 ===
API_URL = "https://opendata.wra.gov.tw/api/v2/2be9044c-6e44-4856-aad5-dd108c2e6679"
DATA_DIR = "/app/data"
OUTPUT_CSV = os.path.join(DATA_DIR, "水庫日資料_即時收集.csv")
ID_MAP_CSV = os.path.join(DATA_DIR, "水庫ID對照表.csv")

# 水庫 ID → 名稱 對照表（已確認的）
RESERVOIR_NAMES = {
    "10201": "寶山水庫",
    "10202": "寶山第二水庫",
    "10203": "石門水庫",
    "10204": "永和山水庫",
    "10211": "明德水庫",
    "10212": "鯉魚潭水庫",
    "10405": "石岡壩",
    "10501": "湖山水庫",
    "10503": "?,?",  # 待確認
    "10601": "曾文水庫",
    "20101": "翡翠水庫",
    "20201": "?,?",  # 待確認
    "20202": "?,?",  # 待確認
    "20501": "?,?",  # 待確認
    "20502": "?,?",  # 待確認
    "20503": "?,?",  # 待確認
    "20509": "?,?",  # 待確認
    "30301": "?,?",  # 待確認
    "30302": "?,?",  # 待確認
    "30501": "?,?",  # 待確認
    "30502": "?,?",  # 待確認
    "30503": "?,?",  # 待確認
    "30504": "?,?",  # 待確認
    "30802": "?,?",  # 待確認
    "30901": "?,?",  # 待確認
    "31002": "?,?",  # 待確認
    "31201": "?,?",  # 待確認
    "31301": "?,?",  # 待確認
    "50102": "?,?",  # 待確認
    "50103": "?,?",  # 待確認
    "50104": "?,?",  # 待確認
    "50105": "?,?",  # 待確認
    "50106": "?,?",  # 待確認
    "50108": "?,?",  # 待確認
    "50109": "?,?",  # 待確認
    "50201": "?,?",  # 待確認
    "50202": "?,?",  # 待確認
    "50203": "?,?",  # 待確認
    "50204": "?,?",  # 待確認
    "50205": "?,?",  # 待確認
    "50206": "?,?",  # 待確認
    "50207": "?,?",  # 待確認
    "50208": "?,?",  # 待確認
    "50209": "?,?",  # 待確認
    "50210": "?,?",  # 待確認
    "50212": "?,?",  # 待確認
    "50213": "?,?",  # 待確認
    "50214": "?,?",  # 待確認
    # 上坪堰相關 ID（待確認）
    "10205": "上坪堰",
}

# 感興趣的水庫（新竹相關 + 全台重要水庫）
TARGET_RESERVOIRS = ["10201", "10203", "10204", "10211", "10212"]


def fetch_all_reservoirs():
    """從 API 抓取所有水庫的最新資料"""
    print(f"[{datetime.now()}] 正在抓取水庫 API...")
    try:
        resp = requests.get(API_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        print(f"  API 總筆數：{len(data)}")
        return data
    except Exception as e:
        print(f"  ❌ API 錯誤：{e}")
        return []


def fetch_single_reservoir(reservoir_id):
    """抓取特定水庫的資料"""
    try:
        resp = requests.get(
            API_URL,
            params={"reservoiridentifier": reservoir_id},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ❌ 抓取 {reservoir_id} 失敗：{e}")
        return []


def process_data(raw_data):
    """將 API 原始資料轉換乾淨的 DataFrame"""
    records = []
    for r in raw_data:
        # 跳過無效資料
        if not r.get("effectivewaterstoragecapacity"):
            continue

        rid = r.get("reservoiridentifier", "")
        records.append({
            "reservoir_id": rid,
            "reservoir_name": RESERVOIR_NAMES.get(rid, f"未知({rid})"),
            "observation_time": r.get("observationtime", ""),
            "effective_storage_萬噸": float(r.get("effectivewaterstoragecapacity", 0) or 0),
            "water_level_m": float(r.get("waterlevel", 0) or 0),
            "inflow_cms": float(r.get("inflowdischarge", 0) or 0) if r.get("inflowdischarge") else None,
            "total_outflow_cms": float(r.get("totaloutflow", 0) or 0) if r.get("totaloutflow") else None,
            "catchment_rainfall_mm": float(r.get("accumulaterainfallincatchment", 0) or 0) if r.get("accumulaterainfallincatchment") else None,
            "water_draw_cms": float(r.get("waterdraw", 0) or 0) if r.get("waterdraw") else None,
            "api_collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["observation_time"] = pd.to_datetime(df["observation_time"])
    df = df.sort_values(["reservoir_id", "observation_time"])
    return df


def append_to_csv(df, csv_path):
    """將新資料 append 到現有 CSV"""
    if df.empty:
        print("  ⚠️ 無有效資料，跳過寫入")
        return

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path, parse_dates=["observation_time"])
        # 合併：只加新的（按 reservoir_id + observation_time 去重）
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["reservoir_id", "observation_time"],
            keep="last"
        )
    else:
        combined = df

    combined = combined.sort_values(["reservoir_id", "observation_time"])
    combined.to_csv(csv_path, index=False)
    print(f"  ✅ 已寫入/更新：{csv_path}")
    print(f"     總筆數：{len(combined)}（+{len(df)} 筆新資料）")


def main():
    print("=" * 60)
    print("水庫日資料收集")
    print("=" * 60)

    # 1. 抓所有水庫
    all_data = fetch_all_reservoirs()

    if not all_data:
        print("❌ 無法取得 API 資料，請檢查網路或 API 狀態")
        return

    # 2. 只保留目標水庫（可節省儲存空間，也可改為全量收集）
    target_data = [r for r in all_data if r.get("reservoiridentifier") in TARGET_RESERVOIRS]
    print(f"  目標水庫筆數：{len(target_data)}")

    # 3. 處理資料
    df = process_data(target_data)

    # 4. Append 到 CSV
    append_to_csv(df, OUTPUT_CSV)

    # 5. 顯示摘要
    if not df.empty:
        print("\n今日收集摘要：")
        print(df.groupby("reservoir_name")["effective_storage_萬噸"].last())

    print(f"\n✅ 收集完成！({datetime.now()})")


if __name__ == "__main__":
    main()
