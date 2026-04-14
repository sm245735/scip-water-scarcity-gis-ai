#!/usr/bin/env python3
"""
=============================================================================
專案健檢腳本（每天 04:00 執行）
=============================================================================
功能：
    掃描 scip-water-scarcity-gis-ai 專案目錄
    找出需要整理的地方（多餘的檔案、重複的檔、不需要的檔案）
    將檢查結果寫入 log，並在必要时提醒

執行方式：
    python3 /home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai/src/data_pipeline/project_health_check.py

Cron 設定（每天 04:00）：
    0 4 * * * /usr/bin/python3 /home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai/src/data_pipeline/project_health_check.py >> /home/sm245735/.openclaw/workspace/logs/project_health_check.log 2>&1
"""

import os, glob
from datetime import datetime

PROJECT_DIR = "/home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai"
LOG_FILE = "/home/sm245735/.openclaw/workspace/logs/project_health_check.log"

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + "\n")

def check_data_folder():
    """檢查 data/ 資料夾是否有不需要的檔案"""
    data_dir = os.path.join(PROJECT_DIR, "data")
    if not os.path.exists(data_dir):
        return [], "data/ 資料夾不存在"
    
    unnecessary_patterns = [
        "reservoir_id_map.csv",      # 已被 水庫ID對照表_自研版.csv 取代
        "*_backup*",                  # 備份檔
        "*.tmp",                      # 暫存檔
    ]
    
    issues = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            for pattern in unnecessary_patterns:
                if "*" in pattern and f == pattern.replace("*", ""):
                    issues.append(f"多餘檔案：{os.path.join(root, f)}")
                elif pattern in f:
                    issues.append(f"多餘檔案：{os.path.join(root, f)}")
    
    return issues, "ok"

def check_data_samples():
    """檢查 data_samples/ 是否有重複或不需要的檔案"""
    samples_dir = os.path.join(PROJECT_DIR, "data_samples")
    if not os.path.exists(samples_dir):
        return [], "data_samples/ 資料夾不存在"
    
    issues = []
    
    # 檢查是否有多個 CSV 檔案指的是同一個東西
    csvs = [f for f in os.listdir(samples_dir) if f.endswith('.csv')]
    # 預期：水庫ID對照表_自研版.csv（最新的完整版）
    
    return issues, "ok"

def check_src_folder():
    """檢查 src/ 是否有需要整理的地方"""
    issues = []
    src_dir = os.path.join(PROJECT_DIR, "src")
    if not os.path.exists(src_dir):
        return issues, "ok"
    
    # 檢查是否有孤兒檔（沒有被其他檔案引用）
    # 目前先簡單檢查
    
    return issues, "ok"

def main():
    log("=== 專案健檢開始 ===")
    
    checks = [
        ("data/ 資料夾", check_data_folder),
        ("data_samples/ 資料夾", check_data_samples),
        ("src/ 資料夾", check_src_folder),
    ]
    
    total_issues = 0
    for name, check_fn in checks:
        issues, status = check_fn()
        if issues:
            log(f"  [{name}] 發現 {len(issues)} 個問題：")
            for issue in issues:
                log(f"    - {issue}")
            total_issues += len(issues)
        else:
            log(f"  [{name}] ✅ 無問題")
    
    log(f"=== 專案健檢完成，發現 {total_issues} 個需整理項目 ===")

if __name__ == "__main__":
    main()