#!/usr/bin/env python3
"""
=============================================================================
每日凌晨 04:00 派遣 coder agent 進行智慧健檢
=============================================================================
功能：
    向自己發送一條消息，觸發我派遣 coder subagent 執行智慧健檢

Cron 設定（每天 04:00）：
    0 4 * * * /usr/bin/python3 /home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai/src/data_pipeline/spawn_coder_for_review.py >> /home/sm245735/.openclaw/workspace/logs/coder_review.log 2>&1
"""

from pathlib import Path
from datetime import datetime

# 專案根目錄
PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()

# 在 log 目錄留下一個 flag，表示需要執行健檢
FLAG_FILE = PROJECT_DIR / "logs" / "coder_review_flag.txt"

ts = datetime.now().strftime('%Y-%m-%d %H:%M')
msg = f"[{ts}] 健檢標記已設定，請盡快派遣 coder agent"

print(msg)

with open(FLAG_FILE, 'w', encoding='utf-8') as f:
    f.write(f"{ts}: 需要執行每日智慧健檢\n")

print("完成")