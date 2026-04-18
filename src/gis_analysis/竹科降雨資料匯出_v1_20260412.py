#!/usr/bin/env python3
"""
竹科每日降雨量匯出
來源：tccip_daily_rainfall.csv（2019-2023，TCCIP 觀測日資料）
用途：LSTM 訓練用的每日降雨特徵

邏輯：
1. 讀取 tccip_daily_rainfall.csv（已有 竹科/中科/南科 每日降雨量）
2. 只取「竹科」欄位（因為寶山水庫供應竹科）
3. 匯出成 CSV，供後續與水庫日蓄水量 merge
"""

import pandas as pd
import os
import sys

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from path_utils import TCCIP_DATA_DIR, DATA_DIR

# 讀取 TCCIP 資料
csv_path = str(TCCIP_DATA_DIR / "tccip_daily_rainfall.csv")
df = pd.read_csv(csv_path)

print(f"原始資料：{len(df)} 筆")
print(f"欄位：{df.columns.tolist()}")
print(f"日期範圍：{df['date'].min()} ~ {df['date'].max()}")

# 只取竹科（新竹科學園區）
df_zhuke = df[df['park'] == '竹科'].copy()
print(f"\n竹科資料：{len(df_zhuke)} 筆")

# 整理格式
df_zhuke = df_zhuke[['date', 'rainfall_mm']].copy()
df_zhuke.columns = ['date', 'rainfall_mm_zhuke']
df_zhuke['date'] = pd.to_datetime(df_zhuke['date'])
df_zhuke = df_zhuke.sort_values('date').reset_index(drop=True)

print(f"竹科日期範圍：{df_zhuke['date'].min().date()} ~ {df_zhuke['date'].max().date()}")

# 輸出
out_path = str(DATA_DIR / '竹科_每日降雨量_2019_2023.csv')
df_zhuke.to_csv(out_path, index=False)
print(f"\n✅ 已匯出：{out_path}")
print(f"   筆數：{len(df_zhuke)}")
print(f"\n前5筆：")
print(df_zhuke.head())
print(f"\n統計：")
print(df_zhuke['rainfall_mm_zhuke'].describe())
