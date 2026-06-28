"""聖嬰/反聖嬰 vs 寶二水庫蓄水率 — 改用日曆年對齊台灣豐枯期"""
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import statistics

load_dotenv(Path('/root/scip-water-scarcity-gis-ai/src/.env'))
eng = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# 用「日曆年」對齊 (1月-12月)
# 同時也試「水庫年」(前一年11月-當年10月) 對齊豐水期後段
ONI_CAL = {
    2013: "NE", 2014: "NE",
    2015: "EL", 2016: "EL",
    2017: "NE",
    2018: "LN",
    2019: "NE",
    2020: "LN", 2021: "LN", 2022: "LN",
    2023: "EL",
    2024: "NE", 2025: "NE",
}

# 抓 2013-2026 全部資料
with eng.connect() as c:
    rows = c.execute(text("""
        SELECT EXTRACT(YEAR FROM data_date)::int AS yr,
               COUNT(*) AS n,
               ROUND(AVG(storage_rate)::numeric, 2) AS mean_sr,
               ROUND(MIN(storage_rate)::numeric, 2) AS min_sr,
               ROUND(MAX(storage_rate)::numeric, 2) AS max_sr
        FROM fhy_reservoir_data
        WHERE reservoir_id = '23'
          AND data_date BETWEEN '2013-01-01' AND '2026-04-30'
        GROUP BY yr ORDER BY yr
    """)).fetchall()

CAT_CN = {"EL": "聖嬰", "LN": "反聖嬰", "NE": "中性"}

print("=" * 75)
print("分組 1: 日曆年 (Jan-Dec) 對齊")
print("=" * 75)
by_cat = {"EL": [], "LN": [], "NE": []}
for r in rows:
    yr, n, mean_sr, min_sr, max_sr = r
    cat = ONI_CAL.get(yr, "NE")
    if yr < 2014 or yr > 2025:
        continue  # 排除不完整的 2013 和 2026
    print(f"  {yr} ({CAT_CN[cat]:<4}) 平均 {mean_sr:>6}% min {min_sr:>6}% max {max_sr:>6}% n={n}")
    by_cat[cat].append((yr, float(mean_sr), n))

print()
for cat, label in [("EL", "聖嬰年"), ("LN", "反聖嬰年"), ("NE", "中性年")]:
    data = by_cat[cat]
    if not data:
        continue
    yrs = [d[0] for d in data]
    total_days = sum(d[2] for d in data)
    weighted = sum(d[1] * d[2] for d in data) / total_days
    print(f"  {label} ({len(yrs)} 年: {yrs}) 加權平均: {weighted:.2f}% (n={total_days})")

print()
print("=== 結果對照 (基準 = 中性年加權平均) ===")
neutral = sum(d[1] * d[2] for d in by_cat["NE"]) / sum(d[2] for d in by_cat["NE"])
for cat, label in [("EL", "聖嬰年"), ("LN", "反聖嬰年")]:
    avg = sum(d[1] * d[2] for d in by_cat[cat]) / sum(d[2] for d in by_cat[cat])
    delta = avg - neutral
    pct = (avg / neutral - 1) * 100
    arrow = "BUFF" if delta > 0 else "DEBUFF"
    print(f"  {label}: {avg:.2f}% vs 中性 {neutral:.2f}% → {delta:+.2f}% ({pct:+.1f}%) [{arrow}]")

# 第二組: 改用「水庫年」定義 (前一年11月-當年10月)
# ENSO 對台灣水庫的影響時序: 聖嬰當年 11-4 月少雨 → 隔年5-10月豐水期決定水庫豐枯
print()
print("=" * 75)
print("分組 2: 水庫年 (前年11月-當年10月) 對齊 — 颱風季決定蓄水率")
print("=" * 75)

with eng.connect() as c:
    rows2 = c.execute(text("""
        WITH labeled AS (
            SELECT data_date, storage_rate,
                CASE
                    WHEN data_date >= '2014-11-01' AND data_date < '2015-11-01' THEN 2015
                    WHEN data_date >= '2015-11-01' AND data_date < '2016-11-01' THEN 2016
                    WHEN data_date >= '2016-11-01' AND data_date < '2017-11-01' THEN 2017
                    WHEN data_date >= '2017-11-01' AND data_date < '2018-11-01' THEN 2018
                    WHEN data_date >= '2018-11-01' AND data_date < '2019-11-01' THEN 2019
                    WHEN data_date >= '2019-11-01' AND data_date < '2020-11-01' THEN 2020
                    WHEN data_date >= '2020-11-01' AND data_date < '2021-11-01' THEN 2021
                    WHEN data_date >= '2021-11-01' AND data_date < '2022-11-01' THEN 2022
                    WHEN data_date >= '2022-11-01' AND data_date < '2023-11-01' THEN 2023
                    WHEN data_date >= '2023-11-01' AND data_date < '2024-11-01' THEN 2024
                    WHEN data_date >= '2024-11-01' AND data_date < '2025-11-01' THEN 2025
                    WHEN data_date >= '2025-11-01' AND data_date < '2026-11-01' THEN 2026
                END AS res_yr
            FROM fhy_reservoir_data
            WHERE reservoir_id = '23'
              AND data_date BETWEEN '2014-11-01' AND '2026-04-30'
        )
        SELECT res_yr,
               COUNT(*) AS n,
               ROUND(AVG(storage_rate)::numeric, 2) AS mean_sr,
               ROUND(MIN(storage_rate)::numeric, 2) AS min_sr,
               ROUND(MAX(storage_rate)::numeric, 2) AS max_sr
        FROM labeled WHERE res_yr IS NOT NULL
        GROUP BY res_yr ORDER BY res_yr
    """)).fetchall()

# 水庫年分類: res_yr=Y 用 ONI_CAL[Y] (因為水庫年涵蓋前年11月-當年10月, ENSO 事件落在前年冬季)
# 實務上: 水庫年2016涵蓋2015-11~2016-10, 對應的 ENSO 是 2015-7~2016-6 (EL)
ONI_RES = {
    2015: "EL",  # 2014-11~2015-10 涵蓋 2014-7~2015-6 ONI (但 2014-2015 EL 是 2014 末才開始, 算 NE 偏 EL)
    2016: "EL", 2017: "EL",
    2018: "NE",  # 2017-7~2018-6 NE
    2019: "LN",
    2020: "NE",
    2021: "LN", 2022: "LN",
    2023: "LN",  # 2022-7~2023-6 LN 三連擊
    2024: "EL",
    2025: "EL",  # 2024-7~2025-6 EL → 轉 NE 中
    2026: "NE",
}

by_cat2 = {"EL": [], "LN": [], "NE": []}
for r in rows2:
    yr, n, mean_sr, min_sr, max_sr = r
    cat = ONI_RES.get(yr, "NE")
    print(f"  水庫年{yr} ({CAT_CN[cat]:<4}) 平均 {mean_sr:>6}% min {min_sr:>6}% max {max_sr:>6}% n={n}")
    by_cat2[cat].append((yr, float(mean_sr), n))

print()
for cat, label in [("EL", "聖嬰"), ("LN", "反聖嬰"), ("NE", "中性")]:
    data = by_cat2[cat]
    if not data:
        continue
    yrs = [d[0] for d in data]
    total_days = sum(d[2] for d in data)
    weighted = sum(d[1] * d[2] for d in data) / total_days
    print(f"  {label} ({len(yrs)} 年: {yrs}) 加權平均: {weighted:.2f}% (n={total_days})")

print()
neutral2 = sum(d[1] * d[2] for d in by_cat2["NE"]) / sum(d[2] for d in by_cat2["NE"])
for cat, label in [("EL", "聖嬰"), ("LN", "反聖嬰")]:
    avg = sum(d[1] * d[2] for d in by_cat2[cat]) / sum(d[2] for d in by_cat2[cat])
    delta = avg - neutral2
    pct = (avg / neutral2 - 1) * 100
    arrow = "BUFF" if delta > 0 else "DEBUFF"
    print(f"  {label}: {avg:.2f}% vs 中性 {neutral2:.2f}% → {delta:+.2f}% ({pct:+.1f}%) [{arrow}]")