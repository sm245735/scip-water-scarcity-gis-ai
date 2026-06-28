"""
匯入 /root/scip-water-scarcity-gis-ai/data/CWA_Data/*.csv 到 public.codis_weatherdata_v2

Schema（仿 codis_weatherdata v1 的命名，但只保留 ML 用的 6 欄）:
  stno   varchar(50)         -- CWA 測站代號 (例: C0D580)
  date   timestamptz          -- 觀測日；儲存為 LST 當地 00:00 = UTC 前一天 16:00
                              --   這樣 SQL 端 group by date::date 仍是當地 LST 日期
  PP01   real                 -- 降水量 (mm)         ; -9.8 / * / 負數 -> NULL
  TX01   real                 -- 平均氣溫 (°C)
  TX02   real                 -- 最高氣溫 (°C)
  RH01   real                 -- 相對溼度 (%)
  WD01   real                 -- 平均風速 (m/s)
  PS01   real                 -- 測站平均氣壓 (hPa)
  id     int GENERATED ALWAYS AS IDENTITY
  crtdt  timestamptz DEFAULT now()
  UNIQUE(stno, date)

設計原則:
- ETL 只做「欄位對齊 + 時區統一 + 異常碼轉 NULL」，不做任何跨站/時序補值
- 補值在 train_end_to_end.py 階段決定（cross-station avg 或 time interpolate）
- 重跑策略: ON CONFLICT (stno, date) DO UPDATE（idempotent）
"""

from __future__ import annotations
import os
import re
import sys
import csv
import calendar
from pathlib import Path
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ==========================================
# 1. 連線
# ==========================================
ROOT = Path("/root/scip-water-scarcity-gis-ai")
load_dotenv(ROOT / "src" / ".env")

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
assert all([DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME]), "DB env 不完整"

ENGINE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ==========================================
# 2. CWA_Data 來源
# ==========================================
CWA_DIR = ROOT / "data" / "CWA_Data"

# 中文表頭 -> v2 欄位 的映射（同時處理 20 欄、30 欄、44 欄、自動雨量站 2 欄）
COL_MAP = {
    "降水量(mm)":         "PP01",
    "氣溫(℃)":            "TX01",
    "最高氣溫(℃)":        "TX02",
    "相對溼度(%)":        "RH01",
    "風速(m/s)":          "WD01",   # 平均風速
    "測站氣壓(hPa)":      "PS01",
}

# ==========================================
# 3. 建表（IF NOT EXISTS）
# ==========================================
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS public.codis_weatherdata_v2 (
    stno   varchar(50)  NOT NULL,
    date   timestamptz  NOT NULL,
    PP01   real,
    TX01   real,
    TX02   real,
    RH01   real,
    WD01   real,
    PS01   real,
    id     int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    crtdt  timestamptz  NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_v2_stno_date
    ON public.codis_weatherdata_v2 (stno, date);
CREATE INDEX IF NOT EXISTS ix_v2_date ON public.codis_weatherdata_v2 (date);
"""

UPSERT_SQL = text("""
INSERT INTO public.codis_weatherdata_v2
    (stno, date, PP01, TX01, TX02, RH01, WD01, PS01)
VALUES
    (:stno, :date, :PP01, :TX01, :TX02, :RH01, :WD01, :PS01)
ON CONFLICT (stno, date) DO UPDATE SET
    PP01 = EXCLUDED.PP01,
    TX01 = EXCLUDED.TX01,
    TX02 = EXCLUDED.TX02,
    RH01 = EXCLUDED.RH01,
    WD01 = EXCLUDED.WD01,
    PS01 = EXCLUDED.PS01
""")

# ==========================================
# 4. 工具
# ==========================================
FILE_RE = re.compile(r"^(?P<name>.+?)_(?P<stno>[A-Z0-9]+)_(?P<yyyymm>\d{6})\.csv$")

def to_float_or_none(tok: str) -> float | None:
    """把 CWA CSV 儲存格轉成 float；異常碼 → None。"""
    if tok is None:
        return None
    s = tok.strip()
    if s == "" or s == "T" or s == "X" or s == "..." or s == "NR" or s == "/" or s == "-":
        return None
    # CWA 異常值前綴 '*'
    if s.startswith("*"):
        s = s[1:]
    try:
        v = float(s)
    except ValueError:
        return None
    # CWA 雨量站的跡 -9.8 = 軌跡缺失；其他負數也算異常
    # 不在這裡 hard-rule「負數=異常」: 風速/氣壓理論上不會負,
    # 但保留保守 — 用 None 表示「資料庫沒拿到」, 讓 train 階段決定
    if v < 0:
        return None
    return v

import calendar

def parse_obs_date(yyyymm: str, day_field: str, stno: str) -> datetime | None:
    """
    CWA CSV 第一欄 ObsTime 是 '01'..'31' (LST 當地日期).
    v1 的儲存 convention: 'date' = LST 00:00 用 UTC 表示 = 前一天 16:00:00+00.
    沿用相同 convention 才能跟 fhy_reservoir_data merge.
    """
    try:
        day = int(day_field.strip())
    except (TypeError, ValueError):
        return None
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])
    # CWA 偶爾會把不存在的日期(如 2/30)寫成 "30" 之類的 → 防呆
    last_day = calendar.monthrange(year, month)[1]
    if day < 1 or day > last_day:
        return None
    # LST 當地 00:00 = UTC (前一天 16:00:00)
    local_midnight = datetime(year, month, day)
    utc_dt = local_midnight - timedelta(hours=8)
    return utc_dt.replace(tzinfo=timezone.utc)


def iter_files():
    for f in sorted(CWA_DIR.iterdir()):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        yield m.group("stno"), m.group("yyyymm"), f


# ==========================================
# 5. 主流程
# ==========================================
def main():
    print(f"讀取來源: {CWA_DIR}")
    print(f"目標表:   public.codis_weatherdata_v2")

    eng = create_engine(ENGINE_URL, future=True)
    with eng.begin() as conn:
        conn.execute(text(CREATE_SQL))
        print("[schema] codis_weatherdata_v2 已就緒")

    # 預先 group files by stno（debug 用）
    files_by_stn: dict[str, list] = {}
    for stno, yyyymm, f in iter_files():
        files_by_stn.setdefault(stno, []).append((yyyymm, f))
    print(f"[scan] 共 {sum(len(v) for v in files_by_stn.values())} 個檔案，"
          f"{len(files_by_stn)} 個站台: {sorted(files_by_stn)}")
    for stno in sorted(files_by_stn):
        ym_first = min(y for y, _ in files_by_stn[stno])
        ym_last = max(y for y, _ in files_by_stn[stno])
        print(f"  - {stno}: {len(files_by_stn[stno])} files, "
              f"{ym_first[:4]}-{ym_first[4:]} ~ {ym_last[:4]}-{ym_last[4:]}")

    # 逐檔處理
    total_rows = 0
    total_inserted = 0
    skipped_header_skew = 0
    with eng.begin() as conn:
        for stno, yyyymm, fpath in iter_files():
            with open(fpath, "r", encoding="utf-8-sig", newline="") as fp:
                rdr = csv.reader(fp)
                try:
                    hdr_cn = next(rdr)
                    hdr_en = next(rdr)
                except StopIteration:
                    print(f"  [skip] {fpath.name} 檔案太短")
                    continue

                # 把中文表頭 map 到 v2 欄位；同時找 ObsTime (第一欄的英文名)
                col_to_field: dict[str, int] = {}
                time_col = None
                for idx, cn in enumerate(hdr_cn):
                    en = hdr_en[idx] if idx < len(hdr_en) else ""
                    if idx == 0 and en == "ObsTime":
                        time_col = idx
                        continue
                    if cn in COL_MAP:
                        col_to_field[COL_MAP[cn]] = idx

                if time_col is None:
                    print(f"  [skip] {fpath.name} 找不到 ObsTime 欄位")
                    skipped_header_skew += 1
                    continue

                # 雨量站只有 ObsTime + Precp → 其他欄位不會進來, 自動 NULL
                # 沒有任何錯誤，直接處理
                batch = []
                for row in rdr:
                    if not row or len(row) <= time_col:
                        continue
                    day_field = row[time_col]
                    date_utc = parse_obs_date(yyyymm, day_field, stno)
                    if date_utc is None:
                        continue
                    record = {
                        "stno": stno,
                        "date": date_utc,
                        "PP01": None, "TX01": None, "TX02": None,
                        "RH01": None, "WD01": None, "PS01": None,
                    }
                    for field, idx in col_to_field.items():
                        if idx < len(row):
                            record[field] = to_float_or_none(row[idx])
                    batch.append(record)
                    total_rows += 1

                if batch:
                    conn.execute(UPSERT_SQL, batch)
                    total_inserted += len(batch)

    print(f"\n[done] 讀入 {total_rows} 列，UPSERT {total_inserted} 列")
    print(f"[done] 跳過 header 不認得的檔案 {skipped_header_skew} 個")


if __name__ == "__main__":
    sys.exit(main())