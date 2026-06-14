#!/usr/bin/env python3
"""
=============================================================================
水庫 Statistics.aspx 歷史回溯收集（2014-2015）— 單日測試版
=============================================================================

用途：從 WRA Statistics.aspx 抓 2014-2015 歷史水庫水情，匯入 fhy_reservoir_data
      （舊版每日排程是 src/data_pipeline/水庫Statistics_每日收集_host.py）

與 daily 版的差異：
  - 目標表：public.fhy_reservoir_data（不是舊 reservoir_daily）
  - DB：gis_db / steveyang（不是 thesis_analysis / sm245735）
  - 名稱對映：public.fhy_reservoir.reservoir_name（已建好的 112 筆）
  - 鎖定時段：06:00（早上 6 點）
  - 跳過已存在：INSERT ... ON CONFLICT (reservoir_id, data_date) DO NOTHING
  - Retry + 跳過該日（水利署擋你時不整批掛）

執行：
  # 單日測試（驗證流程通）
  /root/scip-water-scarcity-gis-ai/venv/bin/python \\
      src/data_pipeline/水庫Statistics_回溯收集_2014-2015.py \\
      --start 2014-01-01 --end 2014-01-01

  # 跑整段（2014-01-01 ~ 2015-12-31，730 天）
  nohup ... >> logs/backfill_2014_2015.log 2>&1 &
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import date as date_type, datetime, timedelta
import psycopg2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# ---------- DB 連線（這台 server 的設定）----------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = "gis_db"
DB_USER = "steveyang"
DB_PASS = os.environ.get("DB_PASSWORD", "")
if not DB_PASS:
    # 從 docker inspect 撈出來（如果 DB_PASSWORD env 沒設的話）
    import subprocess
    r = subprocess.run(
        ["docker", "inspect", "postgres", "--format",
         "{{range .Config.Env}}{{println .}}{{end}}"],
        capture_output=True, text=True
    )
    for line in r.stdout.splitlines():
        if line.startswith("POSTGRES_PASSWORD="):
            DB_PASS = line.split("=", 1)[1]
            break

# ---------- 路徑 ----------
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 設定 logger ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backfill_2014_2015.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ---------- 工具函數 ----------
def get_driver():
    """用 selenium/standalone-chrome container 跑（避免 snap chromium 的 DevToolsActivePort 問題）"""
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    # 連到 docker selenium container
    return webdriver.Remote(
        command_executor="http://localhost:4444",
        options=options
    )


def parse_num(val, remove_comma=False, remove_percent=False):
    if not val or val.strip() in ('--', '', '-'):
        return None
    v = val.strip()
    if remove_comma: v = v.replace(',', '')
    if remove_percent: v = v.replace('%', '').strip()
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def fetch_one_day(driver, target_date: date_type, name_to_id: dict,
                  conn, max_retries: int = 3) -> tuple:
    """
    抓一天的資料，INSERT 進 fhy_reservoir_data
    回傳 (inserted_count, skipped_reason)
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"  導航 {target_date} (attempt {attempt}/{max_retries})")
            driver.get("https://fhy.wra.gov.tw/ReservoirPage_2011/Statistics.aspx")
            time.sleep(5)

            # 切換到「全部」水庫
            driver.execute_script(
                "var s=document.getElementById('ctl00_cphMain_cboSearch');"
                "if(s){s.value='全部';s.dispatchEvent(new Event('change',{bubbles:true}));}"
            )
            time.sleep(3)

            # 選年/月/日/時=0（06:00）
            Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboYear")
                   ).select_by_value(str(target_date.year))
            time.sleep(1)
            Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboMonth")
                   ).select_by_value(str(target_date.month))
            time.sleep(1)
            Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboDay")
                   ).select_by_value(str(target_date.day))
            time.sleep(1)
            Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboHour")
                   ).select_by_value("0")  # 0 = 00 點那組，對應 06:00 的小時段
            time.sleep(1)

            # 觸發查詢
            driver.execute_script("document.getElementById('ctl00_cphMain_btnQuery').click();")
            time.sleep(5)

            # 撈表格
            gvlist = driver.find_element(By.ID, "ctl00_cphMain_gvList")
            rows = gvlist.find_elements(By.TAG_NAME, "tr")
            data_rows = [r for r in rows if r.find_elements(By.TAG_NAME, "td")]

            if not data_rows:
                logger.warning(f"  {target_date} 沒有回傳資料（可能該日無記錄）")
                return 0, "no_data"

            cur = conn.cursor()
            inserted = 0
            # 撈當前 max(id)，整批遞增（避免主鍵衝突）
            cur.execute("SELECT COALESCE(MAX(id), 0) FROM public.fhy_reservoir_data")
            start_id = cur.fetchone()[0]
            next_id = start_id + 1

            for row in data_rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 10:
                    continue
                name = cells[0].text.strip()
                if name not in name_to_id:
                    # 略過不在 fhy_reservoir 表內的水庫
                    continue
                rid = name_to_id[name]
                # 時段鎖定 06:00
                obs_time = datetime.combine(target_date, datetime.min.time()).replace(hour=6)
                basin_rainfall = parse_num(cells[2].text)
                inflow = parse_num(cells[3].text)
                water_level = parse_num(cells[4].text)
                full_water_level = parse_num(cells[5].text)
                effective_storage = parse_num(cells[6].text, remove_comma=True)
                storage_rate = parse_num(cells[7].text, remove_percent=True)
                outflow = parse_num(cells[9].text) if len(cells) > 9 else None

                cur.execute("""
                    INSERT INTO public.fhy_reservoir_data (
                        id, reservoir_id, observation_time,
                        basin_rainfall_mm, inflow_cms, effective_storage, outflow_cms,
                        water_level_m, full_water_level_m, storage_rate,
                        data_date
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (reservoir_id, data_date) DO NOTHING
                """, (next_id, rid, obs_time,
                      basin_rainfall, inflow, effective_storage, outflow,
                      water_level, full_water_level, storage_rate,
                      target_date))
                next_id += 1
                inserted += 1
            conn.commit()
            cur.close()
            logger.info(f"  ✅ {target_date}: {inserted} 筆水庫寫入")
            return inserted, None

        except Exception as e:
            logger.warning(f"  ❌ {target_date} attempt {attempt} 失敗: {e}")
            if attempt < max_retries:
                time.sleep(10 * attempt)  # backoff
            else:
                return 0, f"exception:{type(e).__name__}"
    return 0, "max_retries"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="結束日期 YYYY-MM-DD")
    parser.add_argument("--driver-rotate-days", type=int, default=30,
                        help="每 N 天重新開瀏覽器（防記憶體累積，預設 30）")
    args = parser.parse_args()

    start_date = date_type.fromisoformat(args.start)
    end_date = date_type.fromisoformat(args.end)
    if start_date > end_date:
        logger.error(f"start {start_date} > end {end_date}")
        sys.exit(1)

    total_days = (end_date - start_date).days + 1
    logger.info(f"=== 開始回溯收集：{start_date} ~ {end_date}（{total_days} 天）===")

    # 1. 連 DB + 撈名稱對映
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                            user=DB_USER, password=DB_PASS)
    with conn.cursor() as cur:
        cur.execute("SELECT reservoir_id, reservoir_name FROM public.fhy_reservoir")
        name_to_id = {name: rid for rid, name in cur.fetchall()}
    logger.info(f"已撈 {len(name_to_id)} 個水庫名稱對映")

    # 2. 開瀏覽器
    driver = get_driver()
    days_since_rotate = 0

    # 3. 跑迴圈
    failed = []
    succeeded = 0
    skipped = 0
    current = start_date
    while current <= end_date:
        try:
            inserted, reason = fetch_one_day(driver, current, name_to_id, conn)
            if reason is None:
                succeeded += 1
            else:
                skipped += 1
                failed.append((current.isoformat(), reason))
        except Exception as e:
            logger.error(f"  💥 {current} 完全失敗: {e}")
            failed.append((current.isoformat(), f"top:{type(e).__name__}"))
            skipped += 1

        days_since_rotate += 1
        # 每 N 天重開瀏覽器（防記憶體爆掉）
        if days_since_rotate >= args.driver_rotate_days:
            logger.info(f"  ⟳ 重新啟動瀏覽器（已跑 {days_since_rotate} 天）")
            try:
                driver.quit()
            except Exception:
                pass
            driver = get_driver()
            days_since_rotate = 0

        current += timedelta(days=1)

    # 4. 收尾
    try:
        driver.quit()
    except Exception:
        pass
    conn.close()

    # 5. 統計
    logger.info("=" * 60)
    logger.info(f"✅ 成功: {succeeded} / {total_days} 天")
    logger.info(f"⏭️  跳過/失敗: {skipped} 天")
    if failed:
        logger.info(f"失敗清單（最多列 20）:")
        for d, r in failed[:20]:
            logger.info(f"  {d}: {r}")
        logger.info(f"（完整清單寫到 {LOG_DIR / 'backfill_2014_2015_failed.log'}）")
        with open(LOG_DIR / "backfill_2014_2015_failed.log", "w", encoding="utf-8") as f:
            for d, r in failed:
                f.write(f"{d}\t{r}\n")


if __name__ == "__main__":
    main()
