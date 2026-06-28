#!/usr/bin/env python3
"""
=============================================================================
水庫 Statistics.aspx 指定日期清單收集（覆蓋式）
=============================================================================

用途：從 src/data_pipeline/寶二水庫水位缺值日期.txt 之類的清單
      抓指定日期的全水庫水情，**覆蓋**現有 fhy_reservoir_data 資料
      （跟「每日收集」用 ON CONFLICT DO NOTHING 不一樣）

跟 2014-2015 回溯版的差異：
  - 日期來源：清單檔（一行一個 YYYY-MM-DD），不是 --start/--end 整段
  - 寫入策略：ON CONFLICT (reservoir_id, data_date) DO UPDATE
              覆蓋舊值（修 2016-01-03 這種錯值、obs_time 寫錯的）
  - observation_time：統一設為 target_date 22:00（跟現有 daily 收集一致）
  - 沒抓到的水庫（不在 cboSearch='全部' 列表）就保留 DB 現有值不動

執行：
  # 全 51 天（背景跑）
  nohup /root/scip-water-scarcity-gis-ai/venv/bin/python \
      src/data_pipeline/水庫Statistics_補特定日期.py \
      --dates src/data_pipeline/寶二水庫水位缺值日期.txt \
      >> logs/backfill_53dates.log 2>&1 &

  # 單日測試
  ... --dates /tmp/single_date.txt
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


# ---------- DB 連線（這台 server 的設定）----------
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
        if line.startswith("POSTGRES_PASSWORD="):
            DB_PASS = line.split("=", 1)[1]
            break

# ---------- 路徑 ----------
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- logger：執行時依 --dates 檔名決定 log 檔 ----------
def setup_logger(dates_path: Path) -> tuple:
    log_name = f"backfill_{dates_path.stem}.log"
    failed_name = f"backfill_{dates_path.stem}_failed.log"
    logger = logging.getLogger("backfill_specific")
    logger.setLevel(logging.INFO)
    # 清掉舊 handler（避免重複）
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_DIR / log_name, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, LOG_DIR / failed_name


# ---------- 工具 ----------
def get_driver():
    """用 selenium/standalone-chrome container 跑"""
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Remote(
        command_executor="http://localhost:4444",
        options=options,
    )


def parse_num(val, remove_comma=False, remove_percent=False):
    if not val or val.strip() in ('--', '', '-'):
        return None
    v = val.strip()
    if remove_comma:
        v = v.replace(',', '')
    if remove_percent:
        v = v.replace('%', '').strip()
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_dates(path: Path) -> list:
    """讀日期清單（忽略空行、# 開頭的註解）"""
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        try:
            out.append(date_type.fromisoformat(s))
        except ValueError:
            raise SystemExit(f"❌ 無法解析日期：{s!r}（{path}）")
    return sorted(set(out))  # 去重、排序（保證跑過的順序穩定）


# ---------- 抓一天 ----------
def fetch_one_day(driver, target_date: date_type, name_to_id: dict,
                  conn, max_retries: int = 3) -> tuple:
    """
    抓一天資料，**覆蓋**寫入 fhy_reservoir_data。
    回傳 (inserted_count, updated_count, skipped_reason)
    """
    # 統一 obs_time 為當天 22:00（跟現在 daily 收集一致）
    obs_time = datetime.combine(target_date, datetime.min.time()).replace(hour=22)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"  導航 {target_date} (attempt {attempt}/{max_retries})")
            driver.get("https://fhy.wra.gov.tw/ReservoirPage_2011/Statistics.aspx")
            time.sleep(5)

            # 切到「全部」水庫
            driver.execute_script(
                "var s=document.getElementById('ctl00_cphMain_cboSearch');"
                "if(s){s.value='全部';s.dispatchEvent(new Event('change',{bubbles:true}));}"
            )
            time.sleep(3)

            Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboYear")
                   ).select_by_value(str(target_date.year))
            time.sleep(1)
            Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboMonth")
                   ).select_by_value(str(target_date.month))
            time.sleep(1)
            Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboDay")
                   ).select_by_value(str(target_date.day))
            time.sleep(1)
            # 注意：hour 留 0 即可（hour=0 那組就是當天資料，obs_time 我們自己蓋 22:00）
            Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboHour")
                   ).select_by_value("0")
            time.sleep(1)

            driver.execute_script(
                "document.getElementById('ctl00_cphMain_btnQuery').click();"
            )
            time.sleep(6)

            gvlist = driver.find_element(By.ID, "ctl00_cphMain_gvList")
            rows = gvlist.find_elements(By.TAG_NAME, "tr")
            data_rows = [r for r in rows if r.find_elements(By.TAG_NAME, "td")]

            if not data_rows:
                logger.warning(f"  {target_date} 沒有回傳資料")
                return 0, 0, "no_data"

            cur = conn.cursor()
            inserted = 0
            updated = 0
            skipped_name = 0

            # 撈當前 max(id) 當起始值（fhy_reservoir_data.id 不是 SERIAL，
            # 要手動遞增，否則 INSERT 會被 NOT NULL 擋下）
            cur.execute("SELECT COALESCE(MAX(id), 0) FROM public.fhy_reservoir_data")
            start_id = cur.fetchone()[0]

            for row in data_rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 10:
                    continue
                name = cells[0].text.strip()
                if name not in name_to_id:
                    skipped_name += 1
                    continue
                rid = name_to_id[name]

                basin_rainfall = parse_num(cells[2].text)
                inflow = parse_num(cells[3].text)
                water_level = parse_num(cells[4].text)
                full_water_level = parse_num(cells[5].text)
                effective_storage = parse_num(cells[6].text, remove_comma=True)
                storage_rate = parse_num(cells[7].text, remove_percent=True)
                outflow = parse_num(cells[9].text) if len(cells) > 9 else None

                # 覆蓋式寫入：先看 row 存不存在，決定算 insert 還是 update
                cur.execute(
                    "SELECT id FROM public.fhy_reservoir_data "
                    "WHERE reservoir_id=%s AND data_date=%s",
                    (rid, target_date),
                )
                row0 = cur.fetchone()
                existed = row0 is not None

                if existed:
                    # 已存在 → 用既有 id 做 UPDATE
                    cur.execute("""
                        UPDATE public.fhy_reservoir_data SET
                            observation_time = %s,
                            basin_rainfall_mm = %s,
                            inflow_cms = %s,
                            effective_storage = %s,
                            outflow_cms = %s,
                            water_level_m = %s,
                            full_water_level_m = %s,
                            storage_rate = %s
                        WHERE id = %s
                    """, (obs_time,
                          basin_rainfall, inflow, effective_storage, outflow,
                          water_level, full_water_level, storage_rate,
                          row0[0]))
                    updated += 1
                else:
                    # 不存在 → 用 MAX(id)+N 遞增寫 INSERT
                    start_id += 1
                    cur.execute("""
                        INSERT INTO public.fhy_reservoir_data (
                            id, reservoir_id, observation_time,
                            basin_rainfall_mm, inflow_cms, effective_storage, outflow_cms,
                            water_level_m, full_water_level_m, storage_rate,
                            data_date
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (start_id, rid, obs_time,
                          basin_rainfall, inflow, effective_storage, outflow,
                          water_level, full_water_level, storage_rate,
                          target_date))
                    inserted += 1

            conn.commit()
            cur.close()
            logger.info(
                f"  ✅ {target_date}: insert {inserted} / update {updated} "
                f"（略過 {skipped_name} 個不在 master 的水庫）"
            )
            return inserted, updated, None

        except Exception as e:
            logger.warning(f"  ❌ {target_date} attempt {attempt} 失敗: {e}")
            # 先 ROLLBACK 把 abort 的 transaction 清掉，否則後續 SQL 一定炸
            try:
                conn.rollback()
            except Exception:
                pass
            # WebDriverException / tab crashed → 砍 driver 讓下次重連
            try:
                driver.quit()
            except Exception:
                pass
            driver = get_driver()
            if attempt < max_retries:
                time.sleep(10 * attempt)
            else:
                return 0, 0, f"exception:{type(e).__name__}"
    return 0, 0, "max_retries"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dates", required=True,
        help="日期清單檔路徑（一行一個 YYYY-MM-DD）",
    )
    parser.add_argument(
        "--driver-rotate-days", type=int, default=20,
        help="每 N 天重新開瀏覽器（預設 20）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只列日期清單，不實際抓",
    )
    args = parser.parse_args()

    dates_path = Path(args.dates)
    if not dates_path.is_absolute():
        dates_path = (REPO_ROOT / dates_path).resolve()
    if not dates_path.exists():
        raise SystemExit(f"❌ 找不到日期清單：{dates_path}")

    global logger
    logger, failed_log = setup_logger(dates_path)
    dates = load_dates(dates_path)
    logger.info(f"=== 開始補特定日期清單：{dates_path}（{len(dates)} 天）===")
    logger.info(f"  範圍：{dates[0]} ~ {dates[-1]}")
    if args.dry_run:
        for d in dates:
            print(d)
        return

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
    succeeded = 0
    failed = []
    total_inserted = 0
    total_updated = 0
    for idx, d in enumerate(dates, 1):
        try:
            ins, upd, reason = fetch_one_day(driver, d, name_to_id, conn)
            total_inserted += ins
            total_updated += upd
            if reason is None:
                succeeded += 1
            else:
                failed.append((d.isoformat(), reason))
        except Exception as e:
            logger.error(f"  💥 {d} 完全失敗: {e}")
            # 保險：把外層漏掉的 abort transaction 清掉
            try:
                conn.rollback()
            except Exception:
                pass
            failed.append((d.isoformat(), f"top:{type(e).__name__}"))

        # 每跑 5 天印一次系統 free RAM（看 selenium container 有沒有累積吃光）
        if idx % 5 == 0:
            try:
                free_mb = os.popen("free -m | awk '/^Mem:/ {print $4}'").read().strip()
                logger.info(f"  📊 進度 {idx}/{len(dates)}，系統 free RAM = {free_mb} MB")
            except Exception:
                pass

        days_since_rotate += 1
        if days_since_rotate >= args.driver_rotate_days:
            logger.info(f"  ⟳ 重新啟動瀏覽器（已跑 {days_since_rotate} 天）")
            try:
                driver.quit()
            except Exception:
                pass
            driver = get_driver()
            days_since_rotate = 0

    # 4. 收尾
    try:
        driver.quit()
    except Exception:
        pass
    conn.close()

    # 5. 統計
    logger.info("=" * 60)
    logger.info(f"✅ 成功: {succeeded} / {len(dates)} 天")
    logger.info(f"   新增列: {total_inserted}，覆蓋列: {total_updated}")
    logger.info(f"⏭️  跳過/失敗: {len(failed)} 天")
    if failed:
        for d, r in failed:
            logger.info(f"  {d}: {r}")
        with open(failed_log, "w", encoding="utf-8") as f:
            for d, r in failed:
                f.write(f"{d}\t{r}\n")
        logger.info(f"（完整清單寫到 {failed_log}）")


if __name__ == "__main__":
    main()
