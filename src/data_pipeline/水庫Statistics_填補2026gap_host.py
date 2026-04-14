#!/usr/bin/env python3
"""
=============================================================================
水庫 Statistics.aspx Gap 填補腳本（Host 版，2026-01-19 ~ 2026-04-14）
=============================================================================
修復：
- START_DATE 改為 2026-01-19（DB 已有 2026-01-01 ~ 2026-01-18）
- 每天處理完立刻 commit（不再等 50 天）
- Chrome session 心跳檢查，失效時自動重連
=============================================================================
"""

import os, sys, time, logging, csv
from datetime import datetime, timedelta
import psycopg2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

DB_HOST = "localhost"
DB_PORT = "9235"
DB_NAME = "thesis_analysis"
DB_USER = "sm245735"
DB_PASS = os.environ["DB_PASSWORD"]

PROJECT_DIR = "/home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai"
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "水庫統計_gap_2026.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "statistics_gap_fill.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

START_DATE = datetime(2026, 1, 19)
END_DATE = datetime(2026, 4, 14)

def get_driver():
    options = Options()
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--start-maximized")
    service = Service("/tmp/chromedriver-linux64/chromedriver")
    return webdriver.Chrome(service=service, options=options)

def set_date_select(driver, dt):
    Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboYear")).select_by_value(str(dt.year))
    time.sleep(3)
    Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboMonth")).select_by_value(str(dt.month))
    time.sleep(3)
    Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboDay")).select_by_value(str(dt.day))
    time.sleep(3)

def parse_num(val, remove_comma=False, remove_percent=False):
    if not val or val.strip() in ('--', '', '-'):
        return None
    v = val.strip()
    if remove_comma: v = v.replace(',', '')
    if remove_percent: v = v.replace('%', '').strip()
    try:
        return float(v)
    except:
        return None

def main():
    logger.info(f"=== Gap 填補：{START_DATE.date()} ~ {END_DATE.date()} ===")

    driver = get_driver()

    try:
        logger.info("導航到 Statistics.aspx...")
        driver.get("https://fhy.wra.gov.tw/ReservoirPage_2011/Statistics.aspx")
        time.sleep(10)

        driver.execute_script(
            "var s=document.getElementById('ctl00_cphMain_cboSearch');"
            "if(s){s.value='全部';s.dispatchEvent(new Event('change',{bubbles:true}));}"
        )
        time.sleep(8)

        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                                user=DB_USER, password=DB_PASS)
        cur = conn.cursor()
        cur.execute("SELECT reservoir_id, reservoir_name FROM reservoirs")
        name_to_id = {name: rid for rid, name in cur.fetchall()}
        logger.info(f"水庫對照表：{len(name_to_id)} 筆")

        csv_file = open(OUTPUT_CSV, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'date', 'reservoir_name', 'observation_time',
            'basin_rainfall_mm', 'inflow_cms', 'water_level_m',
            'full_water_level_m', 'effective_storage', 'storage_rate', 'outflow_cms'
        ])

        current = START_DATE
        success = 0
        error = 0

        while current <= END_DATE:
            # Chrome session 心跳檢查
            try:
                _ = driver.current_url
            except Exception:
                logger.warning("Chrome session 失效，重新連線...")
                driver.quit()
                time.sleep(5)
                driver = get_driver()
                driver.get("https://fhy.wra.gov.tw/ReservoirPage_2011/Statistics.aspx")
                time.sleep(10)
                driver.execute_script(
                    "var s=document.getElementById('ctl00_cphMain_cboSearch');"
                    "if(s){s.value='全部';s.dispatchEvent(new Event('change',{bubbles:true}));}"
                )
                time.sleep(8)

            try:
                set_date_select(driver, current)
                driver.execute_script("document.getElementById('ctl00_cphMain_btnQuery').click();")
                time.sleep(15)

                # 取出資料（找有 td 的行）
                gvlist = driver.find_element(By.ID, "ctl00_cphMain_gvList")
                rows = gvlist.find_elements(By.TAG_NAME, "tr")
                data_rows = [r for r in rows if r.find_elements(By.TAG_NAME, "td")]

                for row in data_rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 10:
                        continue
                    name = cells[0].text.strip()
                    if name not in name_to_id:
                        continue

                    rid = name_to_id[name]
                    obs_time_str = cells[1].text.strip(); obs_time = None if obs_time_str in ('--', '', 'NULL') else obs_time_str
                    basin_rainfall = parse_num(cells[2].text)
                    inflow = parse_num(cells[3].text)
                    water_level = parse_num(cells[4].text)
                    full_water_level = parse_num(cells[5].text)
                    effective_storage = parse_num(cells[6].text, remove_comma=True)
                    storage_rate = parse_num(cells[7].text, remove_percent=True)
                    outflow = parse_num(cells[17].text)

                    csv_writer.writerow([
                        current.strftime('%Y-%m-%d'), name, obs_time or '',
                        basin_rainfall or '', inflow or '', water_level or '',
                        full_water_level or '', effective_storage or '',
                        storage_rate or '', outflow or ''
                    ])

                    cur.execute("""
                        INSERT INTO reservoir_daily (
                            data_date, reservoir_id, observation_time,
                            basin_rainfall_mm, inflow_cms, effective_storage, outflow_cms,
                            water_level_m, full_water_level_m, storage_rate
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (data_date, reservoir_id) DO NOTHING
                    """, (
                        current.strftime('%Y-%m-%d'), rid, obs_time or None,
                        basin_rainfall, inflow, effective_storage, outflow,
                        water_level, full_water_level, storage_rate
                    ))

                logger.info(f"  {current.date()} → {len(data_rows)} 筆")
                success += 1
                conn.commit()  # 每天處理完立刻 commit

            except Exception as e:
                logger.error(f"  {current.date()} 失敗：{e}")
                error += 1

            current += timedelta(days=1)

        conn.commit()  # 最後一次 commit
        csv_file.close()
        cur.close()
        conn.close()
        logger.info(f"=== 完成！成功 {success} 天，錯誤 {error} 天 ===")
        logger.info(f"CSV 已寫入：{OUTPUT_CSV}")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
