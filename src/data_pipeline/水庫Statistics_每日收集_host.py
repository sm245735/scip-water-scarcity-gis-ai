#!/usr/bin/env python3
"""
=============================================================================
水庫 Statistics.aspx 每日收集腳本（每日 01:00 執行）
=============================================================================
用途：下載昨天的水庫日水情資料，匯入 reservoir_daily
使用：Chrome Remote Debugging on localhost:18800

Crontab（主機端）：
    0 1 * * * /home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai/src/data_pipeline/水庫Statistics_每日收集_host.py >> /home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai/src/logs/YYYYMMDD.log 2>&1
"""

import os, time, logging
from datetime import datetime, timedelta, date as date_type
import psycopg2
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = "thesis_analysis"
DB_USER = "sm245735"
DB_PASS = os.environ["DB_PASSWORD"]

LOG_DIR = "/home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai/src/logs/"
os.makedirs(LOG_DIR, exist_ok=True)

# 每天一個 LOG 檔，檔名：YYYYMMDD.log
log_date_str = (date_type.today() - timedelta(days=1)).strftime("%Y%m%d")
LOG_FILE = os.path.join(LOG_DIR, f"{log_date_str}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_driver():
    options = Options()
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    service = Service("/tmp/chromedriver-linux64/chromedriver")
    return webdriver.Chrome(service=service, options=options)

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
    yesterday = date_type.today() - timedelta(days=1)
    logger.info(f"=== 水庫每日收集：{yesterday} ===")

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

        Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboYear")).select_by_value(str(yesterday.year))
        time.sleep(3)
        Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboMonth")).select_by_value(str(yesterday.month))
        time.sleep(3)
        Select(driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboDay")).select_by_value(str(yesterday.day))
        time.sleep(3)

        driver.execute_script("document.getElementById('ctl00_cphMain_btnQuery').click();")
        time.sleep(15)

        gvlist = driver.find_element(By.ID, "ctl00_cphMain_gvList")
        rows = gvlist.find_elements(By.TAG_NAME, "tr")
        data_rows = [r for r in rows if r.find_elements(By.TAG_NAME, "td")]

        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                                user=DB_USER, password=DB_PASS)
        cur = conn.cursor()
        cur.execute("SELECT reservoir_id, reservoir_name FROM reservoirs")
        name_to_id = {name: rid for rid, name in cur.fetchall()}

        count = 0
        for row in data_rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) < 10:
                continue
            name = cells[0].text.strip()
            if name not in name_to_id:
                continue

            rid = name_to_id[name]
            obs_time_str = cells[1].text.strip()
            if obs_time_str in ('--', '', 'NULL'):
                obs_time = None
            else:
                try:
                    obs_time = datetime.strptime(obs_time_str, "%Y/%m/%d %H:%M")
                except ValueError:
                    obs_time = None  # 格式不符則寫入 NULL
            basin_rainfall = parse_num(cells[2].text)
            inflow = parse_num(cells[3].text)
            water_level = parse_num(cells[4].text)
            full_water_level = parse_num(cells[5].text)
            effective_storage = parse_num(cells[6].text, remove_comma=True)
            storage_rate = parse_num(cells[7].text, remove_percent=True)
            outflow = parse_num(cells[9].text) if len(cells) > 9 else None

            cur.execute("""
                INSERT INTO reservoir_daily (
                    data_date, reservoir_id, observation_time,
                    basin_rainfall_mm, inflow_cms, effective_storage, outflow_cms,
                    water_level_m, full_water_level_m, storage_rate
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (data_date, reservoir_id) DO NOTHING
            """, (yesterday, rid, obs_time,
                  basin_rainfall, inflow, effective_storage, outflow,
                  water_level, full_water_level, storage_rate))
            count += 1

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"✅ 匯入完成：{count} 筆水庫資料（{yesterday}）")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()