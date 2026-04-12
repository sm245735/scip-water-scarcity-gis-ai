#!/usr/bin/env python3
"""
水庫Statistics下載_遠端Chrome_v1_20260412.py

用途：使用本機 Chrome Remote Debugging，自動化讀取水庫統計日資料（直接解析表格，不靠 XLS 下載）
目標：2016-01-01 到 2026-01-01 全部日期

方法：
- 使用本機已開啟的 Chrome（remote debugging port 18800）
- 用 JS 讀取表格繞過 Selenium DOM 查詢問題
- 每日讀取後直接存入 CSV（不依賴 Excel 下載）

使用方式：
  source ~/selenium_env/bin/activate
  cd /home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai
  python3 src/data_pipeline/水庫Statistics下載_遠端Chrome_v1_20260412.py
"""

import os
import time
import json
import csv
import logging
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# === 設定 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # scip-water-scarcity-gis-ai/
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "水庫歷史水情")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "水庫統計表_2016_2026.csv")

START_DATE = datetime(2016, 1, 1)
END_DATE = datetime(2026, 1, 1)
REMOTE_DEBUG_ADDR = "127.0.0.1:18800"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_DIR = os.path.join(PROJECT_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "statistics_download.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def create_driver():
    options = Options()
    options.add_experimental_option("debuggerAddress", REMOTE_DEBUG_ADDR)
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    return driver

def click_js(driver, element_id):
    driver.execute_script(f"document.getElementById('{element_id}').click();")

def set_date(driver, dt):
    driver.execute_script("""
        var yr = document.getElementById('ctl00_cphMain_ucDate_cboYear');
        if(yr){yr.value = arguments[0]; yr.dispatchEvent(new Event('change', {bubbles:true}));}
    """, str(dt.year))
    time.sleep(4)

    driver.execute_script("""
        var mo = document.getElementById('ctl00_cphMain_ucDate_cboMonth');
        if(mo){mo.value = arguments[0]; mo.dispatchEvent(new Event('change', {bubbles:true}));}
    """, str(dt.month))
    time.sleep(4)

    driver.execute_script("""
        var dy = document.getElementById('ctl00_cphMain_ucDate_cboDay');
        if(dy){dy.value = arguments[0]; dy.dispatchEvent(new Event('change', {bubbles:true}));}
    """, str(dt.day))
    time.sleep(4)

def read_table(driver):
    """用 JS 讀取表格，回傳 [(水庫名, 水情時間, 水位, 蓄水量, ...), ...]"""
    result = driver.execute_script("""
        var tables = document.querySelectorAll('table');
        if (!tables || tables.length === 0) return JSON.stringify([]);
        
        var rows = tables[0].querySelectorAll('tr');
        var data = [];
        for (var row of rows) {
            var cells = row.querySelectorAll('td');
            if (cells.length === 0) continue;
            // Skip header row
            var firstText = cells[0].innerText.trim();
            if (firstText === '水庫名稱' || firstText === '') continue;
            
            var rowData = [];
            for (var c of cells) {
                rowData.push(c.innerText.trim());
            }
            data.push(rowData);
        }
        return JSON.stringify(data);
    """)
    try:
        return json.loads(result) if result else []
    except:
        return []

def write_rows_to_csv(rows, date_str, csv_writer):
    """將一筆資料寫入 CSV（date 為固定值，水庫名為第一欄）"""
    for row in rows:
        if len(row) < 5:
            continue
        # CSV 欄位：date, 水庫名, 水情時間, 集水區降雨, 水位, 滿水位, 蓄水量, 蓄水率, ...
        writer.writerow({
            'date': date_str,
            'reservoir_name': row[0],
            'observation_time': row[1] if len(row) > 1 else '',
            'basin_rainfall_mm': row[2] if len(row) > 2 else '',
            'inflow_cms': row[3] if len(row) > 3 else '',
            'water_level_m': row[4] if len(row) > 4 else '',
            'full_water_level_m': row[5] if len(row) > 5 else '',
            'effective_storage': row[6] if len(row) > 6 else '',
            'storage_rate': row[7] if len(row) > 7 else '',
            'outflow_cms': row[8] if len(row) > 8 else '',
        })

def append_to_csv(rows, date_str):
    """附加資料到 CSV（無標題，與其他日並排）"""
    filepath = OUTPUT_CSV
    
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        # 欄位順序（無 date header，純資料表）
        fieldnames = [
            'date', 'reservoir_name', 'observation_time',
            'basin_rainfall_mm', 'inflow_cms', 'water_level_m',
            'full_water_level_m', 'effective_storage', 'storage_rate', 'outflow_cms'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for row in rows:
            if len(row) < 5:
                continue
            writer.writerow({
                'date': date_str,
                'reservoir_name': row[0],
                'observation_time': row[1] if len(row) > 1 else '',
                'basin_rainfall_mm': row[2] if len(row) > 2 else '',
                'inflow_cms': row[3] if len(row) > 3 else '',
                'water_level_m': row[4] if len(row) > 4 else '',
                'full_water_level_m': row[5] if len(row) > 5 else '',
                'effective_storage': row[6] if len(row) > 6 else '',
                'storage_rate': row[7] if len(row) > 7 else '',
                'outflow_cms': row[8] if len(row) > 8 else '',
            })

def main():
    logger.info("=" * 60)
    logger.info("水庫 Statistics Selenium 自動化（直接讀表）")
    logger.info(f"目標：{START_DATE.date()} ~ {END_DATE.date()}")
    logger.info(f"輸出：{OUTPUT_CSV}")
    logger.info("=" * 60)
    
    driver = None
    try:
        driver = create_driver()
        
        # 1. 導航 + 設定為「全部」
        logger.info("導航到 Statistics.aspx...")
        driver.get("https://fhy.wra.gov.tw/ReservoirPage_2011/Statistics.aspx")
        time.sleep(8)
        
        logger.info("設定為「全部」...")
        driver.execute_script("""
            var s = document.getElementById('ctl00_cphMain_cboSearch');
            if(s){s.value='全部'; s.dispatchEvent(new Event('change', {bubbles:true}));}
        """)
        time.sleep(8)
        
        # 2. 主迴圈
        current = START_DATE
        success_count = 0
        error_count = 0
        
        while current <= END_DATE:
            date_str = current.strftime('%Y-%m-%d')
            try:
                # 設定日期
                set_date(driver, current)
                
                # 點查詢
                click_js(driver, 'ctl00_cphMain_btnQuery')
                time.sleep(6)
                
                # 讀取表格
                rows_data = read_table(driver)
                
                if rows_data:
                    # 寫入 CSV
                    append_to_csv(rows_data, date_str)
                    logger.info(f"  ✅ {date_str} 寫入 {len(rows_data)} 水庫")
                    success_count += 1
                else:
                    logger.warning(f"  ⚠️ {date_str} 無資料")
                    error_count += 1
                
            except Exception as e:
                logger.error(f"  ❌ {date_str} 失敗: {e}")
                error_count += 1
            
            # 每 100 天進度報告
            if (current - START_DATE).days > 0 and (current - START_DATE).days % 100 == 0:
                logger.info(f"\n📊 進度：{current.date()}，成功 {success_count}，失敗 {error_count}\n")
            
            current += timedelta(days=1)
        
        # 最終統計
        total_days = (END_DATE - START_DATE).days + 1
        logger.info("=" * 60)
        logger.info(f"完成！共 {total_days} 天，成功 {success_count}，失敗 {error_count}")
        logger.info(f"輸出檔案：{OUTPUT_CSV}")
        if os.path.exists(OUTPUT_CSV):
            size = os.path.getsize(OUTPUT_CSV) / 1024 / 1024
            lines = sum(1 for _ in open(OUTPUT_CSV, encoding='utf-8'))
            logger.info(f"檔案大小：{size:.1f} MB，共 {lines:,} 列")
        
    except KeyboardInterrupt:
        logger.info("\n使用者中斷！")
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()