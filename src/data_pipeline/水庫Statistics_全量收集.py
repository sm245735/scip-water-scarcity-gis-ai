#!/usr/bin/env python3
"""
水庫 Statistics 全量收集腳本（2016-2026）
"""
import os, time, json, csv, logging
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

PROJECT_DIR = "/home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "水庫歷史水情")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "水庫統計表_2016_2026.csv")
START_DATE = datetime(2016, 1, 1)
END_DATE = datetime(2026, 1, 1)
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "statistics_run.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

options = Options()
options.add_experimental_option("debuggerAddress", "127.0.0.1:18800")
driver = webdriver.Chrome(options=options)

logger.info("導航...")
driver.get("https://fhy.wra.gov.tw/ReservoirPage_2011/Statistics.aspx")
time.sleep(8)

logger.info("設定為全部...")
driver.execute_script("var s=document.getElementById('ctl00_cphMain_cboSearch');if(s){s.value='全部';s.dispatchEvent(new Event('change',{bubbles:true}));}")
time.sleep(8)

def set_date(dt):
    driver.execute_script(f"var y=document.getElementById('ctl00_cphMain_ucDate_cboYear');if(y){{y.value='{dt.year}';y.dispatchEvent(new Event('change',{{bubbles:true}}));}}")
    time.sleep(4)
    driver.execute_script(f"var m=document.getElementById('ctl00_cphMain_ucDate_cboMonth');if(m){{m.value='{dt.month}';m.dispatchEvent(new Event('change',{{bubbles:true}}));}}")
    time.sleep(4)
    driver.execute_script(f"var d=document.getElementById('ctl00_cphMain_ucDate_cboDay');if(d){{d.value='{dt.day}';d.dispatchEvent(new Event('change',{{bubbles:true}}));}}")
    time.sleep(4)

current = START_DATE
success_count = 0
error_count = 0
start_time = datetime.now()

logger.info(f"開始收集：{START_DATE.date()} ~ {END_DATE.date()}")
logger.info(f"輸出：{OUTPUT_CSV}")

while current <= END_DATE:
    date_str = current.strftime('%Y-%m-%d')
    try:
        set_date(current)
        driver.execute_script("document.getElementById('ctl00_cphMain_btnQuery').click();")
        time.sleep(6)
        
        result = driver.execute_script("""
            var tables = document.querySelectorAll('table');
            if(!tables.length) return '[]';
            var rows = tables[0].querySelectorAll('tr');
            var data = [];
            for(var r of rows) {
                var c = r.querySelectorAll('td');
                if(c.length > 0) {
                    var txt = c[0].innerText.trim();
                    if(txt && txt !== '水庫名稱') {
                        data.push(Array.from(c).map(function(t){return t.innerText.trim();}));
                    }
                }
            }
            return JSON.stringify(data);
        """)
        
        rows_data = json.loads(result) if result and result != '[]' else []
        
        if rows_data:
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in rows_data:
                    if len(row) >= 5:
                        writer.writerow([date_str] + row[:9])
            success_count += 1
        else:
            error_count += 1
            logger.warning(f"無資料: {date_str}")
        
    except Exception as e:
        error_count += 1
        logger.error(f"錯誤 {date_str}: {e}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    days_done = (current - START_DATE).days + 1
    total_days = (END_DATE - START_DATE).days + 1
    pct = days_done / total_days * 100
    eta = elapsed / days_done * (total_days - days_done) / 3600 if days_done > 0 else 0
    
    if days_done % 100 == 0 or days_done == 1:
        logger.info(f"📊 {days_done}/{total_days} ({pct:.1f}%) | 成功 {success_count} 失敗 {error_count} | 預估剩下 {eta:.1f} 小時")
    
    current += timedelta(days=1)

logger.info("=" * 60)
logger.info(f"完成！成功 {success_count} 天，失敗 {error_count} 天")
logger.info(f"輸出：{OUTPUT_CSV}")
if os.path.exists(OUTPUT_CSV):
    lines = sum(1 for _ in open(OUTPUT_CSV))
    logger.info(f"共 {lines:,} 筆資料")
driver.quit()