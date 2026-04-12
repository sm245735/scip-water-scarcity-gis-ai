#!/usr/bin/env python3
"""
Statistics.aspx Selenium 自動化
用途：下載 2016-01-01 到 2026-01-01 的水庫日資料（Excel 檔）
"""
import os
import time
import logging
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# === 設定 ===
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "03. 資料", "Statistics")
START_DATE = datetime(2016, 1, 1)
END_DATE = datetime(2026, 1, 1)
TARGET_RESERVOIR = "寶山水庫"  # 只抓這個，不一定要 ALL

LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "statistics_download.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Chrome 設定 ===
def create_driver():
    options = Options()
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--start-maximized")
    # options.add_argument("--headless")  # 若要無頭模式就開這行
    
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)
    return driver

def wait_for_postback(driver, timeout=15):
    """等待 ASP.NET PostBack 完成（畫面不再轉圈）"""
    time.sleep(3)  # 先等 3 秒
    # 可搭配 JS 讓 PostBack 完全完成再往下
    time.sleep(2)

def get_date_selects(driver):
    """取得年月日下拉選單"""
    year_el = driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboYear")
    month_el = driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboMonth")
    day_el = driver.find_element(By.ID, "ctl00_cphMain_ucDate_cboDay")
    return year_el, month_el, day_el

def set_date(driver, year_el, month_el, day_el, dt):
    """設定特定日期（datetime物件）"""
    Select(year_el).select_by_value(str(dt.year))
    wait_for_postback(driver)
    Select(month_el).select_by_value(f"{dt.month:02d}")
    wait_for_postback(driver)
    Select(day_el).select_by_value(f"{dt.day:02d}")
    wait_for_postback(driver)

def click_query(driver):
    """點擊查詢按鈕"""
    btn = driver.find_element(By.ID, "ctl00_cphMain_btnQuery")
    btn.click()
    wait_for_postback(driver)

def download_excel(driver, dt, output_dir):
    """點擊 Excel 下載按鈕，下載到指定資料夾"""
    filename = f"水庫統計_{dt.strftime('%Y%m%d')}.xls"
    filepath = os.path.join(output_dir, filename)
    
    # 如果已下載過就跳過
    if os.path.exists(filepath):
        logger.info(f"已存在，跳過: {filename}")
        return True
    
    try:
        # 點擊 Excel 下載按鈕
        excel_btn = driver.find_element(By.ID, "ctl00_cphMain_btnExcel")
        excel_btn.click()
        
        # 等下載完成（最長 30 秒）
        # Chrome 預設下載到 ~/Downloads/
        # 實務上要確認下載完成，建議用 AutoIt 或其他工具
        time.sleep(5)
        
        # 移動檔案到目標資料夾
        downloads = os.path.expanduser("~/Downloads")
        downloaded = [f for f in os.listdir(downloads) if f.endswith(".xls")]
        
        if downloaded:
            src = os.path.join(downloads, downloaded[0])
            os.rename(src, filepath)
            logger.info(f"下載完成: {filename}")
            return True
        else:
            logger.warning(f"找不到下載的檔案: {filename}")
            return False
    except Exception as e:
        logger.error(f"下載失敗 {dt.date()}: {e}")
        return False

def find_reservoir_row(driver, target_name):
    """在表格中找到特定水庫的回數"""
    rows = driver.find_elements(By.CSS_SELECTOR, "#ctl00_cphMain_pnlGrid table tr")
    for row in rows:
        if target_name in row.text and "第二水庫" not in row.text:
            cells = row.find_elements(By.TAG_NAME, "td")
            logger.info(f"找到 {target_name}: " + " | ".join([c.text for c in cells[:8]]))
            return True
    return False

def main():
    logger.info("=== Statistics.aspx Selenium 自動化 ===")
    logger.info(f"輸出目錄: {OUTPUT_DIR}")
    logger.info(f"日期範圍: {START_DATE.date()} ~ {END_DATE.date()}")
    
    driver = create_driver()
    
    try:
        # 1. 導航到頁面
        logger.info("導航到 Statistics.aspx...")
        driver.get("https://fhy.wra.gov.tw/ReservoirPage_2011/Statistics.aspx")
        wait_for_postback(driver)
        
        # 2. 設定查詢條件為「全部」
        logger.info("設定查詢條件為「全部」...")
        search_select = Select(driver.find_element(By.ID, "ctl00_cphMain_cboSearch"))
        search_select.select_by_value("全部")
        wait_for_postback(driver)
        
        # 3. 迴圈每天
        current_date = START_DATE
        success_count = 0
        
        while current_date <= END_DATE:
            try:
                # 設定日期
                year_el, month_el, day_el = get_date_selects(driver)
                set_date(driver, year_el, month_el, day_el, current_date)
                
                # 點查詢
                click_query(driver)
                
                # 等 1 秒後抓資料
                time.sleep(1)
                
                # 找寶山水庫資料
                if find_reservoir_row(driver, TARGET_RESERVOIR):
                    logger.info(f"✅ {current_date.date()} 找到 {TARGET_RESERVOIR}")
                else:
                    logger.warning(f"⚠️ {current_date.date()} 找不到 {TARGET_RESERVOIR}")
                
                # 下載 Excel
                download_excel(driver, current_date, OUTPUT_DIR)
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"處理 {current_date.date()} 失敗: {e}")
            
            current_date += timedelta(days=1)
            
            # 每 100 天報告一次進度
            if success_count % 100 == 0:
                logger.info(f"已完成 {success_count} 天 ({START_DATE.date()} ~ {current_date.date()})")
        
        logger.info(f"=== 完成！共成功處理 {success_count} 天 ===")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    main()