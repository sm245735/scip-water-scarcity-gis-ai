# 檢查資料的時間區間以及是否有缺值


import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

# ==========================================
# 1. 精準定位 .env 檔案的路徑
# ==========================================
# 您的腳本在: /root/scip-water-scarcity-gis-ai/src/data_pipeline/test_venv.py
# 您的.env在: /root/scip-water-scarcity-gis-ai/src/.env
current_script_dir = Path(__file__).resolve().parent
env_path = current_script_dir.parent / ".env"

# 載入環境變數
load_dotenv(dotenv_path=env_path)

def check_missing_values():
    # ==========================================
    # 2. 讀取變數與連線資料庫
    # ==========================================
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "thesis_analysis")
    DB_USER = os.getenv("DB_USER", "sm245735")
    DB_PASS = os.getenv("DB_PASSWORD")

    if not DB_PASS:
        print(f"❌ 錯誤：找不到密碼！請確認 .env 檔案是否真的存在於 {env_path}")
        return

    print(f"✅ 成功讀取環境變數！準備連線至 {DB_HOST}:{DB_PORT}...\n")
    
    # 建立 SQLAlchemy 連線
    engine_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(engine_url)

    try:
        # ==========================================
        # 3. 表格 1: 水庫水位資料 (fhy_reservoir_data)
        # ==========================================
        # ★ 請確認這裡的表名是您資料庫中實際的名稱
        query_reservoir = "SELECT storage_rate,TO_CHAR(observation_time, 'YYYY-MM-DD') AS obs_date FROM fhy_reservoir_data where reservoir_id = '23' AND observation_time BETWEEN '2016-01-01' AND '2024-01-01 23:59';" 
        df_reservoir = pd.read_sql(query_reservoir, engine)
        
        print("=== 水庫資料表 (fhy_reservoir_data) 缺值統計 ===")
        print(f"總資料筆數: {len(df_reservoir)} 筆")
        
        missing_reservoir = df_reservoir.isnull().sum()
        if missing_reservoir.sum() == 0:
            print("🎉 所有欄位資料完整，沒有缺值！")
        else:
            print("⚠️ 以下欄位有缺漏值 (顯示缺漏筆數):")
            print(missing_reservoir[missing_reservoir > 0])
            
        print("-" * 40)

        # ==========================================
        # 4. 表格 2: 氣象站資料
        # ==========================================
        # ★ 請把 'codis_weatherdata' 換成您實際的氣象表名稱
        query_climate = f"""SELECT TO_CHAR("date", 'YYYY-MM-DD') AS obs_date,"PP01","TX01",COALESCE("TX02", "TX01") AS "TX02","RH01","WD01","PS01" FROM codis_weatherdata  where stno = 'C0D580' and date BETWEEN '2016-01-01' AND '2024-01-01';""" 
        # PP01 降水量 TX01 平均氣溫  TX02 最高氣溫 RH01 相對溼度 WD01 平均風速 PS01 測站平均氣壓
        df_climate = pd.read_sql(query_climate, engine)
        
        print("=== 氣候站資料表 缺值統計 ===")
        print(f"總資料筆數: {len(df_climate)} 筆")
        
        missing_climate = df_climate.isnull().sum()
        if missing_climate.sum() == 0:
            print("🎉 所有欄位資料完整，沒有缺值！")
        else:
            print("⚠️ 以下欄位有缺漏值 (顯示缺漏筆數):")
            print(missing_climate[missing_climate > 0])

    except Exception as e:
        print(f"❌ 執行過程中發生錯誤: {e}")
        
    finally:
        engine.dispose()
        print("\n資料庫連線已關閉，檢查完畢。")

# ==========================================
# 5. 程式的進入點 (叫 Python 真的去執行上面寫好的函數)
# ==========================================
if __name__ == "__main__":
    check_missing_values()