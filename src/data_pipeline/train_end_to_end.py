import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # 用來儲存 Scaler 工具
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# ==========================================
# 全域路徑設定 (根據您的環境配置)
# ==========================================
# 強制指定專案根目錄，確保絕對不會迷路
PROJECT_ROOT = Path("/root/scip-water-scarcity-gis-ai")
ENV_PATH = PROJECT_ROOT / "src" / ".env"

# 自動產生今天的日期字串 (例如: "20260615")
# 加上 %H%M%S (時分秒)
TODAY_STR = datetime.now().strftime("%Y%m%d_%H%M%S")

# 設定模型與圖片的存檔終點資料夾
SAVE_DIR = PROJECT_ROOT / "model" / TODAY_STR

# 如果資料夾不存在，就自動建立 (parents=True 代表會連同上層的 model 一起建)
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 第一階段：從資料庫撈取並清洗資料
# ==========================================
def fetch_and_prepare_data():
    print("=== 第一階段：從 PostgreSQL 撈取資料 ===")
    
    load_dotenv(dotenv_path=ENV_PATH)

    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "thesis_analysis")
    DB_USER = os.getenv("DB_USER", "sm245735")
    DB_PASS = os.getenv("DB_PASSWORD")

    if not DB_PASS:
        raise ValueError(f"❌ 找不到密碼！請確認 .env 檔案是否存在於 {ENV_PATH}")

    engine_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(engine_url)

    try:
        print("正在撈取水庫水位資料 (2016-2023)...")
        query_reservoir = """
        SELECT 
            TO_CHAR(observation_time, 'YYYY-MM-DD') AS obs_date,
            storage_rate 
        FROM fhy_reservoir_data 
        WHERE reservoir_id = '23' 
        AND observation_time >= '2016-01-01' AND observation_time < '2024-01-01';
        """ 
        df_reservoir = pd.read_sql(query_reservoir, engine)

        print("正在撈取氣象站資料 (2016-2023)...")
        query_climate = """
        SELECT 
            TO_CHAR("date", 'YYYY-MM-DD') AS obs_date, 
            "PP01", "TX01", COALESCE("TX02", "TX01") AS "TX02", 
            "RH01", "WD01", "PS01" 
        FROM codis_weatherdata  
        WHERE stno = 'C0D580' 
        AND "date" >= '2016-01-01' AND "date" < '2024-01-01';
        """ 
        df_climate = pd.read_sql(query_climate, engine)

        print("正在合併與對齊時間軸...")
        df_merged = pd.merge(df_climate, df_reservoir, on='obs_date', how='left')
        df_merged['obs_date'] = pd.to_datetime(df_merged['obs_date'])
        df_merged.set_index('obs_date', inplace=True)

        # 讓時間軸由舊到新乖乖排好隊
        df_merged = df_merged.sort_index()
        
        # 防禦性補值
        missing_before = df_merged['storage_rate'].isnull().sum()
        if missing_before > 0:
            print(f"⚠️ 進行時間序列線性內插補值 (補齊 {missing_before} 筆)...")
            df_merged['storage_rate'] = df_merged['storage_rate'].interpolate(method='time').bfill()

        print("✅ 資料準備完畢！")
        return df_merged

    finally:
        engine.dispose()

# ==========================================
# 第二階段：訓練 LSTM 模型
# ==========================================
def train_reservoir_lstm(df_merged):
    print("\n=== 第二階段：開始建構與訓練 LSTM ===")
    
    df_merged = df_merged.sort_index()
    features = ["PP01", "TX01", "TX02", "RH01", "WD01", "PS01", "storage_rate"]
    target = "storage_rate"

    print("1. 切割資料集 (Train: 2016-2021, Val: 2022, Test: 2023)")
    train_df = df_merged.loc['2016-01-01':'2021-12-31', features]
    val_df   = df_merged.loc['2022-01-01':'2022-12-31', features]
    test_df  = df_merged.loc['2023-01-01':'2023-12-31', features]

    print("2. 進行特徵正規化並儲存 Scaler 工具...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    val_scaled   = scaler.transform(val_df)
    test_scaled  = scaler.transform(test_df)

    # 【新增】將 Scaler 存檔
    scaler_path = SAVE_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"   💾 Scaler 已儲存至: {scaler_path}")

    target_idx = features.index(target)
    TIME_STEPS = 14  

    def create_sequences(data, time_steps, target_idx):
        X, Y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i : i + time_steps, :])
            Y.append(data[i + time_steps, target_idx])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_sequences(train_scaled, TIME_STEPS, target_idx)
    X_val, Y_val     = create_sequences(val_scaled, TIME_STEPS, target_idx)

    print("3. 搭建 LSTM 神經網路")
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("\n4. 開始訓練 (具有 EarlyStopping 機制)...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, Y_val),
        callbacks=[early_stop],
        verbose=1
    )

    # 讓程式自己印出到底跑了幾輪
    actual_epochs = len(history.history['loss'])
    print(f"\n💡 報告！模型實際只跑了 {actual_epochs} 輪就觸發 Early Stop 提早結束了！")

    print("\n=== 第三階段：存檔與繪製訓練成果 ===")
    
    # 【新增】將模型存檔 (.keras 格式是目前最新標準)
    model_path = SAVE_DIR / "lstm_model.keras"
    model.save(model_path)
    print(f"💾 模型已成功儲存至: {model_path}")

    # 【新增】繪製圖表並「存成圖片」，不直接顯示以防伺服器報錯
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss Over Epochs (Trained {actual_epochs} epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = SAVE_DIR / "loss_curve.png"
    plt.savefig(plot_path)
    plt.close() # 存檔後關閉圖表釋放記憶體
    print(f"📊 訓練成果圖已儲存至: {plot_path}")

    print(f"\n🎉 完整流程執行完畢！所有產出檔案都在 {SAVE_DIR} 裡面囉！")
    return model, scaler


# ==========================================
# 第三階段：使用 2023 年資料進行預測與驗證
# ==========================================
def evaluate_and_predict(model, scaler, df_merged):
    print("\n=== 第三階段：2023 期末考 (模型預測與驗證) ===")
    
    # 1. 抓出 2023 年的資料
    features = ["PP01", "TX01", "TX02", "RH01", "WD01", "PS01", "storage_rate"]
    target_idx = features.index("storage_rate")
    test_df = df_merged.loc['2023-01-01':'2023-12-31', features].copy()
    
    print(f"1. 取得 2023 年測試資料: {len(test_df)} 筆")

    # 2. 用「訓練時的同一把尺」進行縮放 (千萬不能重新 fit!)
    test_scaled = scaler.transform(test_df)

    # 3. 製作時間滑動視窗 (一樣看過去 14 天預測明天)
    TIME_STEPS = 14
    X_test, Y_test = [], []
    for i in range(len(test_scaled) - TIME_STEPS):
        X_test.append(test_scaled[i : i + TIME_STEPS, :])
        Y_test.append(test_scaled[i + TIME_STEPS, target_idx])
    
    X_test = np.array(X_test)
    Y_test = np.array(Y_test) # 這是真實水位的縮放版 (答案)

    # 4. 讓模型預測 2023 年的水位
    print("2. 模型正在寫考卷 (進行預測)...")
    predicted_scaled = model.predict(X_test)

    # 5. 魔法步驟：將 0~1 的縮放數字「還原」成真實的水位百分比
    print("3. 將預測分數還原為真實水位百分比...")
    # 建立一個跟特徵欄位一樣寬的假矩陣來反轉
    dummy_pred = np.zeros((len(predicted_scaled), len(features)))
    dummy_real = np.zeros((len(Y_test), len(features)))
    
    # 把我們的預測值跟真實值塞回 storage_rate 那個欄位
    dummy_pred[:, target_idx] = predicted_scaled[:, 0]
    dummy_real[:, target_idx] = Y_test

    # 執行逆轉換
    predicted_real = scaler.inverse_transform(dummy_pred)[:, target_idx]
    actual_real = scaler.inverse_transform(dummy_real)[:, target_idx]

    # 抓出對應的日期 (因為前 14 天被拿去當輸入特徵了，所以日期從第 15 天開始)
    test_dates = test_df.index[TIME_STEPS:]

    # 5.5 結算 2023 期末考成績單
    print("\n📝 正在批改 2023 年期末考卷...")
    
    # 計算 R-squared (R平方)
    r2 = r2_score(actual_real, predicted_real)
    
    # 計算 MAPE (為了換成百分比，所以乘上 100)
    mape = mean_absolute_percentage_error(actual_real, predicted_real) * 100
    
    print(f"📊 【模型期末考成績單】")
    print(f"👉 R平方 ($R^2$): {r2:.4f} (滿分為1，代表模型掌握了 {r2*100:.1f}% 的水位變化規律)")
    print(f"👉 MAPE (平均誤差): {mape:.2f}% (代表模型平均每天只猜偏了 {mape:.2f}%)")
    print(f"💡 白話直覺：您可以大略視為，這個模型預測水位有 {100 - mape:.2f}% 的『準確度』！")

    # 6. 畫出預測結果對比圖
    print("\n4. 正在繪製 2023 年【真實水位 vs 預測水位】對比圖...")
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, actual_real, label='Actual Water Level (真實水位)', color='blue', linewidth=2)
    plt.plot(test_dates, predicted_real, label='LSTM Predicted Level (預測水位)', color='red', linestyle='--', linewidth=2)
    
    plt.title('2023 Reservoir Water Level Prediction (LSTM)', fontsize=16)
    plt.xlabel('Date (日期)', fontsize=12)
    plt.ylabel('Storage Rate (%) (蓄水率)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # 存檔
    predict_plot_path = SAVE_DIR / "2023_prediction_results.png"
    plt.savefig(predict_plot_path)
    plt.close()
    print(f"📊 預測成果對比圖已成功儲存至: {predict_plot_path}")

# ==========================================
# 程式進入點
# ==========================================
if __name__ == "__main__":
    # 1. 撈取並清洗資料
    prepared_data = fetch_and_prepare_data()
    # 2. 訓練模型
    trained_model, fitted_scaler = train_reservoir_lstm(prepared_data)
    # 🌟 3. 新增這行：執行 2023 年的預測與畫圖
    evaluate_and_predict(trained_model, fitted_scaler, prepared_data)