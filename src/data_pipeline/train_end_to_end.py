import os
import sys
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
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

# ==========================================
# 全域路徑設定 (根據您的環境配置)
# ==========================================
PROJECT_ROOT = Path("/root/scip-water-scarcity-gis-ai")
ENV_PATH = PROJECT_ROOT / "src" / ".env"

# 自動產生今天的日期字串與時間 (時分秒，防覆蓋)
TODAY_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
# 🆕 SAVE_DIR 改名: 支援 seed 參數 (從 sys.argv[1] 讀), 沒給就是 seed=42
# 用法: python train_end_to_end.py [SEED]
_RUN_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SAVE_DIR = PROJECT_ROOT / "model" / f"{TODAY_STR}_seed{_RUN_SEED}"
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
        # 🌟 修改：年份拉長從 2014 到 2025 底 (即小於 2026)
        print("正在撈取水庫水位資料 (2014-2025)...")
        query_reservoir = """
        SELECT 
            TO_CHAR(data_date, 'YYYY-MM-DD') AS obs_date,
            storage_rate 
        FROM fhy_reservoir_data 
        WHERE reservoir_id = '23' 
        AND data_date >= '2014-01-01' AND data_date < '2026-01-01';
        """ 
        df_reservoir = pd.read_sql(query_reservoir, engine)

        # ==========================================
        # 撈取氣象站資料: codis_weatherdata_v2
        # 8 個站一次撈, 不在 SQL 端過濾負值/異常值 (這批 CWA_Data/ 已清乾淨,
        # 留 NULL 就好; ETL 不負責補值)
        # ==========================================
        CLIMATE_COLS = ["PP01", "TX01", "TX02", "RH01", "WD01", "PS01"]
        # 注意: codis_weatherdata_v2 的欄位在 PG 內被 fold 成小寫 (pp01/tx01/...),
        # 所以 query 用小寫 + 引號明確 escape
        CLIMATE_COLS_SQL = ", ".join(f'"{c.lower()}"' for c in CLIMATE_COLS)
        print("正在撈取氣象站資料 (2014-2025, 8 站) ...")
        query_climate_raw = f"""
        SELECT
            TO_CHAR("date", 'YYYY-MM-DD') AS obs_date,
            stno,
            {CLIMATE_COLS_SQL}
        FROM codis_weatherdata_v2
        WHERE "date" >= '2014-01-01' AND "date" < '2026-01-01';
        """
        df_climate_raw = pd.read_sql(query_climate_raw, engine)
        # PG 回傳小寫欄名, rename 回我們慣用的大寫命名 (跟 v1 一致, 後續 pipeline 看得懂)
        df_climate_raw = df_climate_raw.rename(
            columns={c.lower(): c for c in CLIMATE_COLS}
        )

        print("正在合併與對齊時間軸...")
        df_reservoir['obs_date'] = pd.to_datetime(df_reservoir['obs_date'])
        df_climate_raw['obs_date'] = pd.to_datetime(df_climate_raw['obs_date'])

        # ==========================================
        # 跨站平均: 每個 (obs_date, element) 對所有「有值」的站台取平均
        # - 雨量站 C1D410/C1D420 在 TX01/TX02/RH01/WD01/PS01 是 NULL,
        #   groupby.mean() 自動跳過 NULL → 它們不會污染非 PP01 的平均
        # - PP01 欄位 8 站都有, 所以 PP01 平均 = 8 站平均
        # - TX01/TX02/RH01/WD01/PS01 平均 = 當天 5 個氣象站/農業站平均
        # ==========================================
        print("正在計算跨站每日平均 (NULL 不參與平均)...")
        df_daily_avg = (
            df_climate_raw
            .groupby('obs_date')[CLIMATE_COLS]
            .mean()                # pandas 預設 skipna=True, NULL 不計入
            .reset_index()
        )

        df_merged = pd.merge(df_daily_avg, df_reservoir, on='obs_date', how='left')
        df_merged = df_merged.set_index('obs_date').sort_index()

        # ==========================================
        # 🌟 補值: 先看 avg 後還有沒有缺值 (极罕見, 但 CWA 偶爾會全站當天缺)
        # 用 time-based interpolate + bfill/ffill (跟原本的補值精神一致)
        # storage_rate 也一起補 (水庫偶爾有缺測日), 確保 LSTM 餵進去的 X 全是 finite
        # ==========================================
        ALL_COLS = CLIMATE_COLS + ["storage_rate"]
        if df_merged[ALL_COLS].isnull().values.any():
            nulls_per_col = df_merged[ALL_COLS].isnull().sum()
            nonzero = nulls_per_col[nulls_per_col > 0]
            print(f"⚠️ 補值前仍有缺漏: {nonzero.to_dict()}")
            print("   正在用時序線性內插 + 雙向補齊 ...")
            df_merged[ALL_COLS] = (
                df_merged[ALL_COLS]
                .interpolate(method='time', limit_direction='both')
                .bfill()
                .ffill()
            )
        else:
            print("✅ 補值前無缺漏, 不需補值")

        # 🛡️ 防呆: 補完後還有 NaN/Inf 就大聲叫出來, 不要讓 NaN 偷渡進 LSTM
        n_nan = int(df_merged[ALL_COLS].isnull().values.sum())
        n_inf = int(np.isinf(df_merged[ALL_COLS].to_numpy()).sum())
        if n_nan or n_inf:
            raise ValueError(
                f"❌ 補值後仍有 NaN={n_nan}, Inf={n_inf}, 不能餵給 LSTM. "
                f"請檢查 {ALL_COLS} 的來源資料。"
            )

        print(f"✅ 資料準備完畢! 形狀={df_merged.shape}, 欄位={list(df_merged.columns)}")
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

    # 🌟 修改：重新切分 Train / Val / Test 的年份
    print("1. 切割資料集 (Train: 2014-2022, Val: 2023, Test: 2024-2025)")
    train_df = df_merged.loc['2014-01-01':'2022-12-31', features]
    val_df   = df_merged.loc['2023-01-01':'2023-12-31', features]
    # Test 留到第三階段再切

    print("2. 進行特徵正規化並儲存 Scaler 工具...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    val_scaled   = scaler.transform(val_df)

    scaler_path = SAVE_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"   💾 Scaler 已儲存至: {scaler_path}")

    target_idx = features.index(target)
    TIME_STEPS = 14  # 給前面14天的資料
    LEAD_TIME = 14  # 🌟 新增：設定你想預測幾天後？(可以自由改成 7 或 14)

    def create_sequences_lead_time(data, time_steps, target_idx, lead_time):
        X, Y = [], []
        # ⚠️ 注意：迴圈的邊界要多扣掉 lead_time，確保未來的某一天真的有答案可以對齊
        for i in range(len(data) - time_steps - lead_time + 1):
            X.append(data[i : i + time_steps, :]) # 過去 14 天的題目
            
            # 🌟 關鍵修改：答案不再是緊接著的隔天，而是跳到 lead_time 天後的那一天
            Y.append(data[i + time_steps + lead_time - 1, target_idx])
        return np.array(X), np.array(Y)

    # 呼叫新函數製作資料
    X_train, Y_train = create_sequences_lead_time(train_scaled, TIME_STEPS, target_idx, LEAD_TIME)
    X_val, Y_val     = create_sequences_lead_time(val_scaled, TIME_STEPS, target_idx, LEAD_TIME)

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

    actual_epochs = len(history.history['loss'])
    print(f"\n💡 報告！模型實際只跑了 {actual_epochs} 輪就觸發 Early Stop 提早結束了！")
    # 🆕 機器可讀格式: 給 10-run wrapper 抓 epoch 數
    print(f"@@@EPOCHS@@@{actual_epochs}@@@")

    print("\n=== 儲存模型與訓練圖表 ===")
    model_path = SAVE_DIR / "lstm_model.keras"
    model.save(model_path)
    
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
    plt.close() 

    return model, scaler

# ==========================================
# 第三階段：使用 2024-2025 年資料進行預測、評估與特徵重要性
# ==========================================
def evaluate_and_predict(model, scaler, df_merged):
    print("\n=== 第三階段：2024-2025 最終大考 (模型預測與驗證) ===")
    
    features = ["PP01", "TX01", "TX02", "RH01", "WD01", "PS01", "storage_rate"]
    target_idx = features.index("storage_rate")
    
    # 🌟 修改：切出 2024~2025 的大考資料
    test_df = df_merged.loc['2024-01-01':'2025-12-31', features].copy()
    print(f"1. 取得 2024-2025 年大考測試資料: {len(test_df)} 筆")

    test_scaled = scaler.transform(test_df)

    TIME_STEPS = 14
    LEAD_TIME = 14 # 🌟 與訓練時保持一致
    X_test, Y_test = [], []
    for i in range(len(test_scaled) - TIME_STEPS - LEAD_TIME + 1):
        X_test.append(test_scaled[i : i + TIME_STEPS, :])
        Y_test.append(test_scaled[i + TIME_STEPS + LEAD_TIME - 1, target_idx])
    
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print("2. 模型正在寫考卷 (進行預測)...")
    predicted_scaled = model.predict(X_test)

    print("3. 將預測分數還原為真實水位百分比...")
    dummy_pred = np.zeros((len(predicted_scaled), len(features)))
    dummy_real = np.zeros((len(Y_test), len(features)))
    
    dummy_pred[:, target_idx] = predicted_scaled[:, 0]
    dummy_real[:, target_idx] = Y_test

    predicted_real = scaler.inverse_transform(dummy_pred)[:, target_idx]
    actual_real = scaler.inverse_transform(dummy_real)[:, target_idx]
    # 抓出對應的日期 (因為前 14 天當特徵，且要預測 7 天後，所以日期要往後跳)
    test_dates = test_df.index[TIME_STEPS + LEAD_TIME - 1:]

    print("\n📝 正在批改期末考卷...")
    r2 = r2_score(actual_real, predicted_real)
    mape = mean_absolute_percentage_error(actual_real, predicted_real) * 100
    accuracy = 100 - mape

    # 🌟 建立要存入 txt 檔案的成績單文字
    metrics_text = f"""📊 【模型期末考成績單】
👉 R平方 ($R^2$): {r2:.4f} (滿分為1，代表模型掌握了 {r2*100:.1f}% 的水位變化規律)
👉 MAPE (平均誤差): {mape:.2f}% (代表模型平均每天只猜偏了 {mape:.2f}%)
💡 白話直覺：您可以大略視為，這個模型預測水位有 {accuracy:.2f}% 的『準確度』！

1. R 平方 ($R^2$ Score / 決定係數)這是什麼： 統計學上最標準的考卷分數。最高是 1（滿分），最爛可以到 0 或負數。
白話文： 如果 R^2 算出來是 {r2:.4f}，您可以很有自信地跟教授說：「我的模型成功解釋了水庫 {r2*100:.1f}% 的變化規律！」（這非常接近一般人理解的正確率）。
"""
    print(metrics_text)
    # 🆕 機器可讀格式: 給 10-run wrapper 抓 R²/MAPE
    print(f"@@@METRICS@@@R2={r2:.6f}@@@MAPE={mape:.6f}@@@ACC={accuracy:.6f}@@@")

    # 【新增】將成績單存檔
    txt_path = SAVE_DIR / "準確率.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)
    print(f"💾 成績單已自動儲存至: {txt_path}")

    print("\n4. 正在繪製 2024-2025 年【真實水位 vs 預測水位】對比圖...")
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, actual_real, label='Actual Water Level (真實水位)', color='blue', linewidth=2)
    plt.plot(test_dates, predicted_real, label='LSTM Predicted Level (預測水位)', color='red', linestyle='--', linewidth=2)
    plt.title('2024-2025 Reservoir Water Level Prediction (LSTM)', fontsize=16)
    plt.xlabel('Date (日期)', fontsize=12)
    plt.ylabel('Storage Rate (%) (蓄水率)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    predict_plot_path = SAVE_DIR / "2024_2025_prediction_results.png"
    plt.savefig(predict_plot_path)
    plt.close()

    # ==========================================
    # 🌟 新增：第四階段：計算並繪製特徵重要性 (Permutation Importance)
    # ==========================================
    print("\n=== 第四階段：計算特徵重要性 (打亂特徵法) ===")
    baseline_mse = mean_squared_error(Y_test, predicted_scaled[:, 0])
    importances = {}
    
    for i, col_name in enumerate(features):
        # 複製一份原始測試資料
        X_test_shuffled = X_test.copy()
        # 隨機打亂該特徵的所有時間序列資料
        np.random.shuffle(X_test_shuffled[:, :, i])
        
        # 用打亂後的假資料讓模型考一次
        shuffled_pred = model.predict(X_test_shuffled, verbose=0)
        shuffled_mse = mean_squared_error(Y_test, shuffled_pred[:, 0])
        
        # 誤差增加越多，代表這個特徵越不該被打亂 (越重要)
        importances[col_name] = shuffled_mse - baseline_mse

    # 畫成水平長條圖並存檔
    imp_series = pd.Series(importances).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    imp_series.plot(kind='barh', color='teal')
    plt.title('Feature Importance (LSTM Permutation Method)')
    plt.xlabel('Increase in MSE when feature is shuffled (特徵打亂後的誤差增加量)')
    plt.grid(True)
    
    imp_plot_path = SAVE_DIR / "feature_importance.png"
    plt.savefig(imp_plot_path)
    plt.close()
    print(f"📊 特徵重要性分析圖已儲存至: {imp_plot_path}")
    print(f"\n🎉 完整流程執行完畢！所有產出檔案都在 {SAVE_DIR} 裡面囉！")


# ==========================================
# 程式進入點
# ==========================================
if __name__ == "__main__":
    # 🆕 支援 seed CLI 參數 (預設 42, 用於 10-run 重現性實驗)
    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    print(f"🎲 SEED = {SEED}")
    # 全域隨機種子 (numpy + python random + tensorflow)
    import random as _random
    _random.seed(SEED)
    np.random.seed(SEED)
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except ImportError:
        pass

    prepared_data = fetch_and_prepare_data()
    trained_model, fitted_scaler = train_reservoir_lstm(prepared_data)
    evaluate_and_predict(trained_model, fitted_scaler, prepared_data)