# tests/ 測試說明

本資料夾存放專案的自動化測試，使用 **pytest** 框架。

---

## 為什麼要寫測試？

論文 repo 的測試目的**不是達到 100% 覆蓋率**，而是：

1. **確保關鍵函式正確**：資料清理、指標計算、SQL 邏輯若有 bug，會直接影響論文結果
2. **方便重現**：未來學弟妹 clone 下來 `pytest`，馬上知道環境對不對
3. **改動時不翻車**：當你調整 `utils_metrics.py` 的 NSE 公式時，測試會立刻告訴你有沒有破壞其他計算

---

## 快速開始

```bash
# 安裝 pytest（若還沒裝）
pip install pytest

# 執行全部測試
cd scip-water-scarcity-gis-ai
pytest

# 執行特定測試檔
pytest tests/test_metrics.py

# 顯示詳細輸出
pytest -v

# 只跑失敗的測試（上次跑完後）
pytest --lf
```

---

## 測試分類

| 檔案 | 測什麼 | 需要資料庫？ |
|------|--------|-------------|
| `test_metrics.py` | 水文評估指標（MAE/RMSE/NSE/KGE/PBIAS）計算正確性 | 否 |
| `test_data_parsing.py` | 資料清理函式（處理 `--`、千分號、百分號等） | 否 |
| `test_training_set.py` | 訓練集結構（欄位、日期連續性、缺值比例） | 是（需 `寶二訓練集_v1.csv`） |

**不需要資料庫的測試**：隨時可以跑，應該永遠通過。
**需要資料庫的測試**：先跑過 `01_build_training_set.py` 產出 CSV 之後才能測。

---

## 撰寫測試的原則

1. **測試一件事**：每個 `test_xxx` 函式只驗證一個行為
2. **取名要白話**：`test_nse_is_zero_when_predicting_mean` 比 `test_nse_1` 好
3. **邊界情況先測**：NaN、空 array、全部相同、單一樣本
4. **用已知答案**：例如「完美預測」的 NSE 必須等於 1.0，這是可以確定的

---

## 範例：新增一個測試

```python
# tests/test_data_parsing.py
from src.data_pipeline.水庫Statistics_每日收集_host import parse_num

def test_parse_num_handles_dash():
    """水利署常用 '--' 表示缺值，應該回傳 None"""
    assert parse_num("--") is None
    assert parse_num("") is None
    assert parse_num("-") is None

def test_parse_num_removes_comma():
    """大數字含千分號（如 13,882.07）要能正確解析"""
    assert parse_num("13,882.07", remove_comma=True) == 13882.07

def test_parse_num_removes_percent():
    """蓄水率含 % 符號（如 68.95 %）要能正確解析"""
    assert parse_num("68.95 %", remove_percent=True) == 68.95
```

跑法：`pytest tests/test_data_parsing.py -v`
