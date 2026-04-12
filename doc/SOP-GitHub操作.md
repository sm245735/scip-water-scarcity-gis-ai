# GitHub 操作 SOP

> 適用專案：scip-water-scarcity-gis-ai
> 建立日期：2026-04-12
> 更新日期：2026-04-12

---

## 📁 專案基本資訊

| 項目 | 值 |
|------|-----|
| Repo URL | git@github.com:sm245735/scip-water-scarcity-gis-ai.git |
| 本機路徑 | /home/sm245735/.openclaw/workspace/scip-water-scarcity-gis-ai |
| 預設分支 | main |

---

## 🔑 SSH Key 設定（首次設定）

### 測試連線
```bash
ssh -T git@github.com
```
預期輸出：`Hi sm245735/scip-water-scarcity-gis-ai! You've successfully authenticated...`

### 疑難排解
- 若詢問是否繼續連線，輸入 `yes`
- 若認證失敗，確認 SSH key 已加入 GitHub（Settings → SSH and GPG keys）

---

## 🚀 日常工作流程

### 1. 第一次 Clone（新機器）
```bash
git clone git@github.com:sm245735/scip-water-scarcity-gis-ai.git
cd scip-water-scarcity-gis-ai
```

### 2. 新增/修改程式碼後 Commit
```bash
git add .
git commit -m "feat: add TCCIP data cleaning script for Hsinchu"
```

### 3. Commit 訊息格式
```
<類型>: <英文描述> (<中文補充>)

範例：
feat: add LSTM model for reservoir inflow prediction (新增水庫入流量預測 LSTM 模型)
fix: resolve PostGIS geometry column mismatch (修正 PostGIS 幾何欄位對應錯誤)
docs: update README with database schema (更新資料庫結構說明)
```

**類型列表：**
- `feat:` 新功能
- `fix:` 錯誤修正
- `docs:` 文件更新
- `refactor:` 重構
- `test:` 測試相關
- `chore:` 維護/建構相關

### 4. Push 到 GitHub
```bash
git push origin main
```

---

## 🔐 機密資訊管理（重要！）

### 嚴禁將密碼寫進 Git！
**`1qaz@WSX` 慘痛案例：** 密碼進了 GitHub 就等於公開，必須用 git history rewrite 才能清除。

### 使用 .env 環境變數
1. **建立 `.env` 檔案（不上 Git）**
```bash
# .env（此檔案不进 Git）
DB_HOST=db
DB_PORT=5432
DB_NAME=postgres
DB_USER=sm245735
DB_PASSWORD=1qaz@WSX
```

2. **在 `.gitignore` 中排除 `.env`**
```
.env
```

3. **Python 程式這樣讀取**
```python
import os
from dotenv import load_dotenv

load_dotenv()  # 讀取 .env 檔案

DB_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
```

4. **`docker-compose.yml` 中引用**
```yaml
environment:
  DB_HOST: ${DB_HOST}
  DB_PASSWORD: ${DB_PASSWORD}
  # ...
```

5. **第一次設定 .env 範例**
```bash
# 建立 .env 檔案
cp .env.example .env
# 編輯填入實際密碼
nano .env
```

### 若密碼已進 Git（緊急處理）
```bash
# 安裝 git-filter-repo
pip install git-filter-repo --break-system-packages

# 重寫歷史（把所有 "舊密碼" 替換成 "DB_PASSWORD_PLACEHOLDER"）
git filter-repo --replace-text <(echo "舊密碼==>DB_PASSWORD_PLACEHOLDER") --force

# 重新設定 remote 並強制推送
git remote add origin git@github.com:sm245735/scip-water-scarcity-gis-ai.git
git push -u origin main --force
```

---

## ⚠️ 常見問題

### Q: Push 被拒絕（remote 有新 commits）
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

### Q: 想放棄本地修改，強制同步遠端
```bash
git fetch origin
git reset --hard origin/main
```

### Q: 想看目前 commit 狀態
```bash
git status
git log --oneline -5
```

---

## 📂 .gitignore 內容（已驗證）

```
# 論文原始大資料不進 Git（重要！）
data/
*.csv
*.xlsx
*.zip

# Python 垃圾檔
__pycache__/
*.pyc
*.py[cod]
*$py.class
*.egg-info/
.eggs/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Docker
.docker/
.dockerignore

# 虛擬環境
venv/
.venv/

# 環境變數（存資料庫密碼，不能公開）
.env
```

---

## 📝 文件更新 SOP

**每次完成新的開發工作，原則上要同步更新以下文件：**

### 1. `doc/論文開發環境.md` — 開發日誌（第一優先）
**目的：** 當成工作日誌，快速查詢做過什麼
**更新頻率：** 每次完成新功能就補一列

**格式：**
```
| 2026-04-12 | 水庫蓄水範圍 Shapefile 匯入 |
| | - 資料：`data/水資源（水庫蓄水）/ressub.shp`（129 個水庫） |
| | - 腳本：`src/gis_analysis/水庫蓄水範圍匯入.py` |
| | - 執行：`docker exec thesis_python_dev python /app/src/gis_analysis/水庫蓄水範圍匯入.py` |
```

### 2. `doc/論文研究資料庫.md` — 資料庫結構與技術細節
**目的：** 記錄資料庫結構、技術問題與解法（給日後發表或 Debug 用）

**何時更新：**
- 新增資料表時
- 遇到技術問題並解決時（如 URL encode、ST_Force3DZ 等）

### 3. Commit 訊息（Git 版本歷程）
**格式：** `類型: 英文描述 (中文補充)`
**參考上方 Commit 訊息格式章節**

### 文件更新順序建議
1. 先完成功能（讓程式跑起來）
2. 馬上更新 `doc/論文開發環境.md` 的開發日誌（最快）
3. 若有技術細節要記錄，再補充 `doc/論文研究資料庫.md`
4. **若資料庫 schema 有變更，同步更新 `database/schema.sql`**
5. 最後 Git commit + push

**📌 重要：`database/schema.sql` 是資料庫結構的最後堡壘！**

每次新增或修改資料表，都要把 SQL schema 同步更新到 `database/schema.sql`。
例如：
- 新增 `reservoir_boundaries` 表 → 在 `schema.sql` 補上 `CREATE TABLE`
- 新增索引或 constraint → 也要寫進 `schema.sql`

```sql
-- database/schema.sql 範例
CREATE TABLE IF NOT EXISTS reservoir_boundaries (
    id SERIAL PRIMARY KEY,
    res_name VARCHAR(100),
    area_description TEXT,
    source VARCHAR(200),
    build_date VARCHAR(20),
    geom GEOMETRY(GeometryZ, 4326),
    created_at TIMESTAMP DEFAULT '2026-04-12'
);
```

