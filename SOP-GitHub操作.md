# GitHub 操作 SOP

> 適用專案：scip-water-scarcity-gis-ai
> 建立日期：2026-04-12

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

## 📂 .gitignore 內容
```
# 論文原始大資料不進 Git
data/
*.csv
*.xlsx

# Python
__pycache__/
*.pyc
*.egg-info/
.eggs/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```
