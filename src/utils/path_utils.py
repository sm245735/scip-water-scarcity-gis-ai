#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路徑工具模組
提供專案根目錄、相對路徑解析功能
讓所有腳本擺脫 hardcoded 絕對路徑
"""

from pathlib import Path
import os

# 優先讀取環境變數，否則從本模組位置推算 repo 根目錄
_PROJECT_ROOT = os.getenv("PROJECT_ROOT", "")
if _PROJECT_ROOT:
    PROJECT_ROOT = Path(_PROJECT_ROOT)
else:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


# 常見相對路徑
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "doc"
LOGS_DIR = PROJECT_ROOT / "logs"
SRC_DIR = PROJECT_ROOT / "src"
GIS_DATA_DIR = DATA_DIR / "03_GIS"
RESERVOIR_DATA_DIR = DATA_DIR / "水資源（水庫蓄水）"
CATCHMENT_DATA_DIR = DATA_DIR / "05_集水區"
TCCIP_DATA_DIR = DATA_DIR / "03. 資料" / "01. 氣候（TCCIP）"


def get_data_path(*segments) -> Path:
    """解析 data 目錄下的相對路徑"""
    return DATA_DIR.joinpath(*segments)


def get_log_path(filename: str) -> Path:
    """解析 logs 目錄下的檔案路徑（自動建立目錄）"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / filename
