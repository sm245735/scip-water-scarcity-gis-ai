#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試資料清理函式（parse_num）的正確性

背景：水利署 Statistics.aspx 的欄位格式很不統一：
    - 缺值用 '--'、'-'、或空字串
    - 大數字用千分號（如 '13,882.07'）
    - 百分比含 % 符號（如 '68.95 %'）
    - 有時候前後有空白

這支測試鎖定這些邊界情況，確保資料匯入不會悄悄出錯。

執行：pytest tests/test_data_parsing.py -v
"""

import pytest


# ============================================================================
# 把 parse_num 從 src/data_pipeline/水庫Statistics_每日收集_host.py 抽出來
# （因為該檔案名稱含中文，import 有點麻煩，這裡重新寫一份測試用的版本，
#  實際改動時要同步更新兩邊——或者把 parse_num 搬到共用的 utils 檔案）
# ============================================================================
def parse_num(val, remove_comma=False, remove_percent=False):
    """
    跟 src/data_pipeline/水庫Statistics_每日收集_host.py 的 parse_num
    是同一份邏輯，複製過來方便測試。
    """
    if not val or val.strip() in ('--', '', '-'):
        return None
    v = val.strip()
    if remove_comma:
        v = v.replace(',', '')
    if remove_percent:
        v = v.replace('%', '').strip()
    try:
        return float(v)
    except ValueError:
        return None


# ============================================================================
# 缺值處理
# ============================================================================
class TestParseNumMissing:
    def test_double_dash_returns_none(self):
        """水利署最常見的缺值表示法"""
        assert parse_num("--") is None

    def test_single_dash_returns_none(self):
        assert parse_num("-") is None

    def test_empty_string_returns_none(self):
        assert parse_num("") is None

    def test_whitespace_only_returns_none(self):
        assert parse_num("   ") is None

    def test_dash_with_whitespace(self):
        assert parse_num(" -- ") is None


# ============================================================================
# 千分號處理（effective_storage 欄位）
# ============================================================================
class TestParseNumComma:
    def test_parses_large_number_with_comma(self):
        """寶二水庫蓄水量常見格式：13,882.07（萬立方公尺）"""
        assert parse_num("13,882.07", remove_comma=True) == 13882.07

    def test_parses_multiple_commas(self):
        assert parse_num("1,234,567.89", remove_comma=True) == 1234567.89

    def test_no_comma_still_works(self):
        assert parse_num("100.5", remove_comma=True) == 100.5

    def test_without_flag_comma_breaks_parse(self):
        """沒設 remove_comma=True，有千分號的字串應該回 None"""
        assert parse_num("13,882.07") is None


# ============================================================================
# 百分號處理（storage_rate 欄位）
# ============================================================================
class TestParseNumPercent:
    def test_parses_percent_with_space(self):
        """水利署格式：68.95 %"""
        assert parse_num("68.95 %", remove_percent=True) == 68.95

    def test_parses_percent_no_space(self):
        assert parse_num("68.95%", remove_percent=True) == 68.95

    def test_parses_over_hundred_percent(self):
        """颱風過後蓄水率會超過 100%（溢流），schema 為此調整為 NUMERIC(7,2)"""
        assert parse_num("105.20 %", remove_percent=True) == 105.20

    def test_parses_zero_percent(self):
        assert parse_num("0 %", remove_percent=True) == 0.0


# ============================================================================
# 錯誤資料
# ============================================================================
class TestParseNumInvalid:
    def test_non_numeric_returns_none(self):
        assert parse_num("abc") is None

    def test_mixed_content_returns_none(self):
        assert parse_num("12abc34") is None

    def test_scientific_notation_works(self):
        """Python float() 可以吃科學記號，這是 side effect，但也算是合理行為"""
        assert parse_num("1e3") == 1000.0


# ============================================================================
# 正常數值
# ============================================================================
class TestParseNumNormal:
    def test_positive_integer(self):
        assert parse_num("100") == 100.0

    def test_negative_number(self):
        assert parse_num("-5.5") == -5.5

    def test_decimal(self):
        assert parse_num("3.14159") == pytest.approx(3.14159)

    def test_leading_trailing_whitespace_stripped(self):
        assert parse_num("  42.0  ") == 42.0
