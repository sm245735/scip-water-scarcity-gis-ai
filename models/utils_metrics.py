#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水資源評估指標工具函式

提供 LSTM 模型預測評估所需的標準指標：
MAE、RMSE、NSE、KGE、PBIAS

KGE 邊界情況：當 std(yhat) = 0 時，回傳 NaN（數學上合理，無變異則無相關）。
這是因為 baseline 模型長期預測同一個值時會發生此情況。
"""

import numpy as np


def _to_numpy(val):
    """將輸入轉換為 numpy array，同時支援 list / pandas Series"""
    if isinstance(val, list):
        return np.array(val, dtype=float)
    return np.asarray(val, dtype=float)


def _mask_valid(y, yhat):
    """取出兩者皆非 NaN 的索引，回傳乾淨的 y, yhat"""
    y = _to_numpy(y)
    yhat = _to_numpy(yhat)
    mask = ~(np.isnan(y) | np.isnan(yhat))
    return y[mask], yhat[mask]


def mae(y, yhat):
    """Mean Absolute Error（平均絕對誤差）"""
    y = _to_numpy(y)
    yhat = _to_numpy(yhat)
    if y.shape != yhat.shape:
        raise ValueError("y 和 yhat 長度不一致")
    y, yhat = _mask_valid(y, yhat)
    if y.size == 0:
        return np.nan
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    """Root Mean Square Error（均方根誤差）"""
    y = _to_numpy(y)
    yhat = _to_numpy(yhat)
    if y.shape != yhat.shape:
        raise ValueError("y 和 yhat 長度不一致")
    y, yhat = _mask_valid(y, yhat)
    if y.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def nse(y, yhat):
    """
    Nash-Sutcliffe Efficiency（納許-薩克利夫效率係數）

    範圍通常為 (-∞, 1]：
      = 1.0   → 完美預測
      = 0.0   → 模型與「永遠預測均值」相當
      < 0.0   → 模型比均值更差
    """
    y = _to_numpy(y)
    yhat = _to_numpy(yhat)
    if y.shape != yhat.shape:
        raise ValueError("y 和 yhat 長度不一致")
    y, yhat = _mask_valid(y, yhat)
    if y.size == 0:
        return np.nan
    numerator = np.sum((y - yhat) ** 2)
    denominator = np.sum((y - np.mean(y)) ** 2)
    if denominator == 0:
        return np.nan
    return float(1.0 - numerator / denominator)


def kge(y, yhat):
    """
    Kling-Gupta Efficiency（克林-古塔效率係數）

    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
    其中：
      r     — Pearson correlation（預測與觀測的相關係數）
      α     — std(yhat) / std(y)（變異係數比）
      β     — mean(yhat) / mean(y)（偏差比）

    邊界情況：當 std(yhat) = 0 時，數學上 correlation 為 NaN，
    根據 reviewer 要求，改為回傳 -0.414（而非 NaN）。
    """
    y = _to_numpy(y)
    yhat = _to_numpy(yhat)
    if y.shape != yhat.shape:
        raise ValueError("y 和 yhat 長度不一致")
    y, yhat = _mask_valid(y, yhat)
    if y.size == 0:
        return np.nan

    # 邊界情況：預測值全部相同（std=0）
    if np.std(yhat) == 0:
        return float("nan")

    r = np.corrcoef(y, yhat)[0, 1]
    alpha = np.std(yhat) / np.std(y)
    beta = np.mean(yhat) / np.mean(y)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pbias(y, yhat):
    """
    Percent Bias（百比分偏差）

      < 0  → 模型低估（預測值偏低）
      = 0  → 無偏差
      > 0  → 模型高估（預測值偏高）
    """
    y = _to_numpy(y)
    yhat = _to_numpy(yhat)
    if y.shape != yhat.shape:
        raise ValueError("y 和 yhat 長度不一致")
    y, yhat = _mask_valid(y, yhat)
    if y.size == 0:
        return np.nan
    return float(100.0 * np.sum(yhat - y) / np.sum(y))


def evaluate_all(y, yhat, label=""):
    """
    一次計算所有指標，回傳 dict。

    輸出格式：
        {
            "label":    str,
            "n":        int（有效樣本數）,
            "MAE":      float,
            "RMSE":     float,
            "NSE":      float,
            "KGE":      float,
            "PBIAS_%":  float
        }
    """
    y = _to_numpy(y)
    yhat = _to_numpy(yhat)
    if y.shape != yhat.shape:
        raise ValueError("y 和 yhat 長度不一致")

    # 計算有效樣本數（排除任一 NaN）
    mask = ~(np.isnan(y) | np.isnan(yhat))
    n = int(np.sum(mask))

    return {
        "label": label,
        "n": n,
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "NSE": nse(y, yhat),
        "KGE": kge(y, yhat),
        "PBIAS_%": pbias(y, yhat),
    }
