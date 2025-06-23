"""
工具函数模块

提供通用的辅助函数
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import streamlit as st
from datetime import datetime, timedelta


def detect_time_frequency(dates: pd.Series) -> str:
    """
    检测时间序列的频率
    
    Args:
        dates: 日期序列
        
    Returns:
        str: 频率描述
    """
    if len(dates) < 2:
        return "未知"
    
    # 计算时间间隔
    time_diffs = dates.diff().dropna()
    most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
    
    days = most_common_diff.days
    if days == 1:
        return "日"
    elif days == 7:
        return "周"
    elif 28 <= days <= 31:
        return "月"
    elif 365 <= days <= 366:
        return "年"
    else:
        return f"{days}天"


def validate_time_series_data(data: pd.DataFrame, 
                            date_col: str, 
                            value_col: str) -> Tuple[bool, List[str]]:
    """
    验证时间序列数据的有效性
    
    Args:
        data: 数据框
        date_col: 日期列名
        value_col: 数值列名
        
    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误信息列表)
    """
    errors = []
    
    # 检查列是否存在
    if date_col not in data.columns:
        errors.append(f"未找到日期列：{date_col}")
    
    if value_col not in data.columns:
        errors.append(f"未找到数值列：{value_col}")
    
    if errors:
        return False, errors
    
    # 检查数据类型
    try:
        pd.to_datetime(data[date_col])
    except:
        errors.append(f"日期列 '{date_col}' 无法转换为日期格式")
    
    try:
        pd.to_numeric(data[value_col], errors='coerce')
    except:
        errors.append(f"数值列 '{value_col}' 无法转换为数值格式")
    
    # 检查数据量
    if len(data) < 5:
        errors.append("数据量过少（少于5个数据点），无法进行有效预测")
    
    # 检查缺失值比例
    missing_ratio = data[[date_col, value_col]].isnull().sum().sum() / (len(data) * 2)
    if missing_ratio > 0.5:
        errors.append("缺失值比例过高（超过50%），可能影响预测质量")
    
    return len(errors) == 0, errors


def calculate_prediction_metrics(actual: np.ndarray, 
                               predicted: np.ndarray) -> Dict[str, float]:
    """
    计算预测评估指标
    
    Args:
        actual: 实际值
        predicted: 预测值
        
    Returns:
        Dict[str, float]: 评估指标
    """
    if len(actual) != len(predicted) or len(actual) == 0:
        return {}
    
    # 避免除零错误
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 基本指标
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    # MAPE (平均绝对百分比误差)
    non_zero_mask = actual != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    # R²决定系数
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }


def format_number(value: float, precision: int = 2) -> str:
    """
    格式化数字显示
    
    Args:
        value: 数值
        precision: 小数位数
        
    Returns:
        str: 格式化后的字符串
    """
    if abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def create_date_range(start_date: datetime, 
                     periods: int, 
                     frequency: str = 'D') -> pd.DatetimeIndex:
    """
    创建日期范围
    
    Args:
        start_date: 开始日期
        periods: 期间数
        frequency: 频率
        
    Returns:
        pd.DatetimeIndex: 日期索引
    """
    return pd.date_range(start=start_date, periods=periods, freq=frequency)


def split_train_test(data: pd.DataFrame, 
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    分割训练和测试数据
    
    Args:
        data: 时间序列数据
        test_size: 测试集比例
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (训练集, 测试集)
    """
    split_point = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_point].copy()
    test_data = data.iloc[split_point:].copy()
    
    return train_data, test_data


def detect_outliers(values: np.ndarray, 
                   method: str = 'iqr',
                   threshold: float = 1.5) -> np.ndarray:
    """
    检测异常值
    
    Args:
        values: 数值数组
        method: 检测方法 ('iqr' 或 'zscore')
        threshold: 阈值
        
    Returns:
        np.ndarray: 布尔掩码，True表示正常值
    """
    if method == 'iqr':
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (values >= lower_bound) & (values <= upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((values - np.mean(values)) / np.std(values))
        return z_scores < threshold
    
    else:
        return np.ones(len(values), dtype=bool)


def smooth_series(values: np.ndarray, 
                 window: int = 3,
                 method: str = 'moving_average') -> np.ndarray:
    """
    平滑时间序列
    
    Args:
        values: 数值数组
        window: 窗口大小
        method: 平滑方法
        
    Returns:
        np.ndarray: 平滑后的数值
    """
    if method == 'moving_average':
        return pd.Series(values).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    else:
        return values


@st.cache_data
def load_sample_datasets() -> Dict[str, pd.DataFrame]:
    """
    加载示例数据集
    
    Returns:
        Dict[str, pd.DataFrame]: 示例数据集字典
    """
    datasets = {}
    
    # 销售数据示例
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    trend = np.linspace(1000, 2000, len(dates))
    seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 50, len(dates))
    sales = trend + seasonal + noise
    
    datasets['销售数据'] = pd.DataFrame({
        'date': dates,
        'value': sales
    })
    
    # 股价数据示例
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    price = 100
    prices = [price]
    
    for i in range(1, len(dates)):
        change = np.random.normal(0, 2)
        price = max(10, price + change)  # 确保价格不为负
        prices.append(price)
    
    datasets['股价数据'] = pd.DataFrame({
        'date': dates,
        'value': prices
    })
    
    return datasets
