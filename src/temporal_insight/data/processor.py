"""
数据处理模块

提供时间序列数据的上传、验证、预处理功能
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import streamlit as st
from datetime import datetime
import io


class TimeSeriesDataProcessor:
    """时间序列数据处理器"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.date_column = None
        self.value_column = None
        
    def load_data_from_upload(self, uploaded_file) -> bool:
        """
        从上传的文件加载数据
        
        Args:
            uploaded_file: Streamlit上传的文件对象
            
        Returns:
            bool: 是否成功加载数据
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(uploaded_file)
            else:
                st.error("不支持的文件格式。请上传CSV或Excel文件。")
                return False
                
            st.success(f"成功加载数据，共 {len(self.data)} 行，{len(self.data.columns)} 列")
            return True
            
        except Exception as e:
            st.error(f"加载数据时出错：{str(e)}")
            return False
    
    def validate_data(self, date_col: str, value_col: str) -> bool:
        """
        验证数据格式和完整性
        
        Args:
            date_col: 日期列名
            value_col: 数值列名
            
        Returns:
            bool: 数据是否有效
        """
        if self.data is None:
            st.error("请先上传数据")
            return False
            
        # 检查列是否存在
        if date_col not in self.data.columns:
            st.error(f"未找到日期列：{date_col}")
            return False
            
        if value_col not in self.data.columns:
            st.error(f"未找到数值列：{value_col}")
            return False
            
        self.date_column = date_col
        self.value_column = value_col
        
        # 检查数据类型和缺失值
        try:
            # 尝试转换日期列
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            
            # 检查数值列
            self.data[value_col] = pd.to_numeric(self.data[value_col], errors='coerce')
            
            # 检查缺失值
            missing_dates = self.data[date_col].isna().sum()
            missing_values = self.data[value_col].isna().sum()
            
            if missing_dates > 0:
                st.warning(f"日期列有 {missing_dates} 个缺失值")
                
            if missing_values > 0:
                st.warning(f"数值列有 {missing_values} 个缺失值")
                
            # 检查数据量
            if len(self.data) < 10:
                st.warning("数据量较少（少于10个数据点），可能影响预测效果")
                
            return True
            
        except Exception as e:
            st.error(f"数据验证失败：{str(e)}")
            return False
    
    def preprocess_data(self, 
                       fill_missing: str = 'interpolate',
                       remove_outliers: bool = False,
                       outlier_method: str = 'iqr') -> bool:
        """
        预处理数据
        
        Args:
            fill_missing: 缺失值处理方法 ('drop', 'forward', 'backward', 'interpolate', 'mean')
            remove_outliers: 是否移除异常值
            outlier_method: 异常值检测方法 ('iqr', 'zscore')
            
        Returns:
            bool: 预处理是否成功
        """
        if self.data is None or self.date_column is None or self.value_column is None:
            st.error("请先加载并验证数据")
            return False
            
        try:
            # 创建处理后的数据副本
            processed = self.data[[self.date_column, self.value_column]].copy()
            processed.columns = ['date', 'value']
            
            # 按日期排序
            processed = processed.sort_values('date').reset_index(drop=True)
            
            # 处理缺失值
            initial_count = len(processed)
            if fill_missing == 'drop':
                processed = processed.dropna()
            elif fill_missing == 'forward':
                processed['value'] = processed['value'].fillna(method='ffill')
            elif fill_missing == 'backward':
                processed['value'] = processed['value'].fillna(method='bfill')
            elif fill_missing == 'interpolate':
                processed['value'] = processed['value'].interpolate()
            elif fill_missing == 'mean':
                mean_value = processed['value'].mean()
                processed['value'] = processed['value'].fillna(mean_value)
                
            # 移除仍然存在的缺失值
            processed = processed.dropna()
            
            if len(processed) < initial_count:
                st.info(f"处理缺失值后，数据从 {initial_count} 行减少到 {len(processed)} 行")
            
            # 处理异常值
            if remove_outliers and len(processed) > 0:
                if outlier_method == 'iqr':
                    Q1 = processed['value'].quantile(0.25)
                    Q3 = processed['value'].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (processed['value'] >= lower_bound) & (processed['value'] <= upper_bound)
                elif outlier_method == 'zscore':
                    z_scores = np.abs((processed['value'] - processed['value'].mean()) / processed['value'].std())
                    outlier_mask = z_scores < 3
                    
                outliers_removed = len(processed) - outlier_mask.sum()
                if outliers_removed > 0:
                    processed = processed[outlier_mask].reset_index(drop=True)
                    st.info(f"移除了 {outliers_removed} 个异常值")
            
            # 检查最终数据量
            if len(processed) < 5:
                st.error("预处理后数据量过少，无法进行有效预测")
                return False
                
            self.processed_data = processed
            st.success(f"数据预处理完成，最终数据量：{len(processed)} 行")
            return True
            
        except Exception as e:
            st.error(f"数据预处理失败：{str(e)}")
            return False
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据基本信息"""
        if self.processed_data is None:
            return {}
            
        data = self.processed_data
        return {
            'total_points': len(data),
            'date_range': {
                'start': data['date'].min(),
                'end': data['date'].max()
            },
            'value_stats': {
                'min': data['value'].min(),
                'max': data['value'].max(),
                'mean': data['value'].mean(),
                'std': data['value'].std()
            },
            'frequency': self._detect_frequency()
        }
    
    def _detect_frequency(self) -> str:
        """检测时间序列频率"""
        if self.processed_data is None or len(self.processed_data) < 2:
            return "未知"
            
        # 计算时间间隔
        time_diffs = self.processed_data['date'].diff().dropna()
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
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """获取预处理后的数据"""
        return self.processed_data
    
    def create_sample_data(self) -> pd.DataFrame:
        """创建示例数据"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        # 创建带趋势和季节性的示例数据
        trend = np.linspace(100, 200, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        noise = np.random.normal(0, 5, len(dates))
        values = trend + seasonal + noise
        
        sample_data = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        return sample_data
