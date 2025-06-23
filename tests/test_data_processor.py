"""
数据处理模块测试
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from temporal_insight.data.processor import TimeSeriesDataProcessor


class TestTimeSeriesDataProcessor(unittest.TestCase):
    """时间序列数据处理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.processor = TimeSeriesDataProcessor()
        
        # 创建测试数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = np.random.normal(100, 10, len(dates))
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        # 创建有缺失值的测试数据
        self.test_data_with_missing = self.test_data.copy()
        self.test_data_with_missing.loc[10:15, 'value'] = np.nan
        
    def test_validate_data_success(self):
        """测试数据验证成功情况"""
        self.processor.data = self.test_data
        result = self.processor.validate_data('date', 'value')
        self.assertTrue(result)
        
    def test_validate_data_missing_column(self):
        """测试缺失列的情况"""
        self.processor.data = self.test_data
        result = self.processor.validate_data('nonexistent_date', 'value')
        self.assertFalse(result)
        
    def test_preprocess_data_interpolate(self):
        """测试插值预处理"""
        self.processor.data = self.test_data_with_missing
        self.processor.validate_data('date', 'value')
        
        result = self.processor.preprocess_data(fill_missing='interpolate')
        self.assertTrue(result)
        self.assertIsNotNone(self.processor.processed_data)
        
        # 检查是否没有缺失值
        self.assertEqual(self.processor.processed_data['value'].isnull().sum(), 0)
        
    def test_preprocess_data_drop_missing(self):
        """测试删除缺失值预处理"""
        self.processor.data = self.test_data_with_missing
        self.processor.validate_data('date', 'value')
        
        original_length = len(self.test_data_with_missing)
        result = self.processor.preprocess_data(fill_missing='drop')
        self.assertTrue(result)
        
        # 检查数据长度是否减少
        self.assertLess(len(self.processor.processed_data), original_length)
        
    def test_get_data_info(self):
        """测试获取数据信息"""
        self.processor.data = self.test_data
        self.processor.validate_data('date', 'value')
        self.processor.preprocess_data()
        
        info = self.processor.get_data_info()
        
        self.assertIn('total_points', info)
        self.assertIn('date_range', info)
        self.assertIn('value_stats', info)
        self.assertIn('frequency', info)
        
    def test_detect_frequency(self):
        """测试频率检测"""
        self.processor.processed_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'value': range(10)
        })
        
        frequency = self.processor._detect_frequency()
        self.assertEqual(frequency, '日')
        
    def test_create_sample_data(self):
        """测试创建示例数据"""
        sample_data = self.processor.create_sample_data()
        
        self.assertIsInstance(sample_data, pd.DataFrame)
        self.assertIn('date', sample_data.columns)
        self.assertIn('value', sample_data.columns)
        self.assertGreater(len(sample_data), 0)


if __name__ == '__main__':
    unittest.main()
