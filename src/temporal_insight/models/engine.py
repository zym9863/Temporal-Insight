"""
预测引擎

统一管理所有预测模型，提供模型选择、训练、预测等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import streamlit as st
from datetime import datetime, timedelta

from .predictors import (
    ARIMAPredictor,
    LinearRegressionPredictor, 
    RandomForestPredictor,
    SimpleMovingAveragePredictor
)
from .base import BasePredictor


class PredictionEngine:
    """预测引擎"""
    
    def __init__(self):
        self.available_models = {
            'ARIMA': ARIMAPredictor,
            '线性回归': LinearRegressionPredictor,
            '随机森林': RandomForestPredictor,
            '移动平均': SimpleMovingAveragePredictor
        }
        self.current_model = None
        self.model_name = None
        self.training_data = None
        self.prediction_results = None
        
    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return list(self.available_models.keys())
    
    def select_model(self, model_name: str, **params) -> bool:
        """
        选择并初始化模型
        
        Args:
            model_name: 模型名称
            **params: 模型参数
            
        Returns:
            bool: 是否成功选择模型
        """
        if model_name not in self.available_models:
            st.error(f"不支持的模型：{model_name}")
            return False
            
        try:
            model_class = self.available_models[model_name]
            
            # 对ARIMA模型特殊处理：将p,d,q参数组合成order元组
            if model_name == 'ARIMA':
                # 如果参数中包含p,d,q，则组合成order元组
                if 'p' in params and 'd' in params and 'q' in params:
                    params['order'] = (params['p'], params['d'], params['q'])
                    # 删除单独的p,d,q参数
                    del params['p']
                    del params['d']
                    del params['q']
            
            self.current_model = model_class(**params)
            self.model_name = model_name
            return True
            
        except Exception as e:
            st.error(f"模型初始化失败：{str(e)}")
            return False
    
    def train_model(self, data: pd.DataFrame) -> bool:
        """
        训练当前选择的模型
        
        Args:
            data: 训练数据
            
        Returns:
            bool: 训练是否成功
        """
        if self.current_model is None:
            st.error("请先选择模型")
            return False
            
        try:
            success = self.current_model.fit(data)
            if success:
                self.training_data = data.copy()
                st.success(f"{self.model_name}模型训练完成")
            return success
            
        except Exception as e:
            st.error(f"模型训练失败：{str(e)}")
            return False
    
    def generate_predictions(self, steps: int, frequency: str = 'D') -> bool:
        """
        生成预测结果
        
        Args:
            steps: 预测步数
            frequency: 时间频率 ('D'=日, 'W'=周, 'M'=月)
            
        Returns:
            bool: 预测是否成功
        """
        if self.current_model is None or not self.current_model.is_fitted:
            st.error("请先训练模型")
            return False
            
        try:
            # 生成预测
            predictions, lower_ci, upper_ci = self.current_model.predict(steps)
            
            if len(predictions) == 0:
                return False
            
            # 生成未来日期
            last_date = self.training_data['date'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq=frequency
            )
            
            # 组织预测结果
            self.prediction_results = {
                'dates': future_dates,
                'predictions': predictions,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'model_name': self.model_name,
                'steps': steps
            }
            
            st.success(f"成功生成 {steps} 步预测")
            return True
            
        except Exception as e:
            st.error(f"预测生成失败：{str(e)}")
            return False
    
    def get_prediction_results(self) -> Optional[Dict[str, Any]]:
        """获取预测结果"""
        return self.prediction_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        if self.current_model is None:
            return {}
            
        return self.current_model.get_model_info()
    
    def evaluate_model(self, test_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据，如果为None则使用训练数据的一部分
            
        Returns:
            Dict[str, float]: 评估指标
        """
        if self.current_model is None or not self.current_model.is_fitted:
            return {}
            
        try:
            if test_data is None:
                # 使用训练数据的最后20%作为测试数据
                train_data = self.training_data.copy()
                split_point = int(len(train_data) * 0.8)
                test_data = train_data.iloc[split_point:].reset_index(drop=True)
                train_subset = train_data.iloc[:split_point].reset_index(drop=True)
                
                # 重新训练模型（仅使用80%的数据）
                temp_model = self.available_models[self.model_name]()
                temp_model.set_parameters(**self.current_model.get_parameters())
                temp_model.fit(train_subset)
                
                # 预测测试期间的值
                test_steps = len(test_data)
                predictions, _, _ = temp_model.predict(test_steps)
                actual_values = test_data['value'].values
            else:
                # 使用提供的测试数据
                test_steps = len(test_data)
                predictions, _, _ = self.current_model.predict(test_steps)
                actual_values = test_data['value'].values
            
            # 计算评估指标
            if len(predictions) == len(actual_values):
                mse = np.mean((predictions - actual_values) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - actual_values))
                mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
                
                return {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape
                }
            else:
                return {}
                
        except Exception as e:
            st.error(f"模型评估失败：{str(e)}")
            return {}
    
    def get_model_parameters_config(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型参数配置界面
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict[str, Any]: 参数配置
        """
        configs = {
            'ARIMA': {
                'auto_order': {
                    'type': 'checkbox',
                    'label': '自动选择参数',
                    'default': True,
                    'help': '自动选择最优的ARIMA参数(p,d,q)'
                },
                'p': {
                    'type': 'number',
                    'label': 'AR阶数 (p)',
                    'default': 1,
                    'min_value': 0,
                    'max_value': 5,
                    'help': '自回归项的阶数'
                },
                'd': {
                    'type': 'number',
                    'label': '差分阶数 (d)',
                    'default': 1,
                    'min_value': 0,
                    'max_value': 2,
                    'help': '差分的阶数'
                },
                'q': {
                    'type': 'number',
                    'label': 'MA阶数 (q)',
                    'default': 1,
                    'min_value': 0,
                    'max_value': 5,
                    'help': '移动平均项的阶数'
                }
            },
            '线性回归': {
                'window_size': {
                    'type': 'number',
                    'label': '窗口大小',
                    'default': 7,
                    'min_value': 1,
                    'max_value': 30,
                    'help': '用于预测的历史数据点数量'
                }
            },
            '随机森林': {
                'window_size': {
                    'type': 'number',
                    'label': '窗口大小',
                    'default': 7,
                    'min_value': 1,
                    'max_value': 30,
                    'help': '用于预测的历史数据点数量'
                },
                'n_estimators': {
                    'type': 'number',
                    'label': '树的数量',
                    'default': 100,
                    'min_value': 10,
                    'max_value': 500,
                    'help': '随机森林中树的数量'
                }
            },
            '移动平均': {
                'window_size': {
                    'type': 'number',
                    'label': '窗口大小',
                    'default': 7,
                    'min_value': 1,
                    'max_value': 30,
                    'help': '移动平均的窗口大小'
                }
            }
        }
        
        return configs.get(model_name, {})
