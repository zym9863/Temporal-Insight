"""
预测模型基类

定义所有预测模型的通用接口
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


class BasePredictor(ABC):
    """预测模型基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.train_data = None
        
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> bool:
        """
        训练模型
        
        Args:
            data: 训练数据，包含'date'和'value'列
            
        Returns:
            bool: 训练是否成功
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成预测
        
        Args:
            steps: 预测步数
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (预测值, 下置信区间, 上置信区间)
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        pass
    
    @abstractmethod
    def set_parameters(self, **params) -> None:
        """设置模型参数"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'parameters': self.get_parameters() if self.is_fitted else None
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据格式"""
        if not isinstance(data, pd.DataFrame):
            return False
            
        required_columns = ['date', 'value']
        if not all(col in data.columns for col in required_columns):
            return False
            
        if len(data) < 2:
            return False
            
        return True
