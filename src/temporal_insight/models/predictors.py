"""
具体预测模型实现

包含ARIMA、线性回归、随机森林等多种预测模型
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import streamlit as st

from .base import BasePredictor


class ARIMAPredictor(BasePredictor):
    """ARIMA预测模型"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), auto_order: bool = True):
        super().__init__("ARIMA")
        self.order = order
        self.auto_order = auto_order
        
    def fit(self, data: pd.DataFrame) -> bool:
        """训练ARIMA模型"""
        if not self.validate_data(data):
            return False
            
        try:
            self.train_data = data.copy()
            values = data['value'].values
            
            # 自动选择最优参数
            if self.auto_order:
                self.order = self._find_best_order(values)
                
            # 训练模型
            self.model = ARIMA(values, order=self.order)
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            return True
            
        except Exception as e:
            st.error(f"ARIMA模型训练失败：{str(e)}")
            return False
    
    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成ARIMA预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")

        try:
            # 生成预测
            forecast = self.fitted_model.forecast(steps=steps)
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            # 兼容numpy数组和DataFrame两种返回类型
            if hasattr(forecast_result, 'conf_int'):
                conf_int = forecast_result.conf_int()
                if hasattr(conf_int, 'iloc'):
                    lower_ci = conf_int.iloc[:, 0].values
                    upper_ci = conf_int.iloc[:, 1].values
                else:
                    lower_ci = conf_int[:, 0]
                    upper_ci = conf_int[:, 1]
            else:
                # 如果没有置信区间，使用预测值±10%作为占位
                std = np.std(self.train_data['value'].values) * 0.1
                lower_ci = forecast - 1.96 * std
                upper_ci = forecast + 1.96 * std

            predictions = forecast.values if hasattr(forecast, 'values') else forecast

            return predictions, lower_ci, upper_ci

        except Exception as e:
            # 在测试环境中不显示streamlit错误
            if 'st' in globals():
                st.error(f"ARIMA预测失败：{str(e)}")
            print(f"ARIMA预测失败：{str(e)}")
            return np.array([]), np.array([]), np.array([])
    
    def _find_best_order(self, values: np.ndarray) -> Tuple[int, int, int]:
        """自动选择最优ARIMA参数"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        # 简化的网格搜索
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(values, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
                        
        return best_order
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取ARIMA参数"""
        return {
            'order': self.order,
            'auto_order': self.auto_order
        }
    
    def set_parameters(self, **params) -> None:
        """设置ARIMA参数"""
        if 'order' in params:
            self.order = params['order']
        if 'auto_order' in params:
            self.auto_order = params['auto_order']


class LinearRegressionPredictor(BasePredictor):
    """线性回归预测模型"""
    
    def __init__(self, window_size: int = 7):
        super().__init__("线性回归")
        self.window_size = window_size
        self.scaler = StandardScaler()
        
    def fit(self, data: pd.DataFrame) -> bool:
        """训练线性回归模型"""
        if not self.validate_data(data):
            return False
            
        try:
            self.train_data = data.copy()
            values = data['value'].values
            
            # 创建特征和目标
            X, y = self._create_features(values)
            
            if len(X) == 0:
                return False
                
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model = LinearRegression()
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            return True
            
        except Exception as e:
            st.error(f"线性回归模型训练失败：{str(e)}")
            return False
    
    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成线性回归预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
            
        try:
            values = self.train_data['value'].values
            predictions = []
            
            # 逐步预测
            current_values = values.copy()
            
            for _ in range(steps):
                # 使用最后window_size个值作为特征
                if len(current_values) >= self.window_size:
                    features = current_values[-self.window_size:].reshape(1, -1)
                    features_scaled = self.scaler.transform(features)
                    pred = self.model.predict(features_scaled)[0]
                    predictions.append(pred)
                    current_values = np.append(current_values, pred)
                else:
                    # 如果数据不足，使用简单的趋势预测
                    if len(predictions) > 0:
                        predictions.append(predictions[-1])
                    else:
                        predictions.append(values[-1])
            
            predictions = np.array(predictions)
            
            # 计算置信区间（基于训练误差）
            train_X, train_y = self._create_features(values)
            train_X_scaled = self.scaler.transform(train_X)
            train_pred = self.model.predict(train_X_scaled)
            mse = mean_squared_error(train_y, train_pred)
            std_error = np.sqrt(mse)
            
            # 简单的置信区间估计
            confidence_interval = 1.96 * std_error
            lower_ci = predictions - confidence_interval
            upper_ci = predictions + confidence_interval
            
            return predictions, lower_ci, upper_ci
            
        except Exception as e:
            st.error(f"线性回归预测失败：{str(e)}")
            return np.array([]), np.array([]), np.array([])
    
    def _create_features(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建滑动窗口特征"""
        if len(values) <= self.window_size:
            return np.array([]), np.array([])
            
        X, y = [], []
        for i in range(self.window_size, len(values)):
            X.append(values[i-self.window_size:i])
            y.append(values[i])
            
        return np.array(X), np.array(y)
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取线性回归参数"""
        return {
            'window_size': self.window_size
        }
    
    def set_parameters(self, **params) -> None:
        """设置线性回归参数"""
        if 'window_size' in params:
            self.window_size = max(1, params['window_size'])


class RandomForestPredictor(BasePredictor):
    """随机森林预测模型"""

    def __init__(self, window_size: int = 7, n_estimators: int = 100, random_state: int = 42):
        super().__init__("随机森林")
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()

    def fit(self, data: pd.DataFrame) -> bool:
        """训练随机森林模型"""
        if not self.validate_data(data):
            return False

        try:
            self.train_data = data.copy()
            values = data['value'].values

            # 创建特征和目标
            X, y = self._create_features(values)

            if len(X) == 0:
                return False

            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)

            # 训练模型
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_scaled, y)
            self.is_fitted = True

            return True

        except Exception as e:
            st.error(f"随机森林模型训练失败：{str(e)}")
            return False

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成随机森林预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")

        try:
            values = self.train_data['value'].values
            predictions = []

            # 逐步预测
            current_values = values.copy()

            for _ in range(steps):
                # 使用最后window_size个值作为特征
                if len(current_values) >= self.window_size:
                    features = current_values[-self.window_size:].reshape(1, -1)
                    features_scaled = self.scaler.transform(features)

                    # 获取所有树的预测
                    tree_predictions = []
                    for tree in self.model.estimators_:
                        tree_pred = tree.predict(features_scaled)[0]
                        tree_predictions.append(tree_pred)

                    # 计算均值和标准差
                    pred_mean = np.mean(tree_predictions)
                    pred_std = np.std(tree_predictions)

                    predictions.append((pred_mean, pred_std))
                    current_values = np.append(current_values, pred_mean)
                else:
                    # 如果数据不足，使用简单的趋势预测
                    if len(predictions) > 0:
                        last_pred, last_std = predictions[-1]
                        predictions.append((last_pred, last_std))
                    else:
                        predictions.append((values[-1], 0))

            # 分离预测值和不确定性
            pred_values = np.array([p[0] for p in predictions])
            pred_stds = np.array([p[1] for p in predictions])

            # 计算置信区间
            confidence_interval = 1.96 * pred_stds
            lower_ci = pred_values - confidence_interval
            upper_ci = pred_values + confidence_interval

            return pred_values, lower_ci, upper_ci

        except Exception as e:
            st.error(f"随机森林预测失败：{str(e)}")
            return np.array([]), np.array([]), np.array([])

    def _create_features(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建滑动窗口特征"""
        if len(values) <= self.window_size:
            return np.array([]), np.array([])

        X, y = [], []
        for i in range(self.window_size, len(values)):
            X.append(values[i-self.window_size:i])
            y.append(values[i])

        return np.array(X), np.array(y)

    def get_parameters(self) -> Dict[str, Any]:
        """获取随机森林参数"""
        return {
            'window_size': self.window_size,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state
        }

    def set_parameters(self, **params) -> None:
        """设置随机森林参数"""
        if 'window_size' in params:
            self.window_size = max(1, params['window_size'])
        if 'n_estimators' in params:
            self.n_estimators = max(1, params['n_estimators'])
        if 'random_state' in params:
            self.random_state = params['random_state']


class SimpleMovingAveragePredictor(BasePredictor):
    """简单移动平均预测模型"""

    def __init__(self, window_size: int = 7):
        super().__init__("移动平均")
        self.window_size = window_size

    def fit(self, data: pd.DataFrame) -> bool:
        """训练移动平均模型"""
        if not self.validate_data(data):
            return False

        try:
            self.train_data = data.copy()
            self.is_fitted = True
            return True

        except Exception as e:
            st.error(f"移动平均模型训练失败：{str(e)}")
            return False

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成移动平均预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")

        try:
            values = self.train_data['value'].values

            # 计算最后window_size个值的平均值
            if len(values) >= self.window_size:
                last_avg = np.mean(values[-self.window_size:])
            else:
                last_avg = np.mean(values)

            # 简单预测：所有未来值都等于最后的移动平均值
            predictions = np.full(steps, last_avg)

            # 计算历史移动平均的标准差作为不确定性估计
            if len(values) >= self.window_size:
                moving_avgs = []
                for i in range(self.window_size, len(values)):
                    moving_avgs.append(np.mean(values[i-self.window_size:i]))

                if len(moving_avgs) > 1:
                    std_error = np.std(moving_avgs)
                else:
                    std_error = np.std(values) * 0.1
            else:
                std_error = np.std(values) * 0.1

            # 计算置信区间
            confidence_interval = 1.96 * std_error
            lower_ci = predictions - confidence_interval
            upper_ci = predictions + confidence_interval

            return predictions, lower_ci, upper_ci

        except Exception as e:
            st.error(f"移动平均预测失败：{str(e)}")
            return np.array([]), np.array([]), np.array([])

    def get_parameters(self) -> Dict[str, Any]:
        """获取移动平均参数"""
        return {
            'window_size': self.window_size
        }

    def set_parameters(self, **params) -> None:
        """设置移动平均参数"""
        if 'window_size' in params:
            self.window_size = max(1, params['window_size'])
