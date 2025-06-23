"""
可视化图表模块

使用Plotly创建交互式时间序列图表
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime


class TimeSeriesVisualizer:
    """时间序列可视化器"""
    
    def __init__(self):
        self.color_palette = {
            'historical': '#1f77b4',
            'prediction': '#ff7f0e', 
            'confidence': 'rgba(255, 127, 14, 0.2)',
            'trend': '#2ca02c',
            'seasonal': '#d62728'
        }
    
    def plot_time_series_with_predictions(self, 
                                        historical_data: pd.DataFrame,
                                        prediction_results: Dict[str, Any],
                                        title: str = "时间序列预测") -> go.Figure:
        """
        绘制历史数据和预测结果
        
        Args:
            historical_data: 历史数据，包含'date'和'value'列
            prediction_results: 预测结果字典
            title: 图表标题
            
        Returns:
            go.Figure: Plotly图表对象
        """
        fig = go.Figure()
        
        # 绘制历史数据
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['value'],
            mode='lines+markers',
            name='历史数据',
            line=dict(color=self.color_palette['historical'], width=2),
            marker=dict(size=4)
        ))
        
        # 绘制预测数据
        if prediction_results:
            pred_dates = prediction_results['dates']
            predictions = prediction_results['predictions']
            lower_ci = prediction_results['lower_ci']
            upper_ci = prediction_results['upper_ci']
            model_name = prediction_results.get('model_name', '未知模型')
            
            # 预测线
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=predictions,
                mode='lines+markers',
                name=f'预测值 ({model_name})',
                line=dict(color=self.color_palette['prediction'], width=2, dash='dash'),
                marker=dict(size=4)
            ))
            
            # 置信区间
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=upper_ci,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=lower_ci,
                mode='lines',
                line=dict(width=0),
                name='置信区间',
                fill='tonexty',
                fillcolor=self.color_palette['confidence'],
                hoverinfo='skip'
            ))
            
            # 连接线（历史数据最后一点到预测第一点）
            if len(historical_data) > 0 and len(predictions) > 0:
                last_historical_date = historical_data['date'].iloc[-1]
                last_historical_value = historical_data['value'].iloc[-1]
                first_pred_date = pred_dates[0]
                first_pred_value = predictions[0]
                
                fig.add_trace(go.Scatter(
                    x=[last_historical_date, first_pred_date],
                    y=[last_historical_value, first_pred_value],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title="日期",
            yaxis_title="数值",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_data_overview(self, data: pd.DataFrame) -> go.Figure:
        """
        绘制数据概览图
        
        Args:
            data: 时间序列数据
            
        Returns:
            go.Figure: Plotly图表对象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('时间序列', '数值分布', '趋势分析', '统计信息'),
            specs=[[{"colspan": 2}, None],
                   [{"type": "histogram"}, {"type": "table"}]]
        )
        
        # 时间序列图
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['value'],
                mode='lines',
                name='原始数据',
                line=dict(color=self.color_palette['historical'])
            ),
            row=1, col=1
        )
        
        # 数值分布直方图
        fig.add_trace(
            go.Histogram(
                x=data['value'],
                name='数值分布',
                marker_color=self.color_palette['historical'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 统计信息表格（优化文字颜色）
        stats = data['value'].describe()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['统计量', '数值'],
                    fill_color='#1f77b4',  # 深蓝色背景
                    font_color='white'     # 白色文字
                ),
                cells=dict(
                    values=[
                        ['数据点数', '均值', '标准差', '最小值', '最大值'],
                        [len(data), f"{stats['mean']:.2f}", f"{stats['std']:.2f}",
                         f"{stats['min']:.2f}", f"{stats['max']:.2f}"]
                    ],
                    fill_color='#f0f0f0',  # 浅灰色背景
                    font_color='black'      # 黑色文字
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="数据概览",
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_model_comparison(self, 
                            historical_data: pd.DataFrame,
                            model_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        绘制多个模型的预测结果比较
        
        Args:
            historical_data: 历史数据
            model_results: 多个模型的预测结果
            
        Returns:
            go.Figure: Plotly图表对象
        """
        fig = go.Figure()
        
        # 绘制历史数据
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['value'],
            mode='lines+markers',
            name='历史数据',
            line=dict(color=self.color_palette['historical'], width=2),
            marker=dict(size=4)
        ))
        
        # 为每个模型绘制预测结果
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, (model_name, results) in enumerate(model_results.items()):
            color = colors[i % len(colors)]
            
            pred_dates = results['dates']
            predictions = results['predictions']
            
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=predictions,
                mode='lines+markers',
                name=f'{model_name}',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="模型预测结果比较",
            xaxis_title="日期",
            yaxis_title="数值",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_residuals(self, 
                      actual: np.ndarray, 
                      predicted: np.ndarray,
                      dates: Optional[pd.Series] = None) -> go.Figure:
        """
        绘制残差图
        
        Args:
            actual: 实际值
            predicted: 预测值
            dates: 日期序列
            
        Returns:
            go.Figure: Plotly图表对象
        """
        residuals = actual - predicted
        
        if dates is None:
            dates = range(len(residuals))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('残差时间序列', '残差分布', '实际vs预测', 'Q-Q图'),
            specs=[[{"colspan": 2}, None],
                   [{}, {}]]
        )
        
        # 残差时间序列
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=residuals,
                mode='lines+markers',
                name='残差',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # 残差分布
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='残差分布',
                marker_color='red',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # 实际vs预测散点图
        fig.add_trace(
            go.Scatter(
                x=actual,
                y=predicted,
                mode='markers',
                name='实际vs预测',
                marker=dict(color='blue')
            ),
            row=2, col=2
        )
        
        # 添加对角线
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='理想线',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="模型诊断图",
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_metric_cards(self, metrics: Dict[str, float]) -> None:
        """
        创建评估指标卡片
        
        Args:
            metrics: 评估指标字典
        """
        if not metrics:
            return
            
        cols = st.columns(len(metrics))
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            with cols[i]:
                if metric_name == 'MAPE':
                    st.metric(
                        label=metric_name,
                        value=f"{value:.2f}%"
                    )
                else:
                    st.metric(
                        label=metric_name,
                        value=f"{value:.4f}"
                    )
