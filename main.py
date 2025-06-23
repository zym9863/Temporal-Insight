"""
时序洞察 (Temporal Insight) - 主应用入口

智能时间序列预测系统的Streamlit Web界面
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from temporal_insight.data.processor import TimeSeriesDataProcessor
from temporal_insight.models.engine import PredictionEngine
from temporal_insight.visualization.charts import TimeSeriesVisualizer


def main():
    """主应用函数"""

    # 页面配置
    st.set_page_config(
        page_title="时序洞察 - Temporal Insight",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 应用标题
    st.title("📈 时序洞察 (Temporal Insight)")
    st.markdown("### 智能时间序列预测系统")
    st.markdown("---")

    # 初始化组件
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = TimeSeriesDataProcessor()
    if 'prediction_engine' not in st.session_state:
        st.session_state.prediction_engine = PredictionEngine()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = TimeSeriesVisualizer()

    # 侧边栏
    with st.sidebar:
        st.header("🔧 控制面板")

        # 步骤指示器
        steps = ["📁 数据上传", "⚙️ 数据处理", "🤖 模型选择", "📊 预测分析"]
        current_step = st.selectbox("选择步骤", steps, index=0)

    # 主内容区域
    if current_step == "📁 数据上传":
        data_upload_section()
    elif current_step == "⚙️ 数据处理":
        data_processing_section()
    elif current_step == "🤖 模型选择":
        model_selection_section()
    elif current_step == "📊 预测分析":
        prediction_analysis_section()


def data_upload_section():
    """数据上传部分"""
    st.header("📁 数据上传")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("上传时间序列数据")

        # 文件上传
        uploaded_file = st.file_uploader(
            "选择CSV或Excel文件",
            type=['csv', 'xlsx', 'xls'],
            help="文件应包含日期列和数值列"
        )

        if uploaded_file is not None:
            if st.session_state.data_processor.load_data_from_upload(uploaded_file):
                st.success("✅ 数据加载成功！")

                # 显示数据预览
                st.subheader("数据预览")
                data = st.session_state.data_processor.data
                st.dataframe(data.head(10), use_container_width=True)

                # 数据基本信息
                st.subheader("数据信息")
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("数据行数", len(data))
                    st.metric("数据列数", len(data.columns))
                with col_info2:
                    st.metric("缺失值", data.isnull().sum().sum())
                    st.metric("数据类型", len(data.dtypes.unique()))

    with col2:
        st.subheader("📋 使用说明")
        st.markdown("""
        **数据格式要求：**
        - 支持CSV和Excel格式
        - 必须包含日期列和数值列
        - 日期格式：YYYY-MM-DD 或其他标准格式
        - 数值列：数字类型

        **示例数据结构：**
        ```
        日期        | 销售额
        2023-01-01 | 1000
        2023-01-02 | 1200
        2023-01-03 | 980
        ...
        ```
        """)

        # 示例数据下载
        if st.button("📥 下载示例数据"):
            sample_data = st.session_state.data_processor.create_sample_data()
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="下载示例CSV文件",
                data=csv,
                file_name="sample_time_series.csv",
                mime="text/csv"
            )


def data_processing_section():
    """数据处理部分"""
    st.header("⚙️ 数据处理")

    if st.session_state.data_processor.data is None:
        st.warning("⚠️ 请先上传数据")
        return

    data = st.session_state.data_processor.data

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("列选择")

        # 选择日期列和数值列
        date_column = st.selectbox(
            "选择日期列",
            data.columns.tolist(),
            help="包含时间信息的列"
        )

        value_column = st.selectbox(
            "选择数值列",
            data.columns.tolist(),
            index=1 if len(data.columns) > 1 else 0,
            help="要预测的数值列"
        )

        # 验证数据
        if st.button("🔍 验证数据"):
            if st.session_state.data_processor.validate_data(date_column, value_column):
                st.success("✅ 数据验证通过！")

                # 显示数据概览图
                processed_data = data[[date_column, value_column]].copy()
                processed_data.columns = ['date', 'value']
                processed_data['date'] = pd.to_datetime(processed_data['date'])
                processed_data = processed_data.dropna().sort_values('date')

                fig = st.session_state.visualizer.plot_data_overview(processed_data)
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("预处理选项")

        # 缺失值处理
        fill_missing = st.selectbox(
            "缺失值处理",
            ['interpolate', 'forward', 'backward', 'mean', 'drop'],
            help="选择处理缺失值的方法"
        )

        # 异常值处理
        remove_outliers = st.checkbox("移除异常值", help="自动检测并移除异常值")

        if remove_outliers:
            outlier_method = st.selectbox(
                "异常值检测方法",
                ['iqr', 'zscore'],
                help="IQR: 四分位距方法, Z-score: 标准分数方法"
            )
        else:
            outlier_method = 'iqr'

        # 执行预处理
        if st.button("🔄 执行预处理"):
            if st.session_state.data_processor.validate_data(date_column, value_column):
                if st.session_state.data_processor.preprocess_data(
                    fill_missing=fill_missing,
                    remove_outliers=remove_outliers,
                    outlier_method=outlier_method
                ):
                    st.success("✅ 数据预处理完成！")

                    # 显示处理后的数据信息
                    info = st.session_state.data_processor.get_data_info()
                    if info:
                        st.metric("处理后数据量", info['total_points'])
                        st.metric("时间频率", info['frequency'])


def model_selection_section():
    """模型选择部分"""
    st.header("🤖 模型选择与配置")

    if st.session_state.data_processor.processed_data is None:
        st.warning("⚠️ 请先完成数据处理")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("选择预测模型")

        # 模型选择
        available_models = st.session_state.prediction_engine.get_available_models()
        selected_model = st.selectbox(
            "预测模型",
            available_models,
            help="选择用于预测的机器学习模型"
        )

        # 模型参数配置
        st.subheader("模型参数配置")
        model_params = {}

        param_config = st.session_state.prediction_engine.get_model_parameters_config(selected_model)

        for param_name, config in param_config.items():
            if config['type'] == 'number':
                model_params[param_name] = st.number_input(
                    config['label'],
                    min_value=config.get('min_value', 1),
                    max_value=config.get('max_value', 100),
                    value=config['default'],
                    help=config.get('help', '')
                )
            elif config['type'] == 'checkbox':
                model_params[param_name] = st.checkbox(
                    config['label'],
                    value=config['default'],
                    help=config.get('help', '')
                )

        # 处理ARIMA特殊参数
        # 不再需要手动组合order参数，因为ARIMAPredictor现在直接接受auto_order参数

        # 训练模型
        if st.button("🚀 训练模型"):
            with st.spinner("正在训练模型..."):
                # 选择模型
                if st.session_state.prediction_engine.select_model(selected_model, **model_params):
                    # 训练模型
                    if st.session_state.prediction_engine.train_model(
                        st.session_state.data_processor.processed_data
                    ):
                        st.success("✅ 模型训练完成！")

                        # 显示模型信息
                        model_info = st.session_state.prediction_engine.get_model_info()
                        st.json(model_info)

                        # 模型评估
                        metrics = st.session_state.prediction_engine.evaluate_model()
                        if metrics:
                            st.subheader("模型评估指标")
                            st.session_state.visualizer.create_metric_cards(metrics)

    with col2:
        st.subheader("📚 模型说明")

        model_descriptions = {
            'ARIMA': """
            **ARIMA模型**
            - 适用于有趋势和季节性的时间序列
            - 自动选择最优参数
            - 提供统计学置信区间
            - 适合中短期预测
            """,
            '线性回归': """
            **线性回归模型**
            - 基于滑动窗口的特征工程
            - 计算简单，速度快
            - 适合线性趋势数据
            - 可解释性强
            """,
            '随机森林': """
            **随机森林模型**
            - 集成学习方法
            - 能捕捉非线性关系
            - 提供预测不确定性估计
            - 对异常值鲁棒
            """,
            '移动平均': """
            **移动平均模型**
            - 最简单的预测方法
            - 适合平稳时间序列
            - 计算速度极快
            - 基准模型
            """
        }

        if selected_model in model_descriptions:
            st.markdown(model_descriptions[selected_model])


def prediction_analysis_section():
    """预测分析部分"""
    st.header("📊 预测分析")

    if not st.session_state.prediction_engine.current_model or \
       not st.session_state.prediction_engine.current_model.is_fitted:
        st.warning("⚠️ 请先训练模型")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("生成预测")

        # 预测参数
        prediction_steps = st.number_input(
            "预测步数",
            min_value=1,
            max_value=365,
            value=30,
            help="要预测的未来时间点数量"
        )

        frequency = st.selectbox(
            "时间频率",
            ['D', 'W', 'M'],
            format_func=lambda x: {'D': '日', 'W': '周', 'M': '月'}[x],
            help="预测的时间间隔"
        )

        # 生成预测
        if st.button("🔮 生成预测"):
            with st.spinner("正在生成预测..."):
                if st.session_state.prediction_engine.generate_predictions(
                    steps=prediction_steps,
                    frequency=frequency
                ):
                    st.success("✅ 预测生成完成！")

        # 显示预测结果
        prediction_results = st.session_state.prediction_engine.get_prediction_results()
        if prediction_results:
            st.subheader("预测结果可视化")

            # 绘制预测图表
            historical_data = st.session_state.data_processor.processed_data
            fig = st.session_state.visualizer.plot_time_series_with_predictions(
                historical_data=historical_data,
                prediction_results=prediction_results,
                title=f"{st.session_state.prediction_engine.model_name}模型预测结果"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 预测数据表格
            st.subheader("预测数据")
            pred_df = pd.DataFrame({
                '日期': prediction_results['dates'],
                '预测值': prediction_results['predictions'],
                '下置信区间': prediction_results['lower_ci'],
                '上置信区间': prediction_results['upper_ci']
            })
            st.dataframe(pred_df, use_container_width=True)

            # 下载预测结果
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 下载预测结果",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

    with col2:
        st.subheader("📈 预测摘要")

        if prediction_results:
            predictions = prediction_results['predictions']

            st.metric("预测步数", len(predictions))
            st.metric("预测均值", f"{np.mean(predictions):.2f}")
            st.metric("预测范围", f"{np.max(predictions) - np.min(predictions):.2f}")

            # 趋势分析
            if len(predictions) > 1:
                trend = "上升" if predictions[-1] > predictions[0] else "下降"
                st.metric("总体趋势", trend)


if __name__ == "__main__":
    main()
