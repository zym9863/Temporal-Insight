# 时序洞察 (Temporal Insight)

[中文文档](README.md) | [English Version](README_EN.md)

📈 智能时间序列预测系统

## 项目简介

时序洞察是一个基于Python的智能时间序列分析和预测工具，提供用户友好的Web界面，支持多种预测模型和交互式可视化。

## 核心功能

### 🤖 模型驱动的智能预测
- **多种预测模型**：ARIMA、线性回归、随机森林、移动平均
- **自动参数优化**：智能选择最优模型参数
- **模型评估**：提供MSE、RMSE、MAE、MAPE等评估指标
- **批量预测**：支持多步预测和不同时间频率

### 📊 预测结果与置信区间可视化
- **交互式图表**：基于Plotly的动态可视化
- **置信区间**：直观展示预测不确定性
- **历史对比**：历史数据与预测结果对比展示
- **多模型比较**：同时展示多个模型的预测结果

## 技术栈

- **后端**：Python 3.12+
- **Web框架**：Streamlit
- **数据处理**：Pandas, NumPy
- **机器学习**：Scikit-learn, Statsmodels
- **可视化**：Plotly, Matplotlib, Seaborn
- **包管理**：uv

## 项目结构

```
Temporal-Insight/
├── src/
│   └── temporal_insight/
│       ├── __init__.py
│       ├── data/                 # 数据处理模块
│       │   ├── __init__.py
│       │   └── processor.py
│       ├── models/               # 预测模型模块
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── predictors.py
│       │   └── engine.py
│       ├── visualization/        # 可视化模块
│       │   ├── __init__.py
│       │   └── charts.py
│       └── utils/               # 工具模块
│           ├── __init__.py
│           └── helpers.py
├── tests/                       # 测试文件
├── data/                        # 数据文件
├── main.py                      # 主应用入口
├── pyproject.toml              # 项目配置
└── README.md
```

## 安装和运行

### 环境要求
- Python 3.12+
- uv包管理器

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/zym9863/Temporal-Insight.git
cd Temporal-Insight
```

2. **安装依赖**
```bash
uv sync
```

3. **运行应用**
```bash
uv run streamlit run main.py
```

4. **访问应用**
打开浏览器访问：http://localhost:8501

## 使用指南

### 1. 数据上传
- 支持CSV和Excel格式文件
- 数据应包含日期列和数值列
- 可下载示例数据进行测试

### 2. 数据处理
- 选择日期列和数值列
- 配置缺失值处理方法
- 可选择移除异常值

### 3. 模型选择
- 选择预测模型（ARIMA、线性回归、随机森林、移动平均）
- 配置模型参数
- 训练模型并查看评估指标

### 4. 预测分析
- 设置预测步数和时间频率
- 生成预测结果
- 查看可视化图表和数据表格
- 下载预测结果

## 模型说明

### ARIMA模型
- 适用于有趋势和季节性的时间序列
- 自动选择最优参数(p,d,q)
- 提供统计学置信区间
- 适合中短期预测

### 线性回归模型
- 基于滑动窗口的特征工程
- 计算简单，速度快
- 适合线性趋势数据
- 可解释性强

### 随机森林模型
- 集成学习方法
- 能捕捉非线性关系
- 提供预测不确定性估计
- 对异常值鲁棒

### 移动平均模型
- 最简单的预测方法
- 适合平稳时间序列
- 计算速度极快
- 可作为基准模型

## 开发

### 添加新模型
1. 在`src/temporal_insight/models/predictors.py`中继承`BasePredictor`类
2. 实现`fit`、`predict`、`get_parameters`、`set_parameters`方法
3. 在`engine.py`中注册新模型

### 自定义可视化
1. 在`src/temporal_insight/visualization/charts.py`中添加新的图表方法
2. 在主界面中调用新的可视化功能

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue