# 房价预测项目

一个基于机器学习的房价预测系统，使用多种算法进行房价预测和分析。

## 项目结构

```
house_price_project/
│
├── data/                                    # 数据文件夹
│   ├── house_price_predictions.csv          # 模型预测结果
│   ├── simple_predictions.csv               # 简单预测结果
│   └── improved_predictions.csv             # 改进版预测结果
│
├── src/                                     # 源代码文件夹
│   ├── __init__.py                          # 标记为Python包
│   ├── house_price_analysis.py              # 数据分析代码
│   ├── improved_analysis.py                 # 改进版分析代码
│   ├── model_comparison.py                  # 模型比较代码
│   └── 1.py                                 # 快速分析脚本
│
├── visualizations/                          # 可视化文件夹
│   ├── data_visualization.png               # 数据可视化图片
│   ├── quick_analysis.png                   # 快速分析图片
│   └── model_comparison.png                 # 模型比较图片
│
├── tests/                                   # 测试文件夹
│   ├── __init__.py                          # 测试包初始化
│   ├── test_analysis.py                     # 测试分析代码
│   └── test_model.py                        # 测试模型代码
│
├── requirements.txt                         # 项目依赖文件
├── README.md                                # 项目说明文件
└── .gitignore                               # Git忽略规则
```

## 功能特性

### 1. 数据分析
- 数据探索和可视化
- 相关性分析
- 缺失值处理
- 特征工程

### 2. 模型训练
- 多种机器学习算法
- 超参数优化
- 交叉验证
- 模型性能评估

### 3. 预测功能
- 房价预测
- 结果导出
- 性能指标计算

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 快速分析
```bash
python src/1.py
```

### 完整分析
```bash
python src/house_price_analysis.py
```

### 改进版分析
```bash
python src/improved_analysis.py
```

### 运行测试
```bash
python -m pytest tests/
```

## 模型性能

- **简单模型**: RMSE ≈ 28,883美元，R² ≈ 0.89
- **改进模型**: RMSE ≈ 15,000-20,000美元，R² ≈ 0.92-0.95

## 技术栈

- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- xgboost, lightgbm

## 数据来源

项目使用Kaggle的房价预测数据集：
- House Prices: Advanced Regression Techniques

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License