#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房价预测快速分析脚本
简化版本，用于快速测试
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def quick_analysis():
    """快速数据分析"""
    print("=== 房价预测快速分析 ===")
    
    # 数据路径
    data_path = "D:/Users/Administrator/Desktop/house-prices-advanced-regression-techniques (1)"
    
    try:
        # 1. 加载数据
        print("1. 加载数据...")
        train_data = pd.read_csv(f"{data_path}/train.csv")
        test_data = pd.read_csv(f"{data_path}/test.csv")
        
        print(f"训练集: {train_data.shape}")
        print(f"测试集: {test_data.shape}")
        
        # 2. 基本信息
        print("\n2. 数据基本信息:")
        print(f"特征数量: {train_data.shape[1] - 2}")  # 减去Id和SalePrice
        print(f"样本数量: {train_data.shape[0]}")
        print(f"目标变量范围: ${train_data['SalePrice'].min():,} - ${train_data['SalePrice'].max():,}")
        
        # 3. 简单可视化
        print("\n3. 创建可视化图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 价格分布
        axes[0, 0].hist(train_data['SalePrice'], bins=50, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('房屋价格分布')
        axes[0, 0].set_xlabel('价格')
        axes[0, 0].set_ylabel('频数')
        
        # 面积与价格关系
        axes[0, 1].scatter(train_data['GrLivArea'], train_data['SalePrice'], alpha=0.5)
        axes[0, 1].set_title('居住面积 vs 价格')
        axes[0, 1].set_xlabel('居住面积')
        axes[0, 1].set_ylabel('价格')
        
        # 质量与价格关系
        quality_price = train_data.groupby('OverallQual')['SalePrice'].mean()
        axes[1, 0].bar(quality_price.index, quality_price.values, color='orange', alpha=0.7)
        axes[1, 0].set_title('房屋质量 vs 平均价格')
        axes[1, 0].set_xlabel('质量评分')
        axes[1, 0].set_ylabel('平均价格')
        
        # 相关性热力图（前5个特征）
        numeric_features = train_data.select_dtypes(include=[np.number])
        corr_matrix = numeric_features.corr()
        top_features = corr_matrix['SalePrice'].sort_values(ascending=False).head(6).index
        top_corr = numeric_features[top_features].corr()
        
        sns.heatmap(top_corr, annot=True, cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('与价格最相关的特征')
        
        plt.tight_layout()
        plt.savefig('quick_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. 简单模型训练
        print("\n4. 训练简单模型...")
        
        # 使用相关性分析结果选择重要特征（排除SalePrice本身）
        features = [feature for feature in top_features if feature != 'SalePrice'][:5]
        print(f"基于相关性分析选择的特征: {features}")
        X = train_data[features]
        y = train_data['SalePrice']
        
        # 处理缺失值
        X = X.fillna(X.median())
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练随机森林模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"模型性能:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # 5. 特征重要性
        print("\n5. 特征重要性:")
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(importance)
        
        # 6. 对测试集进行预测
        print("\n6. 对测试集进行预测...")
        
        # 准备测试数据
        X_test_final = test_data[features].fillna(test_data[features].median())
        
        # 使用完整训练数据重新训练模型
        model_full = RandomForestRegressor(n_estimators=100, random_state=42)
        model_full.fit(X, y)
        
        # 预测
        predictions = model_full.predict(X_test_final)
        
        # 创建提交文件
        submission = pd.DataFrame({
            'Id': test_data['Id'],
            'SalePrice': predictions
        })
        
        submission.to_csv('simple_predictions.csv', index=False)
        print("预测完成！结果已保存到: simple_predictions.csv")
        
        print("\n=== 分析完成 ===")
        print("生成的文件:")
        print("- quick_analysis.png: 数据可视化图表")
        print("- simple_predictions.csv: 预测结果")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_analysis()