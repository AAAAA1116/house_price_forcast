#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版房价预测分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def improved_analysis():
    print("=== 改进版房价预测分析 ===")
    
    # 数据路径
    data_path = "D:/Users/Administrator/Desktop/house-prices-advanced-regression-techniques (1)"
    
    try:
        # 1. 加载数据
        print("1. 加载数据...")
        train_data = pd.read_csv(f"{data_path}/train.csv")
        test_data = pd.read_csv(f"{data_path}/test.csv")
        
        print(f"训练集: {train_data.shape}")
        print(f"测试集: {test_data.shape}")
        
        # 2. 更深入的数据探索
        print("\n2. 深入数据探索...")
        
        # 目标变量分析
        print(f"目标变量范围: ${train_data['SalePrice'].min():,} - ${train_data['SalePrice'].max():,}")
        print(f"目标变量偏度: {train_data['SalePrice'].skew():.3f}")
        
        # 对数变换目标变量（处理偏度）
        y_log = np.log1p(train_data['SalePrice'])
        print(f"对数变换后偏度: {y_log.skew():.3f}")
        
        # 3. 特征工程
        print("\n3. 特征工程...")
        
        # 选择更多相关特征
        numeric_features = train_data.select_dtypes(include=[np.number])
        corr_matrix = numeric_features.corr()
        
        # 选择与SalePrice相关性大于0.3的特征
        high_corr_features = corr_matrix['SalePrice'][corr_matrix['SalePrice'] > 0.3].sort_values(ascending=False)
        print(f"高相关性特征数量: {len(high_corr_features)}")
        print("高相关性特征:")
        for feature, corr in high_corr_features.items():
            if feature != 'SalePrice':
                print(f"  {feature}: {corr:.3f}")
        
        # 选择前15个最相关的特征
        selected_features = [feature for feature in high_corr_features.index if feature != 'SalePrice'][:15]
        print(f"\n选择的特征数量: {len(selected_features)}")
        
        # 4. 数据预处理
        print("\n4. 数据预处理...")
        
        X = train_data[selected_features]
        y = y_log  # 使用对数变换后的目标变量
        
        # 更细致的缺失值处理
        for col in selected_features:
            if X[col].isnull().sum() > 0:
                if X[col].dtype in ['int64', 'float64']:
                    X[col].fillna(X[col].median(), inplace=True)
                else:
                    X[col].fillna(X[col].mode()[0], inplace=True)
        
        # 5. 模型训练与优化
        print("\n5. 模型训练与优化...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 参数网格搜索
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"最佳参数: {grid_search.best_params_}")
        
        # 预测（需要将结果转换回原始尺度）
        y_pred_log = best_model.predict(X_test)
        y_pred = np.expm1(y_pred_log)  # 逆对数变换
        y_test_original = np.expm1(y_test)  # 逆对数变换
        
        # 评估指标
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        r2 = r2_score(y_test_original, y_pred)
        
        print(f"\n改进后模型性能:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # 交叉验证
        cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"5折交叉验证RMSE: {cv_rmse:.2f}")
        
        # 6. 特征重要性
        print("\n6. 特征重要性排名:")
        importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(importance.head(10))
        
        # 7. 对测试集进行预测
        print("\n7. 对测试集进行预测...")
        
        # 准备测试数据
        X_test_final = test_data[selected_features].copy()
        for col in selected_features:
            if X_test_final[col].isnull().sum() > 0:
                if X_test_final[col].dtype in ['int64', 'float64']:
                    X_test_final[col].fillna(train_data[col].median(), inplace=True)
                else:
                    X_test_final[col].fillna(train_data[col].mode()[0], inplace=True)
        
        # 使用完整训练数据重新训练最佳模型
        best_model_full = RandomForestRegressor(**grid_search.best_params_, random_state=42)
        best_model_full.fit(X, y)
        
        # 预测
        predictions_log = best_model_full.predict(X_test_final)
        predictions = np.expm1(predictions_log)
        
        # 创建提交文件
        submission = pd.DataFrame({
            'Id': test_data['Id'],
            'SalePrice': predictions
        })
        
        submission.to_csv('improved_predictions.csv', index=False)
        print("改进版预测完成！结果已保存到: improved_predictions.csv")
        
        print("\n=== 改进分析完成 ===")
        print("主要改进:")
        print("- 使用更多高相关性特征")
        print("- 目标变量对数变换处理偏度")
        print("- 网格搜索优化超参数")
        print("- 交叉验证评估模型稳定性")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    improved_analysis()