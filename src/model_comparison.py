"""
模型比较模块
用于比较不同机器学习模型的性能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def compare_models(X, y):
    """
    比较不同机器学习模型的性能
    
    Args:
        X: 特征数据
        y: 目标变量
        
    Returns:
        DataFrame: 模型性能比较结果
    """
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        # 5折交叉验证
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        
        results.append({
            'Model': name,
            'Mean RMSE': rmse_scores.mean(),
            'Std RMSE': rmse_scores.std(),
            'Min RMSE': rmse_scores.min(),
            'Max RMSE': rmse_scores.max()
        })
    
    return pd.DataFrame(results).sort_values('Mean RMSE')


def plot_model_comparison(results_df):
    """
    绘制模型比较结果图
    
    Args:
        results_df: 模型比较结果DataFrame
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE比较图
    ax1.barh(results_df['Model'], results_df['Mean RMSE'])
    ax1.set_xlabel('RMSE')
    ax1.set_title('模型RMSE比较')
    ax1.grid(True, alpha=0.3)
    
    # 标准差比较图
    ax2.barh(results_df['Model'], results_df['Std RMSE'])
    ax2.set_xlabel('RMSE标准差')
    ax2.set_title('模型稳定性比较')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 示例用法
    print("模型比较模块")
    print("使用方法: from src.model_comparison import compare_models, plot_model_comparison")