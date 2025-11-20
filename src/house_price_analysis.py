#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
房价预测数据分析脚本
House Prices Advanced Regression Techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class HousePriceAnalysis:
    def __init__(self, data_path):
        """初始化数据分析类"""
        self.data_path = data_path
        self.train_data = None
        self.test_data = None
        self.train_original = None
        self.test_original = None
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.train_original = pd.read_csv(f"{self.data_path}/train.csv")
        self.test_original = pd.read_csv(f"{self.data_path}/test.csv")
        
        print(f"训练集形状: {self.train_original.shape}")
        print(f"测试集形状: {self.test_original.shape}")
        
        return self.train_original, self.test_original
    
    def explore_data(self):
        """数据探索"""
        print("\n=== 数据探索 ===")
        
        # 基本信息
        print("\n1. 训练集基本信息:")
        print(self.train_original.info())
        
        print("\n2. 训练集描述性统计:")
        print(self.train_original.describe())
        
        # 目标变量分析
        print("\n3. 目标变量(SalePrice)分析:")
        print(f"目标变量描述:\n{self.train_original['SalePrice'].describe()}")
        
        # 缺失值分析
        print("\n4. 缺失值分析:")
        missing_train = self.train_original.isnull().sum()
        missing_test = self.test_original.isnull().sum()
        
        print("训练集缺失值数量:")
        print(missing_train[missing_train > 0].sort_values(ascending=False))
        
        print("\n测试集缺失值数量:")
        print(missing_test[missing_test > 0].sort_values(ascending=False))
        
        return missing_train, missing_test
    
    def visualize_data(self):
        """数据可视化"""
        print("\n=== 数据可视化 ===")
        
        # 创建可视化图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 目标变量分布
        axes[0, 0].hist(self.train_original['SalePrice'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('SalePrice分布')
        axes[0, 0].set_xlabel('SalePrice')
        axes[0, 0].set_ylabel('频数')
        
        # 2. 目标变量对数变换
        axes[0, 1].hist(np.log1p(self.train_original['SalePrice']), bins=50, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('SalePrice对数变换')
        axes[0, 1].set_xlabel('log(SalePrice)')
        axes[0, 1].set_ylabel('频数')
        
        # 3. 数值型特征相关性热力图（前10个最相关的特征）
        numeric_features = self.train_original.select_dtypes(include=[np.number])
        corr_matrix = numeric_features.corr()
        top_corr_features = corr_matrix['SalePrice'].sort_values(ascending=False).head(11).index
        top_corr_matrix = numeric_features[top_corr_features].corr()
        
        sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', ax=axes[0, 2])
        axes[0, 2].set_title('与SalePrice最相关的特征')
        
        # 4. 房屋面积与价格的关系
        axes[1, 0].scatter(self.train_original['GrLivArea'], self.train_original['SalePrice'], alpha=0.5)
        axes[1, 0].set_title('居住面积 vs SalePrice')
        axes[1, 0].set_xlabel('GrLivArea')
        axes[1, 0].set_ylabel('SalePrice')
        
        # 5. 房屋质量与价格的关系
        quality_price = self.train_original.groupby('OverallQual')['SalePrice'].mean()
        axes[1, 1].bar(quality_price.index, quality_price.values, color='orange', alpha=0.7)
        axes[1, 1].set_title('房屋质量 vs 平均价格')
        axes[1, 1].set_xlabel('OverallQual')
        axes[1, 1].set_ylabel('平均SalePrice')
        
        # 6. 建造年份与价格的关系
        year_price = self.train_original.groupby('YearBuilt')['SalePrice'].mean()
        axes[1, 2].plot(year_price.index, year_price.values, marker='o', color='red')
        axes[1, 2].set_title('建造年份 vs 平均价格')
        axes[1, 2].set_xlabel('YearBuilt')
        axes[1, 2].set_ylabel('平均SalePrice')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n=== 数据预处理 ===")
        
        # 复制原始数据
        train = self.train_original.copy()
        test = self.test_original.copy()
        
        # 保存目标变量
        y_train = train['SalePrice']
        
        # 合并训练集和测试集进行统一处理
        train_ids = train['Id']
        test_ids = test['Id']
        
        train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
        test.drop('Id', axis=1, inplace=True)
        
        # 处理缺失值
        all_data = pd.concat([train, test]).reset_index(drop=True)
        
        # 数值型特征用中位数填充
        numeric_cols = all_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            all_data[col].fillna(all_data[col].median(), inplace=True)
        
        # 分类型特征用众数填充
        categorical_cols = all_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            all_data[col].fillna(all_data[col].mode()[0], inplace=True)
        
        # 对分类型特征进行标签编码
        for col in categorical_cols:
            lbl = LabelEncoder()
            lbl.fit(all_data[col])
            all_data[col] = lbl.transform(all_data[col])
        
        # 重新分割训练集和测试集
        n_train = train.shape[0]
        self.train_data = all_data[:n_train]
        self.test_data = all_data[n_train:]
        
        print(f"预处理后训练集形状: {self.train_data.shape}")
        print(f"预处理后测试集形状: {self.test_data.shape}")
        
        return self.train_data, self.test_data, y_train
    
    def train_models(self, X_train, y_train):
        """训练多个模型"""
        print("\n=== 模型训练 ===")
        
        # 划分验证集
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # 标准化特征
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=0.1),
            'Lasso': Lasso(alpha=0.0005),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"训练 {name}...")
            
            if name in ['Linear Regression', 'Ridge', 'Lasso']:
                model.fit(X_tr_scaled, y_tr)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
            
            # 计算评估指标
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'R2': r2,
                'model': model
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        return results
    
    def plot_model_comparison(self, results):
        """模型比较可视化"""
        print("\n=== 模型性能比较 ===")
        
        models = list(results.keys())
        rmse_scores = [results[model]['RMSE'] for model in models]
        r2_scores = [results[model]['R2'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE比较
        bars1 = ax1.bar(models, rmse_scores, color='lightcoral', alpha=0.7)
        ax1.set_title('模型RMSE比较')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # R2比较
        bars2 = ax2.bar(models, r2_scores, color='lightblue', alpha=0.7)
        ax2.set_title('模型R²比较')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def main():
    """主函数"""
    # 数据路径
    data_path = "D:/Users/Administrator/Desktop/house-prices-advanced-regression-techniques (1)"
    
    # 创建分析对象
    analyzer = HousePriceAnalysis(data_path)
    
    try:
        # 1. 加载数据
        train_data, test_data = analyzer.load_data()
        
        # 2. 数据探索
        missing_train, missing_test = analyzer.explore_data()
        
        # 3. 数据可视化
        analyzer.visualize_data()
        
        # 4. 数据预处理
        X_train, X_test, y_train = analyzer.preprocess_data()
        
        # 5. 模型训练
        results = analyzer.train_models(X_train, y_train)
        
        # 6. 模型比较
        analyzer.plot_model_comparison(results)
        
        # 7. 选择最佳模型进行预测
        best_model_name = min(results.keys(), key=lambda x: results[x]['RMSE'])
        best_model = results[best_model_name]['model']
        
        print(f"\n=== 最佳模型: {best_model_name} ===")
        print(f"RMSE: {results[best_model_name]['RMSE']:.2f}")
        print(f"R²: {results[best_model_name]['R2']:.4f}")
        
        # 使用最佳模型进行预测
        if best_model_name in ['Linear Regression', 'Ridge', 'Lasso']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            best_model.fit(X_train_scaled, y_train)
            predictions = best_model.predict(X_test_scaled)
        else:
            best_model.fit(X_train, y_train)
            predictions = best_model.predict(X_test)
        
        # 创建提交文件
        submission = pd.DataFrame({
            'Id': test_data['Id'],
            'SalePrice': predictions
        })
        
        submission.to_csv('house_price_predictions.csv', index=False)
        print("\n预测结果已保存到: house_price_predictions.csv")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()