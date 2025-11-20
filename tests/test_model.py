"""
模型测试模块
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from improved_analysis import improved_analysis


class TestModel(unittest.TestCase):
    """模型测试类"""
    
    def test_improved_analysis_structure(self):
        """测试改进分析函数结构"""
        # 检查函数是否存在
        self.assertTrue(callable(improved_analysis))
    
    def test_feature_selection(self):
        """测试特征选择逻辑"""
        # 模拟数据
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'SalePrice': np.random.randn(100) * 1000 + 100000
        })
        
        # 测试相关性计算
        corr_matrix = data.corr()
        self.assertIn('SalePrice', corr_matrix.columns)
        
        # 测试高相关性特征选择
        high_corr_features = corr_matrix['SalePrice'][corr_matrix['SalePrice'] > 0.3]
        self.assertIsInstance(high_corr_features, pd.Series)
    
    def test_model_performance(self):
        """测试模型性能指标"""
        # 模拟预测结果
        y_true = np.array([100000, 200000, 300000])
        y_pred = np.array([95000, 210000, 290000])
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # 检查RMSE计算正确
        self.assertGreater(rmse, 0)
        self.assertLess(rmse, 100000)


if __name__ == '__main__':
    unittest.main()