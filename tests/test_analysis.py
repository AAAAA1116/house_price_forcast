"""
数据分析测试模块
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from house_price_analysis import HousePriceAnalysis


class TestHousePriceAnalysis(unittest.TestCase):
    """房价分析测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.analysis = HousePriceAnalysis("D:/Users/Administrator/Desktop/house-prices-advanced-regression-techniques (1)")
        
    def test_load_data(self):
        """测试数据加载"""
        train_data, test_data = self.analysis.load_data()
        
        # 检查数据形状
        self.assertGreater(train_data.shape[0], 0)
        self.assertGreater(test_data.shape[0], 0)
        
        # 检查列名
        self.assertIn('SalePrice', train_data.columns)
        self.assertIn('Id', test_data.columns)
    
    def test_explore_data(self):
        """测试数据探索"""
        train_data, _ = self.analysis.load_data()
        
        # 检查目标变量
        self.assertGreater(train_data['SalePrice'].min(), 0)
        self.assertLess(train_data['SalePrice'].max(), 1000000)
        
        # 检查数据类型
        self.assertIsInstance(train_data, pd.DataFrame)
    
    def test_preprocess_data(self):
        """测试数据预处理"""
        train_data, _ = self.analysis.load_data()
        X, y = self.analysis.preprocess_data(train_data)
        
        # 检查预处理结果
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        
        # 检查没有缺失值
        self.assertEqual(X.isnull().sum().sum(), 0)
        self.assertEqual(y.isnull().sum(), 0)


if __name__ == '__main__':
    unittest.main()