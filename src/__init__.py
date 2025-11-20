# 房价预测项目源代码包
# 包含数据分析、模型训练和预测功能

__version__ = "1.0.0"
__author__ = "House Price Prediction Team"

from .house_price_analysis import HousePriceAnalysis
from .improved_analysis import improved_analysis

__all__ = ['HousePriceAnalysis', 'improved_analysis']