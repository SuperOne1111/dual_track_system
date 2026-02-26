"""
数据预处理器

计算基础指标、处理缺失值、数据验证
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


class DataPreprocessor:
    """数据预处理器类"""
    
    def __init__(self, daily_data: pd.DataFrame):
        """
        初始化数据预处理器
        
        Args:
            daily_data: 日频数据DataFrame
        """
        self.daily_data = daily_data.copy()
        self.processed_data = None
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        计算对数收益率
        
        计算 r_1, r_3, r_7, r_14, r_30
        r_n(t) = ln(P(t) / P(t-n))
        
        Returns:
            DataFrame: 添加了收益率列的数据
        """
        df = self.daily_data.copy()
        
        # 按行业分组计算收益率
        def calc_group_returns(group):
            group = group.sort_values('trade_date')
            
            # 使用 front_adj_close 计算收益率
            for n in [1, 3, 7, 14, 30]:
                group[f'r_{n}'] = np.log(
                    group['front_adj_close'] / group['front_adj_close'].shift(n)
                )
            
            return group
        
        df = df.groupby('sector_industry_id', group_keys=False).apply(calc_group_returns)
        
        self.processed_data = df
        print("完成对数收益率计算 (r_1, r_3, r_7, r_14, r_30)")
        return df
    
    def handle_missing_values(self, method: str = 'forward_fill') -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            method: 处理方法，'forward_fill' 或 'linear'
        
        Returns:
            DataFrame: 处理后的数据
        """
        if self.processed_data is None:
            raise ValueError("请先调用 calculate_returns()")
        
        df = self.processed_data.copy()
        
        # 按行业分组处理
        def fill_group(group):
            group = group.sort_values('trade_date')
            
            # 价格数据使用前向填充
            price_cols = ['open', 'high', 'low', 'close', 'front_adj_close']
            for col in price_cols:
                if col in group.columns:
                    group[col] = group[col].ffill()
            
            # 收益率数据使用0填充
            return_cols = [f'r_{n}' for n in [1, 3, 7, 14, 30]]
            for col in return_cols:
                if col in group.columns:
                    group[col] = group[col].fillna(0)
            
            # 其他数值型数据前向填充
            other_cols = ['turnover_rate', 'amount', 'total_market_cap', 
                         'daily_mfv', 'ma_10', 'ma_20', 'ma_60', 
                         'volatility_20', 'cmf_10', 'cmf_20', 'cmf_60']
            for col in other_cols:
                if col in group.columns:
                    group[col] = group[col].ffill()
            
            return group
        
        df = df.groupby('sector_industry_id', group_keys=False).apply(fill_group)
        
        # 删除仍有缺失值的行
        before_drop = len(df)
        df = df.dropna()
        after_drop = len(df)
        
        if before_drop > after_drop:
            print(f"删除了 {before_drop - after_drop} 条有缺失值的记录")
        
        self.processed_data = df
        return df
    
    def validate_data(self, coverage_threshold: float = 0.90,
                     missing_threshold: float = 0.05) -> Tuple[bool, Dict]:
        """
        验证数据质量
        
        Args:
            coverage_threshold: 数据覆盖率阈值
            missing_threshold: 缺失率阈值
        
        Returns:
            Tuple[bool, Dict]: (验证是否通过, 验证结果详情)
        """
        if self.processed_data is None:
            raise ValueError("请先调用 calculate_returns() 和 handle_missing_values()")
        
        df = self.processed_data
        results = {
            'passed': True,
            'checks': {}
        }
        
        # 1. 检查数据覆盖率
        total_sectors = df['sector_industry_id'].nunique()
        total_dates = df['trade_date'].nunique()
        expected_records = total_sectors * total_dates
        actual_records = len(df)
        coverage = actual_records / expected_records if expected_records > 0 else 0
        
        results['checks']['coverage'] = {
            'value': coverage,
            'threshold': coverage_threshold,
            'passed': coverage >= coverage_threshold
        }
        
        if coverage < coverage_threshold:
            results['passed'] = False
            print(f"警告: 数据覆盖率 {coverage:.2%} 低于阈值 {coverage_threshold:.2%}")
        
        # 2. 检查缺失率
        missing_rates = {}
        for col in df.columns:
            if col not in ['trade_date', 'sector_industry_id']:
                missing_rate = df[col].isna().sum() / len(df)
                missing_rates[col] = missing_rate
                
                if missing_rate > missing_threshold:
                    results['passed'] = False
                    print(f"警告: 列 {col} 的缺失率 {missing_rate:.2%} 超过阈值 {missing_threshold:.2%}")
        
        results['checks']['missing_rates'] = missing_rates
        
        # 3. 检查价格数据合理性
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                negative_prices = (df[col] <= 0).sum()
                if negative_prices > 0:
                    results['passed'] = False
                    print(f"警告: 列 {col} 有 {negative_prices} 条非正价格记录")
        
        # 4. 检查收益率异常值
        for n in [1, 3, 7, 14, 30]:
            col = f'r_{n}'
            if col in df.columns:
                extreme_returns = (abs(df[col]) > 0.5).sum()
                if extreme_returns > len(df) * 0.01:  # 超过1%的极端收益率
                    print(f"警告: 列 {col} 有 {extreme_returns} 条极端收益率记录")
        
        if results['passed']:
            print("数据验证通过！")
        
        return results['passed'], results
    
    def preprocess(self, coverage_threshold: float = 0.90,
                  missing_threshold: float = 0.05) -> pd.DataFrame:
        """
        执行完整预处理流程
        
        Args:
            coverage_threshold: 数据覆盖率阈值
            missing_threshold: 缺失率阈值
        
        Returns:
            DataFrame: 预处理后的数据
        """
        # 1. 计算收益率
        self.calculate_returns()
        
        # 2. 处理缺失值
        self.handle_missing_values()
        
        # 3. 验证数据
        passed, results = self.validate_data(coverage_threshold, missing_threshold)
        
        if not passed:
            print("警告: 数据验证未完全通过，但继续处理")
        
        print(f"预处理完成，最终数据量: {len(self.processed_data)} 条记录")
        return self.processed_data
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        获取预处理后的数据
        
        Returns:
            DataFrame: 预处理后的数据
        """
        if self.processed_data is None:
            raise ValueError("数据尚未预处理，请先调用 preprocess()")
        return self.processed_data.copy()
