"""
行业生命周期构建器

构建行业生命周期表，防止未来函数（survivorship bias）
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class LifecycleBuilder:
    """行业生命周期构建器类"""
    
    def __init__(self, daily_data: pd.DataFrame):
        """
        初始化生命周期构建器
        
        Args:
            daily_data: 日频数据DataFrame
        """
        self.daily_data = daily_data.copy()
        self.lifecycle_table = None
    
    def build_lifecycle_table(self, min_gap_days: int = 5) -> pd.DataFrame:
        """
        构建行业生命周期表
        
        对每个行业，识别其连续交易区间：
        - 首次出现的日期作为生命周期开始
        - 最后出现的日期作为生命周期结束
        - 超过min_gap_days天中断视为新生命周期
        
        Args:
            min_gap_days: 超过此天数的中断视为新生命周期
        
        Returns:
            DataFrame: 生命周期表
        """
        lifecycle_records = []
        
        # 按行业分组处理
        for sector_industry_id, group in self.daily_data.groupby('sector_industry_id'):
            group = group.sort_values('trade_date')
            dates = group['trade_date'].tolist()
            
            if len(dates) == 0:
                continue
            
            # 识别连续区间
            current_start = dates[0]
            current_end = dates[0]
            
            for i in range(1, len(dates)):
                gap = (dates[i] - dates[i-1]).days
                
                if gap > min_gap_days:
                    # 中断超过阈值，保存当前区间，开始新区间
                    lifecycle_records.append({
                        'sector_industry_id': sector_industry_id,
                        'start_date': current_start,
                        'end_date': current_end,
                        'is_active': False
                    })
                    current_start = dates[i]
                
                current_end = dates[i]
            
            # 保存最后一个区间
            lifecycle_records.append({
                'sector_industry_id': sector_industry_id,
                'start_date': current_start,
                'end_date': current_end,
                'is_active': True
            })
        
        self.lifecycle_table = pd.DataFrame(lifecycle_records)
        
        print(f"构建了 {len(self.lifecycle_table)} 个行业生命周期记录")
        print(f"覆盖 {self.lifecycle_table['sector_industry_id'].nunique()} 个行业")
        
        return self.lifecycle_table
    
    def is_valid_at_date(self, sector_industry_id: int, trade_date: datetime) -> bool:
        """
        检查行业在指定日期是否有效（处于生命周期内）
        
        Args:
            sector_industry_id: 行业ID
            trade_date: 交易日期
        
        Returns:
            bool: 是否有效
        """
        if self.lifecycle_table is None:
            raise ValueError("生命周期表未构建，请先调用 build_lifecycle_table()")
        
        sector_lifecycle = self.lifecycle_table[
            self.lifecycle_table['sector_industry_id'] == sector_industry_id
        ]
        
        for _, row in sector_lifecycle.iterrows():
            if row['start_date'] <= trade_date <= row['end_date']:
                return True
        
        return False
    
    def get_valid_data(self, sector_industry_id: int, trade_date: datetime, 
                       lookback_days: int = 30) -> pd.DataFrame:
        """
        获取行业在指定日期的有效历史数据（防未来函数）
        
        在调仓日 t，对行业 i 仅使用 s <= t 且 s 在生命周期内的数据
        
        Args:
            sector_industry_id: 行业ID
            trade_date: 调仓日期
            lookback_days: 回溯天数
        
        Returns:
            DataFrame: 有效历史数据
        """
        # 计算回溯日期
        start_date = trade_date - timedelta(days=lookback_days * 2)
        
        # 筛选数据
        mask = (
            (self.daily_data['sector_industry_id'] == sector_industry_id) &
            (self.daily_data['trade_date'] <= trade_date) &
            (self.daily_data['trade_date'] >= start_date)
        )
        
        valid_data = self.daily_data[mask].copy()
        valid_data = valid_data.sort_values('trade_date')
        
        return valid_data
    
    def get_all_valid_sectors_at_date(self, trade_date: datetime) -> List[int]:
        """
        获取在指定日期有效的所有行业ID
        
        Args:
            trade_date: 交易日期
        
        Returns:
            List[int]: 有效行业ID列表
        """
        if self.lifecycle_table is None:
            raise ValueError("生命周期表未构建")
        
        valid_sectors = self.lifecycle_table[
            (self.lifecycle_table['start_date'] <= trade_date) &
            (self.lifecycle_table['end_date'] >= trade_date)
        ]['sector_industry_id'].unique().tolist()
        
        return valid_sectors
    
    def get_sector_lifecycle(self, sector_industry_id: int) -> pd.DataFrame:
        """
        获取指定行业的生命周期记录
        
        Args:
            sector_industry_id: 行业ID
        
        Returns:
            DataFrame: 该行业的生命周期记录
        """
        if self.lifecycle_table is None:
            raise ValueError("生命周期表未构建")
        
        return self.lifecycle_table[
            self.lifecycle_table['sector_industry_id'] == sector_industry_id
        ].copy()
