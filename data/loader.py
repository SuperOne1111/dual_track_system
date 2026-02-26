"""
数据加载器

从PostgreSQL数据库加载行业数据
"""

import os
import pandas as pd
import psycopg2
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, conn_string: Optional[str] = None):
        """
        初始化数据加载器
        
        Args:
            conn_string: 数据库连接字符串，如果为None则从环境变量读取
        """
        self.conn_string = conn_string or os.getenv('PG_CONN_STRING')
        if not self.conn_string:
            raise ValueError("数据库连接字符串未提供，请设置 PG_CONN_STRING 环境变量")
    
    def _get_connection(self):
        """获取数据库连接"""
        return psycopg2.connect(self.conn_string)
    
    def test_connection(self) -> bool:
        """
        测试数据库连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            print("数据库连接测试成功！")
            return True
        except Exception as e:
            print(f"数据库连接测试失败: {e}")
            return False
    
    def load_sector_metadata(self) -> pd.DataFrame:
        """
        加载行业元数据
        
        Returns:
            DataFrame: 行业元数据，包含 id, name, level 字段
        """
        query = """
        SELECT 
            id,
            name,
            level
        FROM dim.sector_industry_cn
        ORDER BY level, id
        """
        
        conn = self._get_connection()
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"加载了 {len(df)} 条行业元数据记录")
        return df
    
    def load_sector_hierarchy(self) -> pd.DataFrame:
        """
        加载行业层级关系
        
        Returns:
            DataFrame: 行业层级关系，包含 csi_sector_level1~4 字段
        """
        query = """
        SELECT 
            csi_sector_level1,
            csi_sector_level2,
            csi_sector_level3,
            csi_sector_level4
        FROM dim.vw_asset_cn_csi_sectors
        ORDER BY csi_sector_level1, csi_sector_level2, csi_sector_level3, csi_sector_level4
        """
        
        conn = self._get_connection()
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"加载了 {len(df)} 条行业层级关系记录")
        return df
    
    def load_daily_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载日频行情数据
        
        Args:
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
        
        Returns:
            DataFrame: 日频行情数据
        """
        query = """
        SELECT 
            trade_date,
            sector_industry_id,
            open,
            high,
            low,
            close,
            front_adj_close,
            turnover_rate,
            amount,
            total_market_cap,
            daily_mfv,
            ma_10,
            ma_20,
            ma_60,
            volatility_20,
            cmf_10,
            cmf_20,
            cmf_60
        FROM fin.daily_sector_industry_cn
        WHERE trade_date BETWEEN %s AND %s
        ORDER BY sector_industry_id, trade_date
        """
        
        conn = self._get_connection()
        df = pd.read_sql(query, conn, params=(start_date, end_date))
        conn.close()
        
        # 转换日期类型
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        print(f"加载了 {len(df)} 条日频数据记录，时间范围: {start_date} 至 {end_date}")
        return df
    
    def load_all_data(self, start_date: str, end_date: str) -> dict:
        """
        加载所有数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            dict: 包含所有数据的字典
        """
        return {
            'sector_metadata': self.load_sector_metadata(),
            'sector_hierarchy': self.load_sector_hierarchy(),
            'daily_data': self.load_daily_data(start_date, end_date)
        }
