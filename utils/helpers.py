"""
工具函数

提供通用工具函数
"""

import yaml
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict


def load_config(config_path: str) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def date_range(start_date: str, end_date: str, 
               freq: str = 'D') -> pd.DatetimeIndex:
    """
    生成日期范围
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        freq: 频率
    
    Returns:
        DatetimeIndex: 日期索引
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def get_last_trading_day_of_month(dates: pd.DatetimeIndex) -> List[datetime]:
    """
    获取每月最后一个交易日
    
    Args:
        dates: 交易日列表
    
    Returns:
        List[datetime]: 每月最后一个交易日
    """
    df = pd.DataFrame({'date': dates})
    df['year_month'] = df['date'].dt.to_period('M')
    
    last_days = df.groupby('year_month')['date'].max().tolist()
    return last_days


def calculate_cagr(start_value: float, end_value: float, 
                   years: float) -> float:
    """
    计算年化收益率
    
    Args:
        start_value: 初始值
        end_value: 结束值
        years: 年数
    
    Returns:
        float: 年化收益率
    """
    return (end_value / start_value) ** (1 / years) - 1


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    格式化百分比
    
    Args:
        value: 数值
        decimals: 小数位数
    
    Returns:
        str: 格式化后的字符串
    """
    return f"{value:.{decimals}%}"


def format_number(value: float, decimals: int = 2) -> str:
    """
    格式化数字
    
    Args:
        value: 数值
        decimals: 小数位数
    
    Returns:
        str: 格式化后的字符串
    """
    return f"{value:,.{decimals}f}"


def build_hierarchy_mapping(sector_hierarchy: pd.DataFrame,
                            sector_metadata: pd.DataFrame) -> Dict:
    """
    从层级名称关系和元数据构建ID层级映射

    Returns:
        dict: {
          parent_map: {child_id: parent_id},
          children_by_parent: {parent_id: [child_ids...]},
          ids_by_level: {level: [ids...]}
        }
    """
    if sector_hierarchy is None or sector_metadata is None:
        raise ValueError("构建层级映射需要 sector_hierarchy 和 sector_metadata")

    meta = sector_metadata.copy()
    required_cols = {'id', 'name', 'level'}
    if not required_cols.issubset(meta.columns):
        raise ValueError(f"sector_metadata 缺少必要列: {sorted(required_cols - set(meta.columns))}")

    name_to_id = {1: {}, 2: {}, 3: {}, 4: {}}
    for _, row in meta.iterrows():
        level = int(row['level'])
        if level in name_to_id and pd.notna(row['name']):
            name_to_id[level][str(row['name'])] = int(row['id'])

    parent_map = {}
    children_by_parent = {}

    for _, row in sector_hierarchy.iterrows():
        l1 = name_to_id[1].get(str(row.get('csi_sector_level1')))
        l2 = name_to_id[2].get(str(row.get('csi_sector_level2')))
        l3 = name_to_id[3].get(str(row.get('csi_sector_level3')))
        l4 = name_to_id[4].get(str(row.get('csi_sector_level4')))

        pairs = [(l2, l1), (l3, l2), (l4, l3)]
        for child, parent in pairs:
            if child is None or parent is None:
                continue
            existing = parent_map.get(child)
            if existing is not None and existing != parent:
                raise ValueError(f"层级映射冲突: child={child}, parent={existing}/{parent}")
            parent_map[child] = parent
            children_by_parent.setdefault(parent, set()).add(child)

    ids_by_level = {
        int(level): sorted(group['id'].astype(int).unique().tolist())
        for level, group in meta.groupby('level')
    }
    children_by_parent = {k: sorted(list(v)) for k, v in children_by_parent.items()}

    return {
        'parent_map': parent_map,
        'children_by_parent': children_by_parent,
        'ids_by_level': ids_by_level
    }
