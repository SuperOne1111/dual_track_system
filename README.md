# 四级行业双轨筛选系统

## 项目简介

这是一个基于行业层级结构的双轨量化筛选系统，通过趋势轨道捕捉主升浪行情，通过左侧轨道捕捉超跌反转机会，实现风险分散与收益增强。

### 核心特点

- **双轨策略**：趋势轨道（85%仓位）+ 左侧轨道（15%仓位）
- **层级漏斗筛选**：Level 2→3→4 逐级筛选，兼顾宏观与微观
- **动态调仓机制**：止损/持有期/评分跌破触发卖出，立即补充评分最高者
- **行业指数基准**：所有 Level 1 行业市值加权作为基准

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        数据源（PostgreSQL）                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │ 数据层   │          │ 特征层   │          │ 筛选层   │
   │ Loader  │    →    │ Trend/  │    →    │ Funnel/ │
   │Lifecycle│          │ Left    │          │Independent
   └────┬────┘          └────┬────┘          └────┬────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   回测引擎 + 动态调仓  │
                    │  Backtest Engine   │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   绩效评估 + 可视化   │
                    │  Evaluation        │
                    └───────────────────┘
```

## 安装步骤

推荐使用 `mlops_env` 虚拟环境运行：

```bash
conda activate mlops_env
```

### 1. 克隆仓库

```bash
git clone <repository_url>
cd dual_track_system
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填写数据库连接字符串
```

### 4. 验证配置

```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### 5. 强依赖数据库测试

`test_full_pipeline.py` 采用强依赖真实 PostgreSQL 策略。运行前必须设置 `PG_CONN_STRING`。

## 使用方法

### 运行完整回测

```bash
python main.py --config config.yaml
```

### 查看输出

回测结果将保存在 `output/` 目录下：
- `equity_curve.png` - 净值曲线
- `drawdown.png` - 回撤曲线
- `monthly_returns.html` - 月度收益热力图
- `performance_report.xlsx` - 绩效报告

## 项目结构

```
dual_track_system/
├── config/
│   ├── config.yaml              # 主配置文件
│   └── __init__.py
├── data/
│   ├── loader.py                # 数据加载器
│   ├── lifecycle.py             # 行业生命周期构建器
│   ├── preprocessor.py          # 数据预处理器
│   └── __init__.py
├── features/
│   ├── trend/                   # 趋势轨道特征
│   │   ├── momentum.py
│   │   ├── price_structure.py
│   │   ├── technical.py
│   │   ├── relative_strength.py
│   │   ├── momentum_quality.py
│   │   ├── score_calculator.py
│   │   └── __init__.py
│   ├── left/                    # 左侧轨道特征
│   │   ├── slowdown.py
│   │   ├── oversold.py
│   │   ├── divergence.py
│   │   ├── volatility_contraction.py
│   │   ├── volume_shrink.py
│   │   ├── score_calculator.py
│   │   └── __init__.py
│   └── __init__.py
├── selection/
│   ├── trend/
│   │   ├── funnel.py            # 漏斗筛选
│   │   └── __init__.py
│   ├── left/
│   │   ├── independent.py       # 独立筛选
│   │   └── __init__.py
│   └── __init__.py
├── rebalance/
│   ├── sell_monitor.py          # 卖出信号监控
│   ├── supplement.py            # 补充机制
│   ├── position_manager.py      # 持仓管理
│   └── __init__.py
├── backtest/
│   ├── engine.py                # 回测引擎
│   ├── benchmark.py             # 基准构建
│   ├── executor.py              # 交易执行
│   └── __init__.py
├── evaluation/
│   ├── metrics.py               # 绩效指标
│   ├── visualizer.py            # 可视化
│   ├── report.py                # 报告生成
│   └── __init__.py
├── utils/
│   ├── logger.py                # 日志配置
│   ├── helpers.py               # 工具函数
│   └── __init__.py
├── main.py                      # 主入口
├── test_full_pipeline.py        # 完整测试
├── requirements.txt
├── config.yaml
├── .env.example
└── README.md
```

## 配置说明

### 趋势轨道权重（Level 4）

| 特征类别 | 权重 |
|---------|------|
| 动量类 | 0.30 |
| 偏离度 | 0.20 |
| 波动率调整 | 0.10 |
| 相对强度 | 0.30 |
| 动量质量 | 0.10 |

### 左侧轨道权重

| 特征类别 | 权重 |
|---------|------|
| 下跌减速 | 0.20 |
| 超跌乖离 | 0.25 |
| 下影线强度 | 0.30 |
| 资金背离 | 0.15 |
| 波动+影线组合 | 0.10 |

左侧筛选参数统一使用 `left.selection`，不再使用重复配置入口。

### 风控参数

- 单行业最大仓位：10%
- 左侧单行业最大仓位：2%
- ATR止损倍数：1.5
- 固定止损：8%
- 持有期：60天

## 数据库表结构

### 1. dim.sector_industry_cn（行业元数据表）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INTEGER | 行业ID |
| name | VARCHAR | 行业名称 |
| level | INTEGER | 行业层级（1/2/3/4）|

### 2. dim.vw_asset_cn_csi_sectors（行业层级关系视图）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| csi_sector_level1~4 | VARCHAR | 各层级行业名称 |

### 3. fin.daily_sector_industry_cn（日频行情表）

| 字段名 | 类型 | 说明 |
|--------|------|------|
| trade_date | DATE | 交易日期 |
| sector_industry_id | INTEGER | 行业ID |
| open/high/low/close | FLOAT | 价格数据 |
| front_adj_close | FLOAT | 前复权收盘价 |
| turnover_rate/amount/total_market_cap | FLOAT | 交易数据 |
| daily_mfv | FLOAT | 日资金流量 |
| ma_10/ma_20/ma_60 | FLOAT | 均线 |
| volatility_20 | FLOAT | 20日波动率 |
| cmf_10/cmf_20/cmf_60 | FLOAT | CMF指标 |

## 许可证

MIT License
