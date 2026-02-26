# 四级行业双轨筛选系统 - 组装指南

## 项目概述

这是一个基于行业层级结构的双轨量化筛选系统，通过趋势轨道捕捉主升浪行情，通过左侧轨道捕捉超跌反转机会。

## 项目结构

```
dual_track_system/
├── config/
│   └── __init__.py              # 配置模块
├── data/                        # 数据层模块
│   ├── __init__.py
│   ├── loader.py                # 数据加载器
│   ├── lifecycle.py             # 行业生命周期构建器
│   └── preprocessor.py          # 数据预处理器
├── features/                    # 特征层模块
│   ├── __init__.py
│   ├── trend/                   # 趋势轨道特征
│   │   ├── __init__.py
│   │   ├── momentum.py          # 动量类特征
│   │   ├── price_structure.py   # 价格结构类特征
│   │   ├── technical.py         # 技术指标类特征
│   │   ├── relative_strength.py # 相对强度类特征
│   │   ├── momentum_quality.py  # 动量质量类特征
│   │   └── score_calculator.py  # 综合得分计算
│   └── left/                    # 左侧轨道特征
│       ├── __init__.py
│       ├── slowdown.py          # 下跌减速类特征
│       ├── oversold.py          # 超跌类特征
│       ├── divergence.py        # 资金背离类特征
│       ├── volatility_contraction.py  # 波动收缩类特征
│       ├── volume_shrink.py     # 成交量萎缩类特征
│       └── score_calculator.py  # 综合得分计算
├── selection/                   # 筛选层模块
│   ├── __init__.py
│   ├── trend/                   # 趋势筛选
│   │   ├── __init__.py
│   │   └── funnel.py            # 漏斗式筛选
│   └── left/                    # 左侧筛选
│       ├── __init__.py
│       └── independent.py       # 独立筛选
├── rebalance/                   # 动态调仓模块
│   ├── __init__.py
│   ├── sell_monitor.py          # 卖出信号监控
│   ├── supplement.py            # 补充机制
│   └── position_manager.py      # 持仓管理
├── backtest/                    # 回测引擎模块
│   ├── __init__.py
│   ├── engine.py                # 回测主引擎
│   ├── benchmark.py             # 基准构建
│   └── executor.py              # 交易执行
├── evaluation/                  # 绩效评估模块
│   ├── __init__.py
│   ├── metrics.py               # 绩效指标计算
│   ├── visualizer.py            # 可视化
│   └── report.py                # 报告生成
├── utils/                       # 工具模块
│   ├── __init__.py
│   ├── logger.py                # 日志配置
│   └── helpers.py               # 工具函数
├── main.py                      # 主入口
├── test_full_pipeline.py        # 完整流程测试
├── config.yaml                  # 配置文件
├── requirements.txt             # 依赖列表
├── .env.example                 # 环境变量模板
└── README.md                    # 项目说明
```

## 安装步骤

建议使用固定环境：

```bash
conda activate mlops_env
```

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填写数据库连接字符串
```

### 3. 验证配置

```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### 4. 运行测试

```bash
python test_full_pipeline.py
```

注意：测试采用强依赖真实 PostgreSQL 策略，必须先设置 `PG_CONN_STRING`。

## 使用方法

### 运行完整回测

```bash
python main.py --config config.yaml
```

### 测试数据库连接

```bash
python main.py --test-connection
```

### 查看输出

回测结果将保存在 `output/` 目录下：
- `equity_curve.png` - 净值曲线
- `drawdown.png` - 回撤曲线
- `monthly_returns.html` - 月度收益热力图
- `performance_report_YYYYMMDD_HHMMSS.xlsx` - 绩效报告

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

## 核心算法

### 趋势综合得分

```
S_trend = w1*动量 + w2*偏离度 + w3*波动率调整 + w4*相对强度 + w5*动量质量
```

### 左侧综合得分

```
S_recovery = 0.20*S_slow + 0.25*S_bias + 0.30*S_shadow + 0.15*S_flow + 0.10*S_vol_shadow
```

### 卖出信号

1. ATR动态止损：回撤 < -1.5 * ATR_20
2. 固定止损：回撤 > 8%
3. 持有期到期：持有天数 ≥ 60
4. 评分跌破：S < 50

## 常见问题

### Q1: 数据库连接失败怎么办？

**A:** 检查 `.env` 文件中的 `PG_CONN_STRING` 是否正确配置。

### Q2: 如何修改回测参数？

**A:** 编辑 `config.yaml` 文件，修改相应的配置项。

### Q3: 如何添加新的特征？

**A:** 在 `features/trend/` 或 `features/left/` 目录下添加新的特征计算器模块。

### Q4: 如何调整权重？

**A:** 修改 `config.yaml` 中的 `trend.weights` 或 `left.score_weights`。

## 性能优化建议

1. **数据缓存**：考虑将预处理后的数据缓存到本地，避免重复计算
2. **并行计算**：特征计算可以并行化处理
3. **增量更新**：只计算新增日期的特征

## 扩展开发

### 添加新的特征类别

1. 在 `features/trend/` 或 `features/left/` 创建新的特征计算器
2. 在 `__init__.py` 中导入并集成
3. 更新配置文件中的权重

### 添加新的筛选策略

1. 在 `selection/` 目录下创建新的筛选器
2. 实现 `select()` 方法
3. 在回测引擎中调用

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

## 许可证

MIT License
