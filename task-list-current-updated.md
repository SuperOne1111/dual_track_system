# 当前任务清单 (Updated) - 四级行业双轨筛选系统

## 任务 T-07：买入信号生成器实现

### 基础信息
- **任务ID**: T-07
- **任务名称**: 买入信号生成器实现
- **对应文件**: `modules/buy_signal_generator.py`
- **前置任务**: T-01-T-06 已完成

### 详细说明
根据《技术规格说明书》v1.0 第5.1章买入信号，实现BuySignalGenerator抽象基类及其具体实现。需要支持趋势跟踪和均值回归两种策略的买入信号生成。具体包括：

1. 定义BuySignalGenerator抽象基类，包含generate_buy_signals方法
2. 实现TrendBuySignalGenerator用于趋势跟踪策略
3. 实现MeanReversionBuySignalGenerator用于均值回归策略
4. 确保信号生成过程遵循防未来函数原则（使用as_of_date参数）

### 合规检查点
- **CONS-02 (逻辑一致性)**: 信号生成逻辑必须通过多态注入，核心业务逻辑中不允许出现策略分支判断
- **DB-03 (防未来函数)**: 所有信号生成方法必须包含时间参数以防止未来函数
- **STRAT-01 (双轨隔离)**: 趋势跟踪与均值回归策略实现物理隔离

### 交付物
- `modules/buy_signal_generator.py` 文件，包含完整的买入信号生成接口和实现
- 包含基本的单元测试验证不同策略下的信号生成功能
- 符合类型注解规范的代码实现