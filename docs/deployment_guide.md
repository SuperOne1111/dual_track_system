# 四级行业双轨筛选系统部署操作手册

## 1. 部署概述

本文档详细描述了四级行业双轨筛选系统的生产环境部署流程。系统遵循《dual-track-tech-spec》v1.0 和《dual-track-api-design》v1.2 的设计要求，确保部署过程满足合规性标准。

### 1.1 部署目标
- 完成生产环境配置
- 验证系统功能完整性
- 建立监控和健康检查机制
- 确保合规性要求得到满足

### 1.2 部署前检查清单
- [ ] 服务器环境已准备就绪
- [ ] 数据库连接已配置
- [ ] Python 3.10+ 已安装
- [ ] 所需依赖包已安装
- [ ] 系统资源满足最低要求
- [ ] 防火墙规则已配置

## 2. 环境准备

### 2.1 系统要求
- **操作系统**: Linux (推荐 Ubuntu 20.04+/CentOS 8+)
- **Python版本**: 3.10 或更高
- **内存**: 最少 8GB RAM
- **磁盘空间**: 最少 50GB 可用空间
- **网络**: 数据库连接可达

### 2.2 依赖安装
```bash
# 安装Python依赖
pip install --upgrade pip
pip install -r requirements.txt

# 或者安装特定依赖
pip install pandas numpy psycopg2-binary
```

### 2.3 目录结构准备
确保以下目录存在并具有适当权限：
```
/workspace/
├── data/
│   └── prod_states/          # 生产状态文件存储
├── config/                   # 配置文件目录
├── logs/                     # 日志文件目录
├── scripts/                  # 脚本目录
└── modules/                  # 模块目录
```

## 3. 配置文件设置

### 3.1 生产环境配置
编辑 `config/production.yaml` 文件，设置生产环境参数：

```yaml
mode: "production_daily"

database:
  host: "your_db_host"
  port: 5432
  database: "your_database"
  username: "your_username"
  password: "your_password"
  timeout: 30

paths:
  data_dir: "./data"
  state_dir: "./data/prod_states"
  log_dir: "./logs"

trading:
  trend_weight: 0.85
  left_weight: 0.15
  max_l1_exposure: 0.20
  max_l2_exposure: 0.10
  volatility_target: 0.15

risk_control:
  atr_stop_loss: 1.5
  fixed_stop_loss: 0.08
  max_holding_days: 60
  cooldown_period: 10

monitoring:
  factor_ic_threshold: 0.05
  ic_win_rate_threshold: 0.30
  position_limit_warning: 0.80
```

### 3.2 状态文件初始化
创建必要的状态文件：

```bash
mkdir -p data/prod_states
touch data/prod_states/state_cool_down.json
touch data/prod_states/state_factor_weights.json
touch data/prod_states/state_portfolio_snapshot.csv

# 初始化冷却池文件
echo "{}" > data/prod_states/state_cool_down.json

# 初始化因子权重文件（使用默认权重）
cat > data/prod_states/state_factor_weights.json << EOF
{
  "trend": {
    "momentum": 0.30,
    "structure": 0.15,
    "technical": 0.10,
    "relative_strength": 0.35,
    "quality": 0.10
  },
  "left": {
    "slowdown": 0.37,
    "oversold": 0.30,
    "divergence": 0.19,
    "vol_contraction": 0.12,
    "vol_reduction": 0.11
  }
}
EOF
```

## 4. 部署验证流程

### 4.1 系统健康检查
运行健康检查脚本验证系统完整性：

```bash
python scripts/health_check.py
```

预期输出应显示所有检查项为 PASS 状态。

### 4.2 监控系统验证
验证监控功能：

```bash
python scripts/monitor_prod.py
```

确认监控报告正常生成并保存到 `data/prod_states/monitoring_reports/` 目录。

### 4.3 数据连接验证
确认能正确连接数据库并获取数据：

```bash
# 可以运行一个简单的数据验证脚本
python -c "
from modules.data_layer import DataLoader
loader = DataLoader()
import datetime
data = loader.load_market_data(
    start_date=datetime.date.today() - datetime.timedelta(days=30),
    end_date=datetime.date.today(),
    as_of_date=datetime.date.today()
)
print(f'Data loaded: {len(data)} records')
"
```

## 5. 部署后配置

### 5.1 定时任务设置
根据业务需求设置定时任务（crontab）：

```bash
# 示例：每天收盘后运行策略（假设15:30）
30 15 * * 1-5 cd /path/to/workspace && python main.py >> logs/strategy.log 2>&1

# 每小时运行健康检查
0 * * * * cd /path/to/workspace && python scripts/health_check.py >> logs/health_check.log 2>&1

# 每天早上运行监控报告
30 9 * * 1-5 cd /path/to/workspace && python scripts/monitor_prod.py >> logs/monitoring.log 2>&1
```

### 5.2 日志轮转配置
配置日志轮转以避免日志文件过大：

```bash
# 创建 logrotate 配置文件 /etc/logrotate.d/dual-track-system
/path/to/workspace/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

## 6. 合规性验证

### 6.1 RUN-01 (脚本化运行) 验证
- 系统不包含任何守护进程或常驻服务
- 所有功能通过脚本方式调用
- 验证命令：`ps aux | grep dual-track` 不应显示任何常驻进程

### 6.2 RUN-02 (原子写入) 验证
- 所有状态文件写入使用 `atomic_write_json` 函数
- 验证函数在 `utils/io.py` 中的实现
- 确保在写入过程中发生中断不会导致文件损坏

### 6.3 DB-03 (防未来函数) 验证
- 所有数据访问包含 `as_of_date` 参数
- 数据库查询使用 PIT (Point-in-Time) 原则
- 验证数据加载函数的实现

### 6.4 CONS-02 (逻辑一致性) 验证
- 核心逻辑无模式分支
- 通过依赖注入实现不同环境的行为差异
- 验证 `StateRepository` 模式的实现

## 7. 监控与运维

### 7.1 监控指标
系统应监控以下关键指标：

- **因子稳定性**: IC值、IC胜率、因子权重
- **系统健康**: 状态文件完整性、数据连接可用性
- **投资组合风险**: 仓位水平、波动率、集中度
- **执行性能**: 脚本运行时间、资源使用情况

### 7.2 告警配置
设置以下告警规则：

- 因子IC连续低于阈值超过X天
- 投资组合波动率超过目标阈值
- 系统健康检查失败
- 脚本执行时间异常延长

### 7.3 应急响应
如遇系统异常，按以下步骤处理：

1. 检查健康检查报告
2. 查看相关日志文件
3. 验证数据库连接
4. 检查状态文件完整性
5. 如需，回滚到上一个稳定状态

## 8. 部署验证清单

部署完成后，确认以下项目已完成：

- [ ] 生产配置文件已正确设置
- [ ] 状态文件已初始化
- [ ] 健康检查脚本运行正常
- [ ] 监控脚本运行正常
- [ ] 数据库连接验证通过
- [ ] 定时任务已设置
- [ ] 日志轮转已配置
- [ ] 合规性验证通过
- [ ] 应急响应计划已制定

## 9. 后续维护

### 9.1 日常维护
- 每日检查监控报告
- 定期清理过期日志
- 监控系统资源使用情况

### 9.2 定期检查
- 每周验证系统功能完整性
- 每月审查因子权重配置
- 每季度评估系统性能

---
*文档版本: v1.0*  
*最后更新: 2024-05-22*  
*依据: 《dual-track-tech-spec》v1.0, 《dual-track-api-design》v1.2*