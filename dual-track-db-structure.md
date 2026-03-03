# 四级行业双轨筛选系统 - 数据库结构说明书

| 文档版本     | v1.0       |
|:-------- |:---------- |
| **最后更新** | 2024-05-22 |
| **状态**   | 生产就绪       |

---

## 1. 数据库概述

### 1.1 数据库类型

- **数据库**: PostgreSQL (兼容 TimescaleDB)
- **Schema**: `dim` (维度表), `fin` (事实表)
- **字符集**: UTF-8
- **时区**: Asia/Shanghai

### 1.2 表结构总览

| 表名                             | 类型  | 用途                     | 主键                                 |
|:------------------------------ |:--- |:---------------------- |:---------------------------------- |
| `dim.sector_industry_cn`       | 表   | 行业元数据 (ID/名称/层级)       | `id`                               |
| `dim.vw_asset_cn_csi_sectors`  | 视图  | 行业层级关系映射 (L1→L2→L3→L4) | 组合键                                |
| `fin.daily_sector_industry_cn` | 表   | 行业日频行情数据               | `(trade_date, sector_industry_id)` |

---

## 2. 表结构详情

### 2.1 dim.sector_industry_cn (行业元数据表)

```sql
CREATE TABLE IF NOT EXISTS dim.sector_industry_cn
(
    id bigint NOT NULL DEFAULT nextval('dim.sector_industry_cn_id_seq'::regclass),
    name text COLLATE pg_catalog."default" NOT NULL,
    level integer NOT NULL,
    CONSTRAINT sector_industry_cn_pkey PRIMARY KEY (id),
    CONSTRAINT sector_industry_cn_name_level_key UNIQUE (name, level)
)
```

| 字段名     | 数据类型    | 约束                           | 说明             |
|:------- |:------- |:---------------------------- |:-------------- |
| `id`    | bigint  | PRIMARY KEY, DEFAULT nextval | 行业唯一标识 (自增)    |
| `name`  | text    | NOT NULL                     | 行业名称 (中文)      |
| `level` | integer | NOT NULL                     | 行业层级 (1/2/3/4) |

**唯一约束**: `(name, level)` 确保同一层级的行业名称不重复。

**示例数据**:
| id | name | level |
| :--- | :--- | :--- |
| 1001 | 信息技术 | 1 |
| 1002 | 金融 | 1 |
| 2001 | 软件开发 | 2 |
| 3001 | 应用软件 | 3 |
| 4001 | 办公软件 | 4 |

---

### 2.2 dim.vw_asset_cn_csi_sectors (行业层级关系视图)

```sql
CREATE OR REPLACE VIEW dim.vw_asset_cn_csi_sectors
 AS
 SELECT csi_sector_level1,
     csi_sector_level2,
     csi_sector_level3,
     csi_sector_level4
    FROM dim.asset_cn
   GROUP BY csi_sector_level1, csi_sector_level2, csi_sector_level3, csi_sector_level4
   ORDER BY csi_sector_level1, csi_sector_level2, csi_sector_level3, csi_sector_level4;
```

| 字段名                 | 数据类型 | 说明                               |
|:------------------- |:---- |:-------------------------------- |
| `csi_sector_level1` | text | Level 1 行业名称 (来自 `dim.asset_cn`) |
| `csi_sector_level2` | text | Level 2 行业名称                     |
| `csi_sector_level3` | text | Level 3 行业名称                     |
| `csi_sector_level4` | text | Level 4 行业名称                     |

**注意**: 

- 这是一个 **VIEW**，不是物理表。
- 数据来源于 `dim.asset_cn` 表 (需在系统中存在)。
- 用于构建行业层级映射关系，支持漏斗筛选。

**使用示例**:

```sql
-- 查询某 Level 4 行业的完整层级路径
SELECT * FROM dim.vw_asset_cn_csi_sectors
WHERE csi_sector_level4 = '办公软件';
```

---

### 2.3 fin.daily_sector_industry_cn (行业日频行情表)

```sql
CREATE TABLE IF NOT EXISTS fin.daily_sector_industry_cn
(
    trade_date date NOT NULL,
    sector_industry_id integer NOT NULL,
    open numeric(12,4),
    high numeric(12,4),
    low numeric(12,4),
    close numeric(12,4),
    front_adj_close numeric(12,4),
    turnover_rate numeric(8,4),
    amount numeric(18,4),
    total_market_cap numeric(18,4),
    daily_mfv numeric(18,4),
    created_ts timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    cmf_10 numeric(10,6),
    cmf_20 numeric(10,6),
    cmf_60 numeric(10,6),
    ma_10 numeric(12,4),
    ma_20 numeric(12,4),
    ma_60 numeric(12,4),
    volatility_20 numeric(10,6),
    CONSTRAINT daily_sector_industry_cn_pkey PRIMARY KEY (trade_date, sector_industry_id)
)
```

| 字段名                  | 数据类型          | 说明                                     | 计算来源    |
|:-------------------- |:------------- |:-------------------------------------- |:------- |
| `trade_date`         | date          | 交易日期 (主键)                              | -       |
| `sector_industry_id` | integer       | 行业 ID (外键→`dim.sector_industry_cn.id`) | -       |
| `open`               | numeric(12,4) | 开盘价                                    | 原始数据    |
| `high`               | numeric(12,4) | 最高价                                    | 原始数据    |
| `low`                | numeric(12,4) | 最低价                                    | 原始数据    |
| `close`              | numeric(12,4) | 收盘价                                    | 原始数据    |
| `front_adj_close`    | numeric(12,4) | 前复权收盘价                                 | 用于回测计算  |
| `turnover_rate`      | numeric(8,4)  | 换手率 (%)                                | 原始数据    |
| `amount`             | numeric(18,4) | 成交金额 (元)                               | 原始数据    |
| `total_market_cap`   | numeric(18,4) | 总市值 (元)                                | 用于市值加权  |
| `daily_mfv`          | numeric(18,4) | 资金流量 (Money Flow)                      | 原始数据/计算 |
| `cmf_10`             | numeric(10,6) | 10 日 Chaikin 资金流                       | 预计算指标   |
| `cmf_20`             | numeric(10,6) | 20 日 Chaikin 资金流                       | 预计算指标   |
| `cmf_60`             | numeric(10,6) | 60 日 Chaikin 资金流                       | 预计算指标   |
| `ma_10`              | numeric(12,4) | 10 日均线                                 | 预计算指标   |
| `ma_20`              | numeric(12,4) | 20 日均线                                 | 预计算指标   |
| `ma_60`              | numeric(12,4) | 60 日均线                                 | 预计算指标   |
| `volatility_20`      | numeric(10,6) | 20 日波动率                                | 预计算指标   |
| `created_ts`         | timestamp     | 数据创建时间                                 | 系统自动    |

**主键**: `(trade_date, sector_industry_id)` 确保每日每行业唯一记录。

**索引建议** (性能优化):

```sql
-- 加速按行业查询
CREATE INDEX idx_daily_sector_id ON fin.daily_sector_industry_cn(sector_industry_id);

-- 加速按日期查询
CREATE INDEX idx_daily_trade_date ON fin.daily_sector_industry_cn(trade_date);

-- 加速层级关联查询
CREATE INDEX idx_daily_sector_date ON fin.daily_sector_industry_cn(sector_industry_id, trade_date);
```

---

## 3. 数据查询规范

### 3.1 标准查询模板

```sql
-- 获取某行业在指定日期范围内的完整数据
SELECT 
    d.trade_date,
    d.sector_industry_id,
    s.name AS sector_name,
    s.level AS sector_level,
    d.open, d.high, d.low, d.close, d.front_adj_close,
    d.total_market_cap, d.daily_mfv,
    d.ma_10, d.ma_20, d.ma_60,
    d.volatility_20,
    d.cmf_10, d.cmf_20, d.cmf_60
FROM fin.daily_sector_industry_cn d
JOIN dim.sector_industry_cn s ON d.sector_industry_id = s.id
WHERE d.sector_industry_id = %s
  AND d.trade_date BETWEEN %s AND %s
ORDER BY d.trade_date;
```

### 3.2 层级关系查询模板

```sql
-- 获取某 Level 4 行业的完整层级路径
SELECT 
    v.csi_sector_level1,
    v.csi_sector_level2,
    v.csi_sector_level3,
    v.csi_sector_level4,
    s.id AS sector_id,
    s.name AS sector_name,
    s.level AS sector_level
FROM dim.vw_asset_cn_csi_sectors v
JOIN dim.sector_industry_cn s ON s.name = v.csi_sector_level4 AND s.level = 4
WHERE v.csi_sector_level4 = %s;
```

### 3.3 Level 1 市值加权指数计算模板

```sql
-- 计算每日 Level 1 市值加权指数
SELECT 
    trade_date,
    SUM(front_adj_close * total_market_cap) / SUM(total_market_cap) AS weighted_index
FROM fin.daily_sector_industry_cn d
JOIN dim.sector_industry_cn s ON d.sector_industry_id = s.id
WHERE s.level = 1
GROUP BY trade_date
ORDER BY trade_date;
```

---

## 4. 数据质量约束

### 4.1 完整性约束

| 约束项   | 要求                                                  | 验证方式     |
|:----- |:--------------------------------------------------- |:-------- |
| 行业覆盖率 | ≥ 90% 的行业每日有数据                                      | SQL 计数校验 |
| 缺失率   | < 5% 的字段允许 NULL                                     | 数据质量检查   |
| 主键唯一性 | 100% 唯一                                             | 数据库约束    |
| 外键关联  | `sector_industry_id` 必须存在于 `dim.sector_industry_cn` | 应用层校验    |

### 4.2 数据类型精度

| 字段类型 | 精度要求          | 说明                   |
|:---- |:------------- |:-------------------- |
| 价格字段 | numeric(12,4) | 保留 4 位小数             |
| 金额字段 | numeric(18,4) | 支持大额交易               |
| 比率字段 | numeric(10,6) | 保留 6 位小数 (如 CMF/波动率) |
| 日期字段 | date          | 标准日期格式               |

### 
