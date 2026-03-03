# 四级行业双轨筛选系统技术规格说明书



---

## 1. 系统概览

### 1.1 核心目标

本系统旨在构建一个**分散但不过度分散**、能够获取**超额收益**的行业轮动投资组合。通过**物理隔离的双轨策略**捕捉不同市场风格下的 Alpha，采用**事件驱动调仓机制**降低无效换手。系统基于**零数据库变更**原则设计，状态管理采用**内存/本地文件**混合模式，适配**脚本化手动/定时运行**。

| 轨道        | 目标仓位       | 功能定位              | 标准化域      |
|:--------- |:---------- |:----------------- |:--------- |
| **趋势轨道**  | 85% (动态调整) | 进攻主力，跟随市场强势方向     | 全局横截面     |
| **左侧轨道**  | 15% (动态调整) | 防御反击，博弈超跌反弹       | 左侧候选池内横截面 |
| **现金/其他** | 0% (基准满仓)  | 仅在波动率控仓或无标的时产生，计息 | 无风险收益率    |

### 1.2 部署模式

系统支持两种运行模式，配置隔离，**均为非驻留脚本 (Batch Script)**：

| 模式                          | 用途       | 状态存储方式         | 关键差异                                          |
|:--------------------------- |:-------- |:-------------- |:--------------------------------------------- |
| **Backtest (回测)**           | 评估历史表现   | **纯内存对象**      | 使用固定成本、全量数据、严格防未来函数。**包含内存版冷却池模拟**，确保与实盘逻辑一致。 |
| **Production Daily (生产日频)** | 生成实际投资建议 | **本地 JSON 文件** | **T 日收盘计算，T+1 日开盘执行**，动态滑点，状态跨日持久化。           |

### 1.3 关键特性

| 特性           | 说明                                                    |
|:------------ |:----------------------------------------------------- |
| **双轨物理隔离**   | 趋势与左侧策略**独立计算、独立标准化、独立筛选**，仅在组合构建层合并权重                |
| **事件驱动调仓**   | **卖出触发式补仓**。无卖出信号则持有，不强制定期调仓，降低换手率                    |
| **全局冷却机制**   | 卖出后的行业 **10 个交易日内** 禁止重新入选任一轨道，状态存于本地文件 (实盘) 或内存 (回测) |
| **数据鲁棒性增强**  | 缺失值跳过、异常值截断、**业务逻辑硬阈值校验 (Sanity Check)**              |
| **零 DDL 架构** | **不新增任何数据库表/字段**，依赖现有 3 张表/视图，状态数据本地化                 |
| **防未来函数设计**  | 依赖个股历史行业标签动态聚合 (PIT)，滑点基于历史成交额估算                      |
| **初始化容错**    | 首次运行或状态文件丢失时，自动加载默认配置，不报错崩溃                           |
| **单一可信源**    | 核心指标 (如 Level 1 指数) 统一计算入口，杜绝逻辑偏差                     |

---

## 2. 数据架构与流转

### 2.1 数据库 Schema (只读不可变)

系统严格依赖以下 PostgreSQL 表/视图，**禁止修改字段名、表名或新增任何结构**：

| 表/视图名                          | 类型  | 核心字段 (必须包含)                                                                                                                                                            | 说明                           |
|:------------------------------ |:--- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:---------------------------- |
| `dim.sector_industry_cn`       | 表   | `id`, `name`, `level` (1/2/3/4)                                                                                                                                        | 行业维度表                        |
| `dim.vw_asset_cn_csi_sectors`  | 视图  | `csi_sector_level1`, `csi_sector_level2`, `csi_sector_level3`, `csi_sector_level4`                                                                                     | **个股 - 行业映射视图 (含历史 PIT 信息)** |
| `fin.daily_sector_industry_cn` | 表   | `sector_industry_id`, `trade_date`, `open`, `high`, `low`, `close`, `front_adj_close`, `total_market_cap`, `daily_mfv`, `ma_10/20/60`, `volatility_20`, `cmf_10/20/60` | 行情与因子表                       |

**注意**：所有代码实现中必须严格使用 `csi_sector_level1` 至 `csi_sector_level4` 字段名，严禁使用简写。

### 2.2 状态管理架构 (本地文件)

由于数据库不可变更且系统为脚本化运行，跨日状态通过本地文件管理。首次运行文件不存在时，系统自动初始化默认值。

| 状态文件                           | 格式   | 用途              | 缺失/为空时的默认行为               |
|:------------------------------ |:---- |:--------------- |:------------------------- |
| `state_cool_down.json`         | JSON | 记录冷却池行业及解锁日期    | 初始化为空字典 `{}` (无冷却限制)      |
| `state_factor_weights.json`    | JSON | 记录因子权重及 IC 统计缓存 | 加载 `config.yaml` 中的默认权重配置 |
| `state_portfolio_snapshot.csv` | CSV  | 记录每日持仓快照 (用于绩效) | 视为空持仓 (触发初始建仓逻辑)          |

**原子写入保障**：所有文件写入必须采用 **写临时文件 -> 校验和验证 -> `os.replace` 覆盖** 机制，防止脚本中断导致文件损坏。

### 2.3 数据预处理与鲁棒性

#### 2.3.1 业务逻辑硬阈值校验 (Sanity Check)

在进行统计处理前，必须先通过业务逻辑校验，防止脏数据污染窗口。
| 校验项 | 阈值 | 处理策略 |
| :--- | :--- | :--- |
| **单日涨跌幅** | > 30% (非指数正常波动) | 标记该日数据为 `Invalid`，不参与计算 |
| **价格有效性** | Close <= 0 或 Open <= 0 | 标记该日数据为 `Invalid` |
| **成交量异常** | Volume = 0 且非停牌 | 标记该日数据为 `Invalid` |

#### 2.3.2 缺失值与异常值

| 场景         | 处理策略                                           |
|:---------- |:---------------------------------------------- |
| **原始数据缺失** | 假设数据库原始数据完整，若缺失则标记行业当日 `Invalid`               |
| **指标计算缺失** | 如因历史窗口不足 (如 MA60 需 60 日数据) 导致 NaN，标记 `Invalid` |
| **异常值截断**  | 所有原始特征值进行 1%/99% 分位缩尾 (Winsorization)          |

#### 2.3.3 早期数据不足处理

* **最小窗口阈值**: 60 交易日。
* **降级策略**: 若数据 < 60 日，该行业当日标记为 `Invalid`，不可入选。
* **市场状态计算**: Level 1 指数计算时，仅包含生命周期有效的行业；MA200 计算初期若不足 200 日，标记 Regime 为 `UNKNOWN`，沿用上一交易日状态。

### 2.4 生命周期处理 (防未来函数)

**数据源信任原则**：`dim.vw_asset_cn_csi_sectors` 视图已包含**历史时点 (PIT)** 的行业分类信息（基于个股动态聚合）。

**数据切片规则**:
在调仓日 $t$，对行业 $i$ 的特征计算仅允许使用以下数据子集：
$$ \mathcal{D}_{i,t} = \left\{ (p_s, \text{indicators}_s) \mid s \le t, \ s \in \mathcal{L}_i \right\} $$
其中 $\mathcal{L}_i$ 由数据库视图自动提供，无需额外映射表。若某行业在 `fin.daily_sector_industry_cn` 中无当日数据，视为不可选。

---

## 3. 特征工程详解

### 3.1 标准化与归一化流程 (强制 - 轨道隔离)

为避免量纲差异及权重和不等于 1 的问题，严格执行以下**轨道独立**流程：

1. **异常值截断**: Winsorization (1%/99%)。
2. **轨道内标准化 (关键)**:
   * **趋势轨道**: 直接进行 **横截面百分位排名 (Cross-Sectional Rank)**，映射至 0-100 分。
     $$ S_{\text{trend}} = \text{Rank}_{\text{cross-section}}(S_{\text{raw}}) \times 100 $$
   * **左侧轨道**: 先进行 **时间序列分位 (判断绝对超跌)**，再在候选池内进行 **横截面排名**，映射至 0-100 分。
     $$ S_{\text{left}} = \text{Rank}_{\text{cross-section}}(\text{Percentile}_{\text{time-series}}(S_{\text{raw}})) \times 100 $$
3. **权重收缩**: 逻辑与 ICIR 混合。
   $$ w_{\text{raw}} = \lambda \cdot w_{\text{logic}} + (1 - \lambda) \cdot w_{\text{ICIR}} $$
4. **权重归一化**: 确保**各轨道内**权重和为 1。
   $$ w_{\text{final}} = \frac{w_{\text{raw}}}{\sum w_{\text{raw}}} $$

### 3.2 因子去冗余 (轨道内去相关)

**原则**: 仅在轨道内部进行因子相关性处理，**严禁跨轨道正交化**，以保留风格暴露。

| 步骤         | 操作方法                     | 阈值         |
|:---------- |:------------------------ |:---------- |
| **相关系数计算** | 计算轨道内各因子 60 日滚动相关系数      | -          |
| **聚类分组**   | 若 $                      | \rho(A, B) |
| **簇内合成**   | 保留 ICIR 最高者，或按 ICIR 加权合成 | -          |
| **跨轨道处理**  | **无操作**                  | -          |

### 3.3 趋势轨道特征 (5 类)

| 类别       | 特征符号                                 | 计算公式 (示例)                                              | 经济逻辑            |
|:-------- |:------------------------------------ |:------------------------------------------------------ |:--------------- |
| **动量类**  | $a_3, a_7, a_{14}$                   | $a_3 = \frac{r_3 - r_7}{                               | r_7             |
| **价格结构** | $E_{\text{eff}}, B_{\text{high}}$    | $E_{\text{eff}} = \frac{1}{5}\sum                      | \frac{C-O}{H-L} |
| **技术指标** | $S_{\text{dev}}, S_{\text{ma}}$      | $d = \frac{P - MA60}{MA60}$                            | 均线偏离与排列         |
| **相对强度** | $S_{\text{rel}}$                     | $0.6 \cdot S_{\text{out}} + 0.4 \cdot S_{\text{rank}}$ | 相对父级与市场         |
| **动量质量** | $S_{\text{persist}}, S_{\text{vol}}$ | $\text{VolRatio} = \frac{TO_t}{TO_{\text{avg}}}$       | 量价配合确认          |

**层级权重配置 (经 IC 验证后定稿)**:
| 层级 $\ell$ | $w_1$ (动量) | $w_2$ (结构) | $w_3$ (技术) | $w_4$ (相对强度) | $w_5$ (质量) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Level 4 | 0.30 | 0.15 | 0.10 | 0.35 | 0.10 |
| Level 3 | 0.25 | 0.15 | 0.10 | 0.25 | 0.25 |
| Level 2 | 0.20 | 0.15 | 0.10 | 0.30 | 0.25 |

### 3.4 左侧轨道特征 (5 类)

| 类别        | 特征符号                                    | 计算公式 (示例)                                       | 经济逻辑       |
|:--------- |:--------------------------------------- |:----------------------------------------------- |:---------- |
| **下跌减速**  | $S_{\text{slow}}, S_{\text{shadow}}$    | $S_{\text{shadow}} = \frac{O-L}{H-L} \times 30$ | 下影线支撑 (核心) |
| **超跌类**   | $S_{\text{bias}}, S_{\text{rsi}}$       | $d = \frac{P - MA60}{MA60}$ (负向)                | 乖离率超跌      |
| **资金背离**  | $S_{\text{flow}}, S_{\text{rel}}$       | $P=\min \land \text{MFV} > \text{Avg}$          | 价格新低资金流入   |
| **波动收缩**  | $S_{\text{vol}}, S_{\text{vol-shadow}}$ | $\frac{\sigma_5}{\sigma_{20}} < 0.7$            | 抛压衰减       |
| **成交量萎缩** | $S_{\text{volShrink}}$                  | $\frac{TO_t}{TO_{\text{avg}}} < 0.8$            | 缩量企稳       |

**综合得分 (动态权重)**:
$$ S_{\text{recovery}} = \sum_{k} w_k(\text{Regime}) \cdot S_{k,\text{norm}} $$

### 3.5 市场状态与动态权重 (经 IC 验证)

利用 **Level 1 行业市值加权指数** 定义市场状态，驱动权重切换。**注意：此处指数计算必须调用全局统一函数 `calculate_level1_index()`**。

| 市场状态 (Regime)     | 定义逻辑                          | 左侧权重配置 (已归一化)                                                 |
|:----------------- |:----------------------------- |:------------------------------------------------------------- |
| **BEAR_HIGH_VOL** | 指数 < 0.95×MA200 且 Vol > 80 分位 | shadow:0.37, bias:0.30, slow:0.19, flow:0.12, vol-shadow:0.11 |
| **BEAR_LOW_VOL**  | 指数 < 0.95×MA200 且 Vol ≤ 80 分位 | shadow:0.32, bias:0.26, slow:0.22, flow:0.17, vol-shadow:0.12 |
| **CHOP**          | 0.95×MA200 ≤ 指数 ≤ 1.05×MA200  | shadow:0.26, bias:0.24, slow:0.21, flow:0.23, vol-shadow:0.13 |
| **BULL**          | 指数 > 1.05×MA200               | shadow:0.19, bias:0.12, slow:0.16, flow:0.41, vol-shadow:0.12 |
| **UNKNOWN**       | 数据不足 (早期)                     | 沿用上一状态或默认 CHOP 权重                                             |

### 3.6 因子稳定性监控 (IC Check)

| 监控指标         | 计算方式                       | 处理逻辑         |
|:------------ |:-------------------------- |:------------ |
| **滚动 IC 均值** | 60 日滚动 IC 平均 (Forward 5 日) | 若长期<0，检查因子方向 |
| **IC 胜率**    | IC>0 的交易日比例                | <30% 则降权或停用  |
| **IC 标准差**   | IC 序列波动                    | 过高则因子不稳定     |

**处理逻辑**:

- **回测模式**: 自动降权 50% 若胜率 < 30%。
- **实盘模式**: 更新 `state_factor_weights.json`，若连续 3 个月 IC 胜率 < 30%，自动将权重降为 0.5 倍。

### 3.7 关键衍生指标计算 (统一标准)

为确保回测与实盘结果一致，以下核心指标必须使用统一计算逻辑。**注意：ATR 为衍生指标，非原始字段，需按真实波幅（TR）公式滚动计算**。

**1. 20 日平均真实波幅 (ATR)**

```python
def calculate_atr(high, low, close):
    # 计算真实波幅 TR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # 20 日滚动平均
    return pd.Series(tr).rolling(20).mean()
```

**2. 200 日移动平均线 (MA200)**

```python
def calculate_ma200(close_series):
    return close_series.rolling(200).mean()
```

**3. Level 1 市值加权指数 (单一可信源)**
**注意**：必须通过 JOIN 获取行业层级信息，严禁直接使用假设字段。

```python
def calculate_level1_index(df_fin, df_sector_map):
    """
    df_fin: fin.daily_sector_industry_cn 数据
    df_sector_map: dim.vw_asset_cn_csi_sectors 视图数据
    """
    # 1. 联合查询获取层级信息
    # 假设 sector_industry_id 为关联键
    merged = pd.merge(
        df_fin, 
        df_sector_map[['sector_industry_id', 'csi_sector_level1']].drop_duplicates(), 
        on='sector_industry_id', 
        how='left'
    )

    # 2. 仅选取 Level 1 行业，按总市值加权计算每日指数点位
    level1_data = merged[merged['csi_sector_level1'].notna()] 

    index = level1_data.groupby('trade_date').apply(
        lambda x: np.average(x['front_adj_close'], weights=x['total_market_cap'])
    )
    return index
```

---

## 4. 筛选与组合构建

### 4.1 趋势轨道筛选 (灵活漏斗 + 约束)

**筛选流程**:

1. **Level 2 初选**: 选取综合得分最高的 8 个 Level 2 行业。
2. **Level 3 分组**: 在每个选中的 Level 2 下，选取得分最高的 **最多 2 个** Level 3 行业。
   - **边界处理**: 若某 Level 2 下有效 Level 3 数量 < 2（例如只有 1 个），则**全选该可用项**，不报错、不跳过。
3. **Level 4 终选**: 在每个选中的 Level 3 下，选取得分最高的 1 个 Level 4 行业。

**暴露度约束**:
| 约束项 | 阈值 | 处理方式 |
| :--- | :--- | :--- |
| 单一 Level 1 暴露 | ≤ 20% | 按得分优先级剔除超额部分 |
| 单一 Level 2 暴露 | ≤ 10% | 同上 |
| 总持仓数量 | 15-25 个 | 控制分散度 |

### 4.2 左侧轨道筛选 (独立池)

**筛选流程**:

1. **候选池构建**: 
   $$ \text{Pool}_{\text{left}} = \text{All}_{\text{L4}} - \text{Pool}_{\text{trend}} - \text{Current Holdings} - \text{CoolDown Pool} $$
   (**严格物理隔离**，且排除冷却池行业)
2. **准入过滤**:
   - 中期跌幅: $r_{30} < -10\%$
   - 得分阈值: $S_{\text{recovery}} \ge \theta_{\text{entry}}$ (默认 65 分，0-100 分制)
3. **候选池不足处理**:
   - 若达标行业 < 3 个：
     1. 第一次尝试：阈值降低 5 分 (≥60 分)。
     2. 第二次尝试：阈值再降低 5 分 (≥55 分)。
     3. 若仍不足：**空仓等待**，不强制从趋势池调剂。
4. **最终选择**: 按得分降序，选取前 3 个行业。

### 4.3 权重分配与现金管理

| 场景         | 分配逻辑                              |
|:---------- |:--------------------------------- |
| **初始建仓**   | 趋势池等权重 85%，左侧池等权重 15%             |
| **事件驱动补仓** | 卖出释放的权重 **继承** 给新入选行业，按新行业评分加权分配  |
| **波动率目标**  | 若组合预测波动率 >15%，同比例降低总仓位，**剩余现金计息** |
| **无标的现金**  | 若卖出后无达标候选，资金保留为现金，计息              |

**波动率目标计算 (细化)**:

* **预测模型**: EWMA (指数加权移动平均), $\lambda=0.94$。
* **预测窗口**: 20 日。
* **降仓公式**:
  $$ \text{Target Position} = \min \left( 1.0, \frac{15\%}{\text{Forecasted Vol}} \right) $$
* **现金收益**: 未投资仓位按无风险收益率 ($r_f$, 默认年化 2%) 计入绩效。

**补充权重公式**:
$$ w_i = \frac{\text{Weight}_{\text{released}} \cdot S_i}{\sum S_{\text{new}}} $$

---

## 5. 动态调仓与风控 (事件驱动版)

### 5.1 卖出信号 (任一触发)

| 信号类型         | 触发条件                                                                                                  | 适用轨道      |
|:------------ |:----------------------------------------------------------------------------------------------------- |:--------- |
| **ATR 动态止损** | $\frac{P_{\min} - P_{\text{entry}}}{P_{\text{entry}}} < -1.5 \cdot \frac{ATR_{20}}{P_{\text{entry}}}$ | 双轨        |
| **固定比例止损**   | 回撤 > 8%                                                                                               | 双轨        |
| **持有期到期**    | 持有天数 ≥ 60 天                                                                                           | 左侧优先      |
| **评分跌破**     | $S_t < \theta_{\text{exit}}$ (默认 60 分)                                                                | 双轨        |
| **异常波动**     | 单日波动 > 2.5 * ATR                                                                                      | 双轨 (强制风控) |

### 5.2 补仓机制与执行时序

| 参数         | 设定值          | 说明                   |
|:---------- |:------------ |:-------------------- |
| **信号生成时间** | **T 日 收盘后**  | 确保数据完整，防未来函数         |
| **执行时间**   | **T+1 日 开盘** | 避免当日价格冲击，确保成交        |
| **调仓触发**   | **事件驱动**     | 仅当有卖出信号释放资金时，才触发补仓筛选 |
| **冷却期**    | **10 交易日**   | 卖出行业 10 日内禁止重新入选任一轨道 |
| **补充数量**   | 最多 6 个行业     | 防止过度交易               |

### 5.3 交易成本与滑点 (差异化)

| 模式                   | 成本设定                                              | 说明                             |
|:-------------------- |:------------------------------------------------- |:------------------------------ |
| **Backtest**         | 双边 0.2% + 滑点 0.1%                                 | 固定成本，便于复现                      |
| **Production Daily** | **分档滑点**                                          | 根据行业日均成交额分档                    |
| **滑点分档**             | 成交额>10 亿：0.05% <br> 1 亿 -10 亿：0.1% <br> <1 亿：0.2% | **基于 T-1 日及过去 20 日均值估算**，防未来函数 |

### 5.4 全局冷却池管理 (文件版/内存版)

* **存储介质**: 实盘使用 `state_cool_down.json`，回测使用内存字典。
* **结构**: `{"industry_id": {"sell_date": "2024-05-20", "unlock_date": "2024-06-03"}}`
* **写入**: 当行业 $i$ 在 $T$ 日触发卖出信号，更新内存对象，脚本结束前原子写入文件 (实盘)。
* **读取**: 每日筛选前，加载文件/内存，过滤 `current_date < unlock_date` 的行业。
* **清理**: 每次写入前，自动移除 `unlock_date < current_date` 的过期记录。
* **范围**: 全局生效 (趋势轨道与左侧轨道均不可选)。
* **初始化**: 若文件不存在，视为空池，无行业受冷却限制。

---

## 6. 回测与绩效评估

### 6.1 回测设置

| 参数项      | 设定值                     | 说明                                        |
|:-------- |:----------------------- |:----------------------------------------- |
| **回测区间** | 2018-01-01 至 2023-12-31 | 覆盖牛熊震荡周期                                  |
| **初始资金** | 1,000,000 元             | 虚拟本金                                      |
| **基准指数** | Level 1 行业市值加权指数        | 代表全行业平均表现                                 |
| **行业映射** | **数据库视图自动处理**           | 依赖 `dim.vw_asset_cn_csi_sectors` 的 PIT 信息 |
| **现金收益** | **年化 2%**               | 波动率控仓产生的闲置现金计息                            |
| **调仓模拟** | **事件驱动模拟**              | 仅在卖出信号触发日执行调仓逻辑                           |
| **冷却模拟** | **内存跟踪器**               | **回测模式下启用内存版冷却池，逻辑与实盘完全一致**               |

### 6.2 评估指标

**通用指标**:
| 指标 | 公式 | 说明 |
| :--- | :--- | :--- |
| 年化收益率 | $\mu = \left( \prod_{t=1}^{T} (1 + r_t) \right)^{252/T} - 1$ | 几何平均年化 |
| 夏普比率 | $\text{SR} = (\mu - r_f) / \sigma$, $r_f = 3\%$ | 风险调整后收益 |
| 最大回撤 | $\text{MDD} = \max_{t} \left( \frac{\max_{s \leq t} V_s - V_t}{\max_{s \leq t} V_s} \right)$ | 历史最大回撤 |
| 信息比率 | $(\mu_p - \mu_b) / \sigma_{\text{tracking}}$ | 相对基准超额收益 |
| **换手率** | 年度累计卖出金额 / 平均持仓市值 | **重点监控事件驱动下的换手** |

**策略健康度监控指标**:
| 指标 | 计算公式 | 说明 |
| :--- | :--- | :--- |
| **冷却池命中率** | 被冷却排除候选数 / 总潜在候选数 | 评估冷却机制对选股范围的限制程度 |
| **左侧轨道空仓频率** | 左侧空仓交易日 / 总交易日 | 评估左侧策略的市场适应性 |

**左侧专项指标**:
| 指标 | 合格标准 | 说明 |
| :--- | :--- | :--- |
| 止跌成功率 | > 60% | 入选后 20 日内出现正收益的比例 |
| 最大浮亏 (MAE) | median ≥ -4% | 入选后最低点的平均回撤 |
| 假阳性率 | < 25% | 入选后 20 日收益率 < -5% 的比例 |

**稳定性指标**:

- **分状态绩效**: 牛/熊/震市场下的分别收益
- **因子 IC 稳定性**: 各特征 IC 序列的标准差 (Forward 5 日)
- **参数敏感性**: 权重±10% 波动下的绩效变化

### 6.3 IC 监控反馈机制

* **回测模式**: 自动记录 IC 序列，生成报告。
* **实盘模式**: 更新 `state_factor_weights.json`，若连续 3 个月 IC 胜率 < 30%，自动将权重降为 0.5 倍。

---

## 7. 实施指南 (脚本化专用)

### 7.1 文件原子写入与校验方案

为防止脚本中断导致状态文件损坏，所有 JSON 配置文件写入必须遵循以下逻辑（含校验和验证）：

```python
import json
import os
import tempfile
import hashlib

def calculate_checksum(data_str):
    return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

def atomic_write_json(filepath, data):
    dir_name = os.path.dirname(filepath)
    os.makedirs(dir_name, exist_ok=True)

    # 1. 序列化数据
    data_str = json.dumps(data, ensure_ascii=False, indent=2)
    expected_checksum = calculate_checksum(data_str)

    # 2. 写入临时文件
    fd, tmp_path = tempfile.mkstemp(suffix='.json', dir=dir_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
            tmp.write(data_str)

        # 3. 校验和验证 (防止磁盘 I/O 错误)
        with open(tmp_path, 'r', encoding='utf-8') as tmp:
            actual_checksum = calculate_checksum(tmp.read())

        if expected_checksum != actual_checksum:
            raise IOError("Checksum verification failed during write.")

        # 4. 原子替换原文件
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
```

### 7.2 安全加载与初始化逻辑

首次运行或文件损坏时，安全加载默认值，防止崩溃。

```python
import yaml

def load_state_file(filepath, default_value=None):
    if not os.path.exists(filepath):
        return default_value

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return default_value
            return json.loads(content)
    except Exception:
        return default_value

# 使用示例
cool_down_data = load_state_file('data/state_cool_down.json', default_value={})

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
default_weights = config['scoring']['left_track']['regime_weights']['chop']
weights_data = load_state_file('data/state_factor_weights.json', default_value=default_weights)
```

### 7.3 模式切换逻辑

通过 `config.yaml` 中的 `system.mode` 区分：

* **`backtest`**: 初始化 `CoolDownManager` 为内存字典，**启用冷却逻辑模拟**，不加载/保存文件。
* **`production_daily`**: 初始化 `CoolDownManager` 时加载 `state_cool_down.json`，结束时保存。若文件丢失，初始化为空字典。

### 7.4 数据加载与性能优化

* **按需加载**: 仅加载最近 250 交易日数据用于因子计算，避免全表加载内存溢出。
* **行业过滤**: 仅在 `dim.sector_industry_cn` 中存在的行业才加载行情数据。
* **统一指数计算**: 所有需要 Level 1 指数的地方，必须调用 `calculate_level1_index()` 函数，禁止重复实现。
* 

---

## 8. 权重配置速查表 (config.yaml 参考)

```yaml
system:
  version: "4.3"
  mode: "production_daily"
  data_source:
    pit_enabled: true

state_management:
  files:
    cool_down: "data/state_cool_down.json"
    weights: "data/state_factor_weights.json"
    snapshot: "data/state_portfolio_snapshot.csv"
  atomic_write: true
  checksum_enabled: true

scoring:
  isolation:
    enabled: true
    standardization:
      trend: "cross_sectional_rank"
      left: "time_series_percentile_then_cross_sectional"
    orthogonalization:
      method: "correlation_clustering"
      threshold: 0.7
      scope: "intra_track_only"

  left_track:
    regime_weights:
      bear_high_vol:
        shadow: 0.37
        bias: 0.30
        slow: 0.19
        flow: 0.12
        vol_shadow: 0.11
      bear_low_vol:
        shadow: 0.32
        bias: 0.26
        slow: 0.22
        flow: 0.17
        vol_shadow: 0.12
      chop:
        shadow: 0.26
        bias: 0.24
        slow: 0.21
        flow: 0.23
        vol_shadow: 0.13
      bull:
        shadow: 0.19
        bias: 0.12
        slow: 0.16
        flow: 0.41
        vol_shadow: 0.12
      unknown:
        shadow: 0.26
        bias: 0.24
        slow: 0.21
        flow: 0.23
        vol_shadow: 0.13

  trend_track:
    level_weights:
      level4:
        momentum: 0.30
        structure: 0.15
        technical: 0.10
        relative_strength: 0.35
        quality: 0.10
      level3:
        momentum: 0.25
        structure: 0.15
        technical: 0.10
        relative_strength: 0.25
        quality: 0.25
      level2:
        momentum: 0.20
        structure: 0.15
        technical: 0.10
        relative_strength: 0.30
        quality: 0.25

data_processing:
  sanity_check:
    enabled: true
    max_daily_return: 0.30
    min_price: 0.01
  missing_value:
    strategy: "skip"
  outlier:
    method: "winsorization"
    percentiles: [0.01, 0.99]
  early_
    min_window_days: 60

screening:
  funnel:
    level2_top_n: 8
    level3_top_n: 2
    level4_top_n: 1
  left_track:
    entry_score_threshold: 65
    threshold_decay_step: 5
    max_decay_steps: 2
  cool_down:
    enabled: true
    days: 10
    scope: "global"
    storage: "local_file"

rebalancing:
  type: "event_driven"
  trigger_signals:
    - "stop_loss_atr"
    - "stop_loss_fixed"
    - "hold_period_expire"
    - "score_drop"
  weight_allocation:
    method: "inherit_and_split"

risk_control:
  volatility_target:
    enabled: true
    target_vol: 0.15
    model: "ewma"
    lambda: 0.94
    window: 20
  cash_management:
    idle_cash_yield: 0.02

  transaction_cost:
    backtest:
      commission: 0.001
      slippage: 0.001
    production_daily:
      tiered_slippage: true
      volume_window: 20

deployment:
  execution_timing:
    signal_generation: "T_close_after"
    order_execution: "T+1_open"
  ic_monitoring:
    review_frequency: "monthly"
    auto_downweight_threshold: 0.30
    ic_forward_days: 5
  health_metrics:
    - "cool_down_hit_rate"
    - "left_track_empty_frequency"
```
