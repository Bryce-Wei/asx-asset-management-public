# 数据放置与格式

我将四份原始输入放在 `data/` 目录中，随仓库公开；将便于检查程序流程的模拟样本放在 `data/sample/`。价格文件为 Yahoo Finance 的复权收盘价（含分红）；新闻标题通过纽约时报 API 获取；客户资料为模拟数据，不含真实客户信息；宏观指标由课程提供。滚动回测另用 ASX 200 指数 ETF（STW）的价格作为比较基准，用澳洲储备银行（RBA）的现金利率作为无风险利率。运行生成的结果不提交。

## 完整分析输入

四份原始文件位于 `data/`，除价格外保留原文件名及工作簿布局。程序默认从 `data/raw/` 读取，因此运行完整分析或回测时用 `--data-dir data` 指定该目录；把文件复制到 `data/raw/` 则不需要这个参数。各文件的内容与读取约定如下：

| 文件 | 内容与读取约定 | 用途 |
|---|---|---|
| `ASX200top10_yahoo.csv` | 第一列为 `date`，其余为 10 只股票的 Yahoo Finance 复权收盘价（Adj Close，已计入现金分红和拆股），2015-03-30 至 2020-07-31 共 1,354 个 ASX 交易日（截止日与新闻数据的覆盖期对齐）；由 `scripts/fetch_yahoo_prices.py` 生成，同名 `.json` 记录下载时间、参数、分红与拆股明细及文件哈希 | 价格清洗、收益计算、蒙特卡洛模拟与组合优化 |
| `Client_Details.xlsx` | 优先读取 `Data` 工作表，包含唯一的 `client_ID`、`risk_profile`、`age_group` 和数值型持仓特征 | 客户特征标准化与 KMeans 分群 |
| `Economic_Indicators.xlsx` | 读取第一张工作表中的 `Monthly Indicators` 区域；列为月份、行为指标，季度指标不参与当前流程 | 月度宏观指标对齐、OLS 回归与调整量计算 |
| `news_dump.json` | 新闻记录列表，包含 `Equity`、`Source`、`Date/Time`、`Headline` | VADER 标题情绪评分与按股票汇总的权重调整 |
| `ASX200_STW_yahoo.csv` | 第一列为 `date`，第二列 `STW` 为 ASX 200 指数 ETF 的 Yahoo Finance 复权收盘价，交易日与股票价格相同；用 `python scripts/fetch_yahoo_prices.py --tickers STW --output data/ASX200_STW_yahoo.csv` 生成，同名 `.json` 为下载记录 | 只在滚动回测中使用，作为买入持有的比较基准 |
| `RBA_cash_rate.csv` | 第一列 `effective_date` 为生效日，第二列 `cash_rate_target` 为现金利率目标（年利率，百分数）；每次议息会议一行，包括维持不变的决定，1990-01-23 至 2026-09-30 共 402 行。用 `python scripts/fetch_rba_cash_rate.py` 从 RBA 网站重新下载，同名 `.json` 记录来源网址、下载时间与文件哈希 | 只在滚动回测中使用：每个交易日取当日已生效的利率作为无风险利率，计算夏普比率；没有这个文件时改用 `backtest_risk_free_rate` 设定的固定利率 |

重新下载价格：`python -m pip install -r scripts/requirements.txt`，再运行 `python scripts/fetch_yahoo_prices.py`。Yahoo 会随新的分红回溯调整整段历史，因此不同时间下载的价格水平可能不同，但相邻日期的收益率不变。早期版本使用的 Bloomberg 工作簿 `ASX200top10.xlsx`（`PX_LAST`，不含分红，并把 38 个 ASX 休市日按前一日价格补齐）不再随仓库提供；持有该文件时，可用 `load_data.load_prices` 读取，对照早期结果。

股票代码在载入时统一为资产标签，并用于对齐客户持仓、新闻和最终权重。股票日期与客户编号必须唯一；价格必须为正数，客户特征必须为完整的有限数值。加载器不会改写输入文件。

价格中的缺失值保留为缺失，不以零值代替。客户字段不接受未经处理的缺失记录。宏观数据保留缺失观测，分析阶段按所属月份对齐并向前填充；这一处理不代表指标在当月交易时已发布，具体限制见 [方法与限制](../docs/methodology.md)。

新闻按照输入顺序读取，`Date/Time` 保留为文本。当前情绪模块把每只股票的前 5 条记录视为近期新闻，程序不会自动按时间排序，因此我在整理输入时需要确保每只股票的记录由新到旧排列。每只待配置股票都需要至少一条有效新闻；不足 5 条时使用已有记录。滚动回测则会解析 `Date/Time`（例如 `22 Jul '20 10:05 AM`），按时间排序并只使用信号日收盘前发布的新闻，不依赖输入顺序。

接入其他数据源时，需要同时核对工作表、表头、资产标签、字段单位和日期含义，并相应调整 `asset_management/load_data.py`。NLP 使用的 `vader_lexicon` 是 NLTK 的外部语言资源，首次运行前需安装：`python -m nltk.downloader vader_lexicon`。

## 模拟演示输入

`sample/prices.csv` 第一列为 `date`，其余列为虚构资产价格；`sample/allocations.csv` 第一列为 `ticker`，其余列为基础权重、情绪调整、宏观调整和最终权重。字段及样本含义见 [sample/README.md](sample/README.md)。

`python main.py --mode demo` 只读取模拟样本，用于检查收益计算、权重校验和报告输出。完整分析和滚动回测分别由 `python main.py --data-dir data` 和 `python main.py --mode backtest --data-dir data` 启动，需要上述四份原始输入。

## 结果目录

完整模式的结果表写入 `outputs/tables/`，图像写入 `outputs/figures/`；演示和回测结果分别写入 `outputs/demo/`、`outputs/backtest/`。这些结果由 `.gitignore` 排除，保留在本地。`data/raw/` 同样被 `.gitignore` 排除，适合存放不打算公开的本地数据。
