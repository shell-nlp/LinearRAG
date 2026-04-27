# LinearRAG

基于论文 [LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora](https://arxiv.org/pdf/2510.10114) 的重构版本。

这次重构重点：

- `LLM` 和 `Embedding` 改为 `LangChain` 接入，配置从 `.env` 读取。
- 向量检索、关键词检索、向量存储统一走 `Elasticsearch`。
- 图能力抽象成 `KnowledgeGraph`，默认实现为 `igraph`，后续可扩展到 `Neo4j` 等后端。
- 项目依赖切换到 `uv` 管理。

## 架构

- `src/langchain_clients.py`
  - `ChatOpenAI` / `OpenAIEmbeddings` 封装。
- `src/document_store.py`
  - 单索引 ES 文档存储，使用 `doc_type` 区分 `passage`、`entity`、`sentence`。
- `src/knowledge_graph.py`
  - 图抽象接口和 `igraph` 默认实现。
- `src/LinearRAG.py`
  - 保留论文里的主流程：`seed entity retrieval -> sentence expansion -> passage weighting -> personalized PageRank`。
- `src/elastic_utils.py`
  - 在尽量兼容原接口的前提下，增强了索引创建、过滤检索、批量 upsert 等能力。

## 环境准备

1. 安装依赖

```bash
uv sync
```

2. 下载 spaCy 模型

```bash
uv run python -m spacy download en_core_web_trf
```

3. 配置环境变量

```bash
cp .env.example .env
```

默认配置已经按当前项目需求填好：

- `OPENAI_BASE_URL=https://miyun.archermind.com/v1`
- `OPENAI_API_KEY=123`
- `ES_URL=http://127.0.0.1:9200`
- `ES_USER=elastic`
- `ES_PWD=elastic@2024`

4. 准备数据集

目录结构保持不变：

```text
dataset/
  <dataset_name>/
    chunks.json
    questions.json
```

## 运行

```bash
uv run python run.py --dataset_name 2wikimultihop
```

常用参数：

- `--llm_model`
- `--embedding_model`
- `--spacy_model`
- `--max_workers`
- `--max_iterations`
- `--iteration_threshold`
- `--passage_ratio`
- `--top_k_sentence`
- `--skip_eval`

也可以直接用脚本：

```bash
bash scripts/run.sh
```

## Elasticsearch 设计

默认每个数据集使用一个 index：

- index 名格式：`linearrag-<dataset_name>`
- 同一个 index 中的文档通过 `doc_type` 字段区分：
  - `passage`
  - `entity`
  - `sentence`

不再为不同向量类型拆多个 index。

## 输出

- 检索缓存：`import/<dataset_name>/`
- 图导出：`import/<dataset_name>/LinearRAG.graphml`
- 运行结果：`results/<dataset_name>/<timestamp>/`

## 说明

- 当前图后端只实现了 `igraph`。
- `Neo4j` 等其他知识图谱实现的扩展点已经通过 `KnowledgeGraph` 抽象预留。
- 如果 ES 中不存在目标 index，程序会自动按 embedding 维度创建。
