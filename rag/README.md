
---


# 展品知识检索系统 (RAG) - 完整数据链路版

本系统是一套基于混合检索（向量 + BM25）及深度语义重排序（Reranking）的知识检索架构，旨在为人形机器人或其他智能终端提供精准、高效的展品知识检索服务。

## 1. 核心架构职责

| 文件/路径 | 核心职责 |
| :--- | :--- |
| `rag/excel_to_json.py` | **数据 ETL 层**：将原始 Excel 清洗、标准化为 `standard.json`。 |
| `rag/ingest_from_json.py` | **索引构建层**：调用 `Qwen3-Embedding` 将 JSON 数据向量化并存入 ChromaDB。 |
| `rag/retriever_v2_mix_Reranking.py` | **检索核心引擎**：集成了混合召回（向量+BM25）与重排序逻辑。 |
| `rag/test_batch_mixreranking.py` | **测试基准**：评估检索质量、相关度及召回/重排耗时指标。 |
| `models/` | **权重仓库**：存放 `Qwen3-Embedding` 及 `BGE-Reranker` 模型权重。 |



## 2. 环境依赖

请在项目根目录下安装必要的环境依赖：

```bash
pip install -r requirements.txt
```

*注意：依赖包含 `pandas`, `openpyxl` (用于 Excel 处理), `chromadb` (向量库), `sentence-transformers` (嵌入模型) 等。*

## 3. 快速部署流程 (Data Pipeline)

如果你的数据发生了变动，请按照以下顺序执行，完成从数据到索引的闭环：

### Step 1: 清洗数据
将原始 Excel 转为系统要求的标准 JSON 格式：
```bash
python -m rag.excel_to_json --input data/raw/data.xlsx --output data/processed/standard.json
```

### Step 2: 构建索引
将 JSON 内容向量化并入库（会自动生成 `data/chroma_db_qwen3` 文件夹）：
```bash
python -m rag.ingest_from_json
```

### Step 3: 运行验证
执行单次查询或批量测试，确保系统正常运行：
```bash
# 单次查询
python -m rag.retriever_v2_mix_Reranking -q "未来出行概念汽车的作者是谁"

# 批量性能测试
python -m rag.test_batch_mixreranking
```

## 4. 核心接口说明 (MuseumRetriever)

在集成开发时，直接通过 `retriever_v2_mix_Reranking.py` 中的 `MuseumRetriever` 类调用：

- **`hybrid_query(query, top_k=3)`**
  - 功能：完整链路检索，包含向量召回、BM25 召回、重排序。
  - 返回：`(results, metrics)`，metrics 包含 `recall`、`rerank` 和 `total` 耗时。
- **`vector_search(query, collection_name, n_results=5)`**
  - 功能：针对特定集合进行原子向量召回。
- **`rerank(query, candidates)`**
  - 功能：使用重排序模型进行深度语义精排。
- **`get_stats()`**
  - 功能：返回当前知识库规模、引擎类型等系统状态信息。
- **`rebuild_index()`**
  - 功能：刷新内存中的 BM25 倒排索引，适用于数据增量更新场景。


