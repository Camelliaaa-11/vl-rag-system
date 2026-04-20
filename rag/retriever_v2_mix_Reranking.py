import sys
import argparse
import jieba
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# 路径管理
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

class QwenEmbeddingFunction:
    def __init__(self, model_path):
        self.model = SentenceTransformer(str(model_path), device="cpu")
    
    def name(self):
        return "qwen3-embedding"
        
    # 给 ChromaDB 调用，用于文档（列表）向量化
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input).tolist()
    
    # 给 ChromaDB 调用，专门处理单个查询字符串
    def embed_query(self, input: str) -> List[float]:
        # 必须返回 List[float] (即一维向量)
        return self.model.encode(input).tolist()
    
    # 给 ChromaDB 调用，处理文档列表 (通常和__call__逻辑一致)
    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input).tolist()

class MuseumRetriever:
    def __init__(self):
        # 1. 先定义所有必要的路径和变量
        self.model_path = project_root / "models" / "qwen3-embedding"
        self.reranker_path = project_root / "models" / "bge-reranker-base"
        self.chroma_path = project_root / "data" / "chroma_db_qwen3"
        self.collection_name = "works" 
        
        # 2. 然后加载模型
        embedding_id = str(self.model_path) if self.model_path.exists() else "Qwen/Qwen3-Embedding-0.6B"
        if not self.model_path.exists():
            print(f"⚠️ 本地模型未找到，将尝试从 HuggingFace 自动下载: {embedding_id}")
        else:
            print(f"✅ 已加载本地模型: {embedding_id}")
        reranker_id = str(self.reranker_path) if self.reranker_path.exists() else "BAAI/bge-reranker-base"
        if not self.reranker_path.exists():
            print(f"⚠️ 本地 Reranker 模型未找到，尝试从 HuggingFace 下载: {reranker_id}")
        else:
            print(f"✅ 已加载本地 Reranker 模型: {reranker_id}")

        print(f"🚀 正在加载 Embedding : {embedding_id}")
        self.qwen_func = QwenEmbeddingFunction(embedding_id)
        
        print(f"🔄 正在加载重排序: {reranker_id}")
        self.reranker = CrossEncoder(reranker_id, max_length=512)
        
        # 3. 最后初始化数据库引擎
        self._init_vector_engine()
        self._init_bm25_engine()

    # --- 1. 原子召回 (支持指定集合) ---
    def vector_search(self, query: str, collection_name: str = "works", n_results: int = 5) -> List[Dict]:
        # 允许灵活切换集合
        target_collection = self.client.get_collection(name=collection_name, embedding_function=self.qwen_func)
        res = target_collection.query(query_texts=[query], n_results=n_results)
        return [{"content": res['documents'][0][i], "metadata": res['metadatas'][0][i]} 
                for i in range(len(res['ids'][0]))]

    # --- 2. 统计接口 (更详细) ---
    def get_stats(self) -> Dict:
        # 获取当前集合的文档总数
        count = self.collection.count()
        return {
            "total_docs": count, 
            "collection": self.collection_name,
            "engine": "Qwen3-Embedding + BM25"
        }

    # --- 3. 索引维护 ---
    def rebuild_index(self):
        # 建议这里直接调用你的 VectorBuilder，或者重新载入 BM25 索引
        print("🔄 正在触发索引重建流程...")
        self._init_bm25_engine()
        print("✅ 索引已刷新。")

    def bm25_search(self, query: str, n_results: int = 5) -> List[Dict]:
        tokenized_query = jieba.lcut(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_n_idx = np.argsort(scores)[::-1][:n_results]
        return [{"content": self.all_docs[idx], "metadata": self.all_metadatas[idx]} for idx in top_n_idx]

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        pairs = [(query, cand['content']) for cand in candidates]
        scores = self.reranker.predict(pairs)
        for i, cand in enumerate(candidates):
            cand['score'] = float(scores[i])
        return sorted(candidates, key=lambda x: x['score'], reverse=True)

    def fuse_knowledge(self, chunks: List[Dict]) -> str:
        if not chunks: return "（未找到相关信息）"
        blocks = [f"【相关度 {round(c['score']*100, 1)}%】\n{c['content']}" for c in chunks]
        return "\n\n".join(blocks)

    def hybrid_query(self, query: str, top_k: int = 2) -> tuple:
        """
        执行混合检索流程：向量召回 + BM25 召回 -> 候选集融合 -> BGE重排 -> Top-K 截断。
        
        Args:
            query (str): 用户输入的查询指令。
            top_k (int): 返回给大模型参考的最相关结果数量，默认取前3。
            
        Returns:
            tuple: (results, metrics) 
                   results: 包含内容和 metadata 的列表。
                   metrics: 包含 recall/rerank/total 的耗时指标。
        """
        start_total = time.time()
        t0 = time.time()
        vec_cands = self.vector_search(query)
        bm25_cands = self.bm25_search(query)
        
        seen = set()
        candidates = []
        for c in vec_cands + bm25_cands:
            if c['content'] not in seen:
                candidates.append(c)
                seen.add(c['content'])
        t_recall = time.time() - t0
        
        t1 = time.time()
        candidates = candidates[:5]
        final_results = self.rerank(query, candidates)[:top_k]
        t_rerank = time.time() - t1
        
        return final_results, {"recall": t_recall, "rerank": t_rerank, "total": time.time() - start_total}

    def _init_vector_engine(self):
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, 
            embedding_function=self.qwen_func
        )

    def _init_bm25_engine(self):
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.all_docs, self.all_metadatas = all_data['documents'], all_data['metadatas']
        self.bm25 = BM25Okapi([jieba.lcut(doc) for doc in self.all_docs])
        print(f"📊 BM25 引擎初始化完成，索引 {len(self.all_docs)} 条数据。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', type=str, required=True)
    args = parser.parse_args()
    retriever = MuseumRetriever()
    results, metrics = retriever.hybrid_query(args.query)
    print(f"\n🔍 查询: {args.query} (耗时: {metrics['total']:.3f}s)\n" + "="*50)
    print(retriever.fuse_knowledge(results))