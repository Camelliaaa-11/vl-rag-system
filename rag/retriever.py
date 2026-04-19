"""
retriever.py - RAG 搜索系统

当前主路径：
- 优先使用 Qwen/Qwen3-Embedding-0.6B 本地模型
- 在服务启动时全局预加载嵌入模型
- 首次启动时自动把旧 collection 中的文档重建到新的 Qwen collection
- 如果本地模型或依赖不可用，则退回轻量关键词检索
"""
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class GlobalQwenEmbeddingModel:
    """全局单例嵌入模型，避免重复加载。"""

    _model = None
    _model_path = None
    _load_error = None

    @classmethod
    def load(cls, model_path: Path):
        if cls._model is not None and cls._model_path == str(model_path):
            return cls._model

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            cls._load_error = exc
            raise RuntimeError(f"sentence-transformers 不可用: {exc}") from exc

        try:
            cls._model = SentenceTransformer(str(model_path), trust_remote_code=True)
            cls._model_path = str(model_path)
            cls._load_error = None
            print(f"Qwen3 Embedding 模型加载成功: {model_path}")
            return cls._model
        except Exception as exc:
            cls._load_error = exc
            raise RuntimeError(f"Qwen3 Embedding 模型加载失败: {exc}") from exc

    @classmethod
    def encode_queries(cls, texts: List[str]) -> List[List[float]]:
        if cls._model is None:
            raise RuntimeError("Qwen3 Embedding 模型尚未加载")
        embeddings = cls._model.encode(texts, prompt_name="query", normalize_embeddings=True)
        return embeddings.tolist()

    @classmethod
    def encode_documents(cls, texts: List[str]) -> List[List[float]]:
        if cls._model is None:
            raise RuntimeError("Qwen3 Embedding 模型尚未加载")
        embeddings = cls._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


class MuseumRetriever:
    SOURCE_COLLECTION_NAME = "museum_local"
    TARGET_COLLECTION_NAME = "museum_qwen3_embedding"

    def __init__(self):
        print("=" * 60)
        print(" 初始化 RAG 搜索系统")
        print("=" * 60)

        self.fallback_mode = False
        self.client = None
        self.source_collection = None
        self.collection = None

        self.data_dir = project_root / "data"
        self.model_path = project_root / "models" / "Qwen3-Embedding-0.6B"
        self.legacy_model_path = project_root / "models" / "bge-small-zh-v1.5"
        self.chroma_path = self.data_dir / "chroma_db_local_model"

        if not self.chroma_path.exists():
            print(f"向量数据库不存在: {self.chroma_path}")
            print("请先运行 'python rag/ingest.py' 构建数据库")
            sys.exit(1)

        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._load_source_collection()
        self._init_embedding_backend()
        self._prepare_target_collection()

    def _load_source_collection(self):
        try:
            self.source_collection = self.client.get_collection(name=self.SOURCE_COLLECTION_NAME)
            count = self.source_collection.count()
            print(f"加载源向量数据库成功 ({count} 条记录)")
        except Exception as exc:
            print(f"加载源集合失败: {exc}")
            print("请确保已运行 'python rag/ingest.py' 构建数据库")
            sys.exit(1)

    def _init_embedding_backend(self):
        print(f"Qwen3 模型路径: {self.model_path}")
        print(f"向量库路径: {self.chroma_path}")

        if not self.model_path.exists():
            self.fallback_mode = True
            print("Qwen3 Embedding 模型目录不存在，切换到轻量关键词检索模式")
            return

        try:
            GlobalQwenEmbeddingModel.load(self.model_path)
            self.fallback_mode = False
        except Exception as exc:
            self.fallback_mode = True
            print(f"Qwen3 Embedding 加载失败，切换到轻量关键词检索模式: {exc}")

    def _prepare_target_collection(self):
        if self.fallback_mode:
            self.collection = self.source_collection
            return

        source_count = self.source_collection.count()
        rebuild = False

        try:
            target = self.client.get_collection(name=self.TARGET_COLLECTION_NAME)
            target_count = target.count()
            if target_count != source_count:
                rebuild = True
            self.collection = target
        except Exception:
            rebuild = True

        if rebuild:
            self._rebuild_qwen_collection()

        if self.collection is None:
            self.collection = self.client.get_collection(name=self.TARGET_COLLECTION_NAME)

        print(f"Qwen 检索集合就绪 ({self.collection.count()} 条记录)")

    def _rebuild_qwen_collection(self):
        print("开始构建 Qwen3 Embedding 检索集合...")
        try:
            self.client.delete_collection(self.TARGET_COLLECTION_NAME)
        except Exception:
            pass

        target = self.client.create_collection(
            name=self.TARGET_COLLECTION_NAME,
            metadata={"description": "museum knowledge base with Qwen3 embeddings"},
        )

        payload = self.source_collection.get(include=["documents", "metadatas"])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        ids = self.source_collection.get().get("ids", [])

        batch_size = 16
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            batch_docs = documents[start:end]
            batch_metas = metadatas[start:end]
            batch_ids = ids[start:end]
            batch_embeddings = GlobalQwenEmbeddingModel.encode_documents(batch_docs)
            target.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
                embeddings=batch_embeddings,
            )
            print(f"已重建 {end}/{len(documents)} 条记录")

        self.collection = target
        print("Qwen3 Embedding 检索集合构建完成")

    def _tokenize(self, text: str) -> List[str]:
        normalized = (text or "").strip().lower()
        if not normalized:
            return []

        synonym_map = {
            "展品": "作品",
            "这个展品": "这个作品",
            "那件展品": "那件作品",
            "创作者": "设计作者",
            "作者是谁": "设计作者",
            "作者": "设计作者",
            "哪个区": "所属展区",
            "展区": "所属展区",
        }
        for source, target in synonym_map.items():
            normalized = normalized.replace(source, target)

        parts = re.split(r"[\s，。！？、；：,.!?:()（）\[\]【】《》\"'“”]+", normalized)
        tokens: List[str] = []
        stopwords = {
            "请", "给我", "一个", "一下", "这件", "这个", "那个", "是不是", "有个",
            "是什么", "什么", "是谁", "有", "的", "吗", "呢", "吧", "一下子", "我想",
            "介绍", "可以", "帮我", "告诉我",
        }

        def add_token(value: str):
            value = value.strip()
            if not value or value in stopwords or len(value) == 1:
                return
            if value not in tokens:
                tokens.append(value)

        for part in parts:
            add_token(part)
            compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", part)
            if len(compact) >= 2:
                for size in (2, 3, 4):
                    if len(compact) >= size:
                        for idx in range(len(compact) - size + 1):
                            add_token(compact[idx: idx + size])

        return tokens

    def _fallback_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        data = self.source_collection.get(include=["documents", "metadatas"])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        query_tokens = self._tokenize(query)

        scored_results: List[Dict[str, Any]] = []
        for idx, doc in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            searchable_text = " ".join(
                [
                    str(doc or ""),
                    str(meta.get("作品名称", "")),
                    str(meta.get("设计作者", "")),
                    str(meta.get("指导老师", "")),
                    str(meta.get("类别标签", "")),
                    str(meta.get("呈现形式", "")),
                    str(meta.get("所属展区", "")),
                ]
            ).lower()

            score = 0
            for token in query_tokens:
                if token in searchable_text:
                    score += 5 if len(token) >= 4 else 2

            if score > 0:
                scored_results.append({"document": doc, "metadata": meta, "score": score})

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results[:top_k]

    def _semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = GlobalQwenEmbeddingModel.encode_queries([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        records: List[Dict[str, Any]] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for idx, doc in enumerate(documents):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else None
            records.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "score": distance,
                }
            )
        return records

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "未找到相关作品"

        formatted_results = []
        for item in results:
            doc = item["document"]
            meta = item["metadata"]

            work_info = [
                f"作品名称：《{meta.get('作品名称', '')}》",
                f"设计作者：{meta.get('设计作者', '')}",
                f"指导老师：{meta.get('指导老师', '')}",
                f"类别标签：{meta.get('类别标签', '')}",
                f"呈现形式：{meta.get('呈现形式', '')}",
            ]

            for line in doc.split("\n"):
                if any(keyword in line for keyword in ["作品描述", "设计动机", "灵感来源", "设计目的", "技术特点"]):
                    work_info.append(f"{line[:100]}..." if len(line) > 100 else line)

            work_info.append(f"所属展区：{meta.get('所属展区', '')}")
            formatted_results.append("\n".join(work_info))

        return "\n\n" + "=" * 60 + "\n\n".join(formatted_results) + "\n\n" + "=" * 60

    def retrieve(self, query: str, top_k: int = 3) -> str:
        try:
            if self.fallback_mode:
                return self._format_results(self._fallback_search(query, top_k=top_k))
            return self._format_results(self._semantic_search(query, top_k=top_k))
        except Exception as exc:
            return f"检索失败: {exc}"

    def get_stats(self) -> dict:
        try:
            return {
                "total_documents": self.collection.count() if self.collection else 0,
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B" if not self.fallback_mode else "keyword-fallback",
                "status": "ready",
                "database_path": str(self.chroma_path),
                "model_path": str(self.model_path),
                "fallback_mode": self.fallback_mode,
            }
        except Exception:
            return {
                "total_documents": 0,
                "embedding_model": "unknown",
                "status": "error",
                "database_path": str(self.chroma_path),
                "model_path": str(self.model_path),
                "fallback_mode": self.fallback_mode,
            }

    def search(self, query: str, top_k: int = 5, show_full: bool = True):
        print(f"\n🔍 搜索: '{query}'")
        try:
            results = self._fallback_search(query, top_k) if self.fallback_mode else self._semantic_search(query, top_k)
            if not results:
                print("📭 未找到相关作品")
                return []

            print(f"✅ 找到 {len(results)} 个结果:")
            for idx, item in enumerate(results, start=1):
                doc = item["document"]
                meta = item["metadata"]
                score = item["score"]
                print(f"\n【{idx}】得分/距离: {score}")
                print(f"📌 作品: 《{meta.get('作品名称', '')}》")
                print(f"👤 作者: {meta.get('设计作者', '')}")
                print(f"👨‍🏫 指导: {meta.get('指导老师', '')}")
                print(f"🏷️ 类别: {meta.get('类别标签', '')}")
                print(f"🎨 形式: {meta.get('呈现形式', '')}")
                print(f"📍 展区: {meta.get('所属展区', '')}")
                if show_full:
                    print("\n📄 完整内容:")
                    print("-" * 60)
                    print(doc)
                    print("-" * 60)
            return results
        except Exception as exc:
            print(f"❌ 搜索失败: {exc}")
            return []


class Retriever:
    """兼容接口。"""

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        del persist_dir
        self.exhibition_retriever = MuseumRetriever()

    def retrieve(self, query: str, top_k: int = 3) -> str:
        return self.exhibition_retriever.retrieve(query, top_k)

    def get_stats(self) -> Dict[str, Any]:
        return self.exhibition_retriever.get_stats()


__all__ = ["MuseumRetriever", "Retriever"]


def main():
    parser = argparse.ArgumentParser(description="RAG 检索系统 - 博物馆作品知识库")
    parser.add_argument("--query", "-q", type=str, help="直接查询的内容")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="返回结果数量，默认 3")
    parser.add_argument("--simple", "-s", action="store_true", help="简洁模式")
    parser.add_argument("--stats", action="store_true", help="只显示统计信息")
    args = parser.parse_args()

    retriever = MuseumRetriever()

    if args.stats:
        print(retriever.get_stats())
        return

    if args.query:
        retriever.search(args.query, top_k=args.top_k, show_full=not args.simple)
        return

    print("\n💬 交互模式 (输入 'exit' 退出)")
    while True:
        query = input("\n🔎 请输入查询: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            break
        if not query:
            continue
        retriever.search(query, top_k=args.top_k, show_full=not args.simple)


if __name__ == "__main__":
    main()
