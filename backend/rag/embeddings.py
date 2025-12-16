"""
向量化嵌入管理模块 - 使用 transformers 版本
"""
import os
import sys
from typing import List, Dict, Any
import numpy as np
import torch
import chromadb
from chromadb.config import Settings


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化嵌入模型管理器 - 使用 transformers
        """
        # 定义本地模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.model_dir = os.path.join(project_root, "models", "all-MiniLM-L6-v2")

        print(f"🔧 初始化嵌入模型 (transformers)")
        print(f"   模型路径: {self.model_dir}")

        # 检查模型文件
        required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(self.model_dir, file)):
                raise FileNotFoundError(f"模型文件缺失: {file}")

        try:
            # 设置环境变量
            os.environ['TRANSFORMERS_OFFLINE'] = "1"
            os.environ['HF_HUB_OFFLINE'] = "1"

            # 导入 transformers
            from transformers import AutoTokenizer, AutoModel

            print(f"   加载 tokenizer 和 model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModel.from_pretrained(self.model_dir)

            # 设置为评估模式
            self.model.eval()

            # 获取模型维度（从config.json）
            import json
            config_path = os.path.join(self.model_dir, "config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.embedding_dim = config.get("hidden_size", 384)

            print(f"✅ 嵌入模型加载成功")
            print(f"   向量维度: {self.embedding_dim}")
            print(f"   模型类型: {type(self.model).__name__}")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本向量化 - 使用均值池化
        """
        if not texts:
            return []

        print(f"🔢 向量化 {len(texts)} 个文本...")

        try:
            # 分批处理，避免内存问题
            batch_size = 32
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )

                # 推理
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # 均值池化获得句子向量
                # attention_mask 用于忽略padding
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state

                # 扩展 attention_mask 以匹配嵌入维度
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

                # 加权平均
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

                # 归一化（余弦相似度需要）
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                all_embeddings.append(batch_embeddings.numpy())

            # 合并所有批次
            embeddings_array = np.vstack(all_embeddings)
            embeddings_list = embeddings_array.tolist()

            print(f"✅ 向量化完成")
            print(f"  向量维度: {len(embeddings_list[0]) if embeddings_list else 0}")
            print(f"  向量数量: {len(embeddings_list)}")

            return embeddings_list

        except Exception as e:
            print(f"❌ 向量化失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        单个查询向量化
        """
        return self.embed_texts([query])[0]

    def create_chroma_client(self, persist_directory: str = "data/chroma_db"):
        """
        创建ChromaDB客户端
        """
        os.makedirs(persist_directory, exist_ok=True)

        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )

        return client

    def get_or_create_collection(self,
                                 collection_name: str = "exhibition_docs",
                                 persist_directory: str = "data/chroma_db"):
        """
        获取或创建向量集合
        """
        client = self.create_chroma_client(persist_directory)

        try:
            collection = client.get_collection(name=collection_name)
            print(f"📂 加载现有集合: {collection_name}")
            print(f"  文档数量: {collection.count()}")

        except Exception:
            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "description": "艺术与科技展览作品数据库",
                    "created_by": "RAG System",
                    "embedding_model": "all-MiniLM-L6-v2 (transformers)",
                    "embedding_dim": self.embedding_dim
                }
            )
            print(f"📂 创建新集合: {collection_name}")

        return collection

    def get_collection_info(self, collection_name: str = "exhibition_docs") -> Dict[str, Any]:
        """
        获取集合信息
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            info = {
                "collection_name": collection.name,
                "document_count": collection.count(),
                "metadata": collection.metadata,
                "embedding_dim": self.embedding_dim,
                "status": "active"
            }

            return info

        except Exception as e:
            return {
                "collection_name": collection_name,
                "error": str(e),
                "status": "error"
            }


# 全局单例实例
_embedding_manager = None


def get_embedding_manager(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingManager:
    """
    获取嵌入管理器单例
    """
    global _embedding_manager

    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(model_name)

    return _embedding_manager


if __name__ == "__main__":
    # 模块测试
    print("🧪 embeddings.py 模块测试 (transformers版)")
    print("=" * 50)

    try:
        manager = get_embedding_manager()

        test_texts = [
            "这是一个测试文本，用于验证向量化功能。",
            "艺术与科技展览作品数据库",
            "数字文娱设计，互动装置，创新技术"
        ]

        embeddings = manager.embed_texts(test_texts)

        print(f"✅ 测试通过")
        print(f"  测试文本数量: {len(test_texts)}")
        print(f"  生成向量数量: {len(embeddings)}")
        print(f"  向量维度: {len(embeddings[0]) if embeddings else 0}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
