"""
retriever.py - RAG搜索交互系统（支持命令行参数和交互模式）
"""
import os
import sys
import argparse
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, Any, List  # <-- 添加这一行

# 添加项目根目录到路径 # 确保这一行在 Orin 上能准确找到目录
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class MuseumRetriever:
    def __init__(self):
        print("=" * 60)
        print(" 初始化RAG搜索系统")
        print("=" * 60)

        # 路径配置
        self.data_dir = project_root / "data"
        self.model_path = project_root / "models" / "bge-small-zh-v1.5"
        self.chroma_path = self.data_dir / "chroma_db_local_model"

        # 检查文件
        if not self.chroma_path.exists():
            print(f"❌ 向量数据库不存在: {self.chroma_path}")
            print("💡 请先运行 'python backend/rag/ingest.py' 构建数据库")
            sys.exit(1)

        if not self.model_path.exists():
            print(f"❌ 模型不存在: {self.model_path}")
            sys.exit(1)

        print(f"🤖 模型路径: {self.model_path}")
        print(f"🗄️  向量库: {self.chroma_path}")

        # 初始化ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))

        # 使用本地BGE模型
        print("🤖 加载本地BGE模型...")
        try:
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=str(self.model_path)
            )
            print("✅ 本地模型加载成功")
        except Exception as e:
            print(f"❌ 本地模型加载失败: {e}")
            sys.exit(1)

        # 加载集合
        self._load_collection()

    def _load_collection(self):
        try:
            self.collection = self.client.get_collection(
                name="museum_local",
                embedding_function=self.embedding_fn
            )
            count = self.collection.count()
            print(f"✅ 加载向量数据库成功 ({count} 条记录)")
            return count
        except Exception as e:
            print(f"❌ 加载集合失败: {e}")
            print("💡 请确保已运行 'python backend/rag/ingest.py' 构建数据库")
            sys.exit(1)

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        核心检索接口 - 返回知识文本（给A同学调用）

        参数：
            query: 查询问题
            top_k: 返回结果数量

        返回：
            str: 格式化后的知识文本
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            if not results['documents'][0]:
                return "未找到相关作品"

            # 格式化返回文本
            formatted_results = []
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]

                # 构建单个作品的信息
                work_info = []
                work_info.append(f"作品名称：《{meta.get('作品名称', '')}》")
                work_info.append(f"设计作者：{meta.get('设计作者', '')}")
                work_info.append(f"指导老师：{meta.get('指导老师', '')}")
                work_info.append(f"类别标签：{meta.get('类别标签', '')}")
                work_info.append(f"呈现形式：{meta.get('呈现形式', '')}")

                # 添加详细描述（只取前500字符）
                doc_lines = doc.split('\n')
                for line in doc_lines:
                    if any(keyword in line for keyword in ['作品描述', '设计动机', '灵感来源', '设计目的', '技术特点']):
                        if len(line) > 100:
                            work_info.append(f"{line[:100]}...")
                        else:
                            work_info.append(line)

                work_info.append(f"所属展区：{meta.get('所属展区', '')}")
                formatted_results.append('\n'.join(work_info))

            return '\n\n' + '='*60 + '\n\n'.join(formatted_results) + '\n\n' + '='*60

        except Exception as e:
            return f"检索失败: {e}"

    def get_stats(self) -> dict:
        """返回知识库统计信息"""
        try:
            return {
                "total_documents": self.collection.count(),
                "embedding_model": "bge-small-zh-v1.5",
                "status": "ready",
                "database_path": str(self.chroma_path)
            }
        except:
            return {
                "total_documents": 0,
                "embedding_model": "unknown",
                "status": "error",
                "database_path": str(self.chroma_path)
            }

    def search(self, query: str, top_k: int = 5, show_full: bool = True):
        """交互式搜索（带格式显示）"""
        print(f"\n🔍 搜索: '{query}'")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            if not results['documents'][0]:
                print("📭 未找到相关作品")
                return []

            print(f"✅ 找到 {len(results['documents'][0])} 个结果:")

            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                dist = results['distances'][0][i]
                similarity = 1 - dist

                print(f"\n【{i+1}】相似度: {similarity:.3f}")
                print(f"📌 作品: 《{meta.get('作品名称', '')}》")
                print(f"👤 作者: {meta.get('设计作者', '')}")
                print(f"👨‍🏫 指导: {meta.get('指导老师', '')}")
                print(f"🏷️ 类别: {meta.get('类别标签', '')}")
                print(f"🎨 形式: {meta.get('呈现形式', '')}")
                print(f"📅 时间: {meta.get('创作时间', '')}")
                print(f"📍 展区: {meta.get('所属展区', '')}")

                if show_full:
                    print("\n📄 完整内容:")
                    print("-" * 60)
                    print(doc)
                    print("-" * 60)
                else:
                    print("\n📋 关键信息:")
                    lines = doc.split('\n')
                    displayed = 0
                    for line in lines:
                        if any(keyword in line for keyword in ['设计动机', '灵感来源', '设计目的', '技术特点', '设计理念']):
                            if len(line) > 100:
                                print(f"   {line[:100]}...")
                            else:
                                print(f"   {line}")
                            displayed += 1
                            if displayed >= 3:
                                break

            return results

        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []


# 添加在 retriever.py 文件的合适位置（可以在 MuseumRetriever 类后面）

class Retriever:
    """
    为A同学提供的统一接口类
    接口规范：retrieve(query: str, top_k=3) -> str
    """

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        """
        初始化RAG检索器
        persist_dir: Chroma数据库路径
        """
        # 复用现有的 MuseumRetriever
        self.exhibition_retriever = MuseumRetriever()

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        核心检索接口 - A同学会调用这个

        参数：
            query: 用户问题
            top_k: 返回几个相关文档

        返回：
            str: 检索到的知识文本，用\n\n分隔
        """
        # 调用现有的搜索功能
        results = self.exhibition_retriever.search(query, top_k)

        # 格式化为字符串（按文档要求的格式）
        texts = []
        for result in results:
            content = result.get("content", "").strip()
            if content:
                texts.append(content)

        # 用两个换行符分隔每个文档
        return "\n\n".join(texts)

    def get_stats(self) -> Dict[str, Any]:
        """返回知识库统计（可选，用于调试）"""
        stats = self.exhibition_retriever.get_collection_statistics()
        return {
            "total_documents": stats.get("total_documents", 0),
            "embedding_model": "all-MiniLM-L6-v2",
            "status": "ready"
        }

__all__ = ['MuseumRetriever', 'Retriever']

def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='RAG检索系统 - 博物馆作品知识库')
    parser.add_argument('--query', '-q', type=str, help='直接查询的内容，如："这是什么作品？"')
    parser.add_argument('--top_k', '-k', type=int, default=3, help='返回结果数量，默认3')
    parser.add_argument('--simple', '-s', action='store_true', help='简洁模式（不显示完整内容）')
    parser.add_argument('--stats', action='store_true', help='只显示统计信息')
    parser.add_argument('--version', '-v', action='store_true', help='显示版本信息')

    args = parser.parse_args()

    # 显示版本信息
    if args.version:
        print("RAG检索系统 v1.0")
        print("支持中文语义搜索的博物馆作品知识库")
        print("使用本地BGE-small-zh模型 + ChromaDB")
        return

    print("=" * 60)
    print("🎯 博物馆RAG检索系统")
    print("=" * 60)

    # 创建检索器实例
    try:
        retriever = MuseumRetriever()
    except SystemExit:
        return  # 初始化失败，直接退出

    # 如果指定了 --stats，只显示统计
    if args.stats:
        stats = retriever.get_stats()
        print(f"📊 知识库统计信息:")
        print(f"   - 总作品数: {stats['total_documents']}")
        print(f"   - 向量模型: {stats['embedding_model']}")
        print(f"   - 状态: {stats['status']}")
        print(f"   - 数据库: {stats['database_path']}")
        return

    # 如果指定了 --query，直接查询并退出
    if args.query:
        print(f"\n🔍 查询: '{args.query}'")
        retriever.search(args.query, top_k=args.top_k, show_full=not args.simple)
        return

    # 否则进入交互模式
    print("\n" + "=" * 60)
    print(f"📊 数据库统计:")
    stats = retriever.get_stats()
    print(f"   - 总作品数: {stats['total_documents']}")
    print("=" * 60)

    # 交互模式
    print("\n💬 交互模式 (输入 'exit' 退出, 'simple' 切换简洁模式)")
    print("💡 搜索提示:")
    print("  1. 按技术搜索: '磁悬浮技术', '虚幻引擎5', 'RFID'")
    print("  2. 按主题搜索: '传统文化', '环境保护', '儿童心理'")
    print("  3. 按理念搜索: '场景驱动', '多模态交互', '视觉叙事'")
    print("  4. 按人员搜索: '王心妍', '温馨', '林哲轩'")
    print("  5. 输入 'simple' 切换简洁/完整显示")
    print("-" * 60)

    show_full = True

    while True:
        try:
            query = input("\n🔎 请输入查询: ").strip()

            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 再见！")
                break
            elif query.lower() in ['simple', '简洁', '简', 's']:
                show_full = not show_full
                mode_text = "显示完整内容" if show_full else "显示关键信息"
                print(f"📄 切换到: {mode_text}")
                continue

            if not query:
                continue

            # 执行搜索
            retriever.search(query, show_full=show_full)

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()