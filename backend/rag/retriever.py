"""
展览数据检索模块
"""
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

try:
    from backend.rag.embeddings import get_embedding_manager

    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    print(f"❌ 无法导入嵌入模块: {e}")
    EMBEDDINGS_AVAILABLE = False


class ExhibitionRetriever:
    """展览数据检索器"""

    def __init__(self, collection_name: str = "exhibition_docs"):
        """
        初始化检索器

        Args:
            collection_name: 集合名称
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("嵌入模块不可用，请检查依赖安装")

        print(f"🔍 初始化展览检索器...")

        # 获取嵌入管理器
        self.embedder = get_embedding_manager()

        # 获取向量集合
        self.collection = self.embedder.get_or_create_collection(collection_name)

        print(f"✅ 检索器初始化完成")
        print(f"  集合名称: {self.collection.name}")
        print(f"  文档数量: {self.collection.count()}")

        # 缓存统计信息
        self._stats_cache = None
        self._stats_time = None

    def search(self, query: str, top_k: int = 5,
               filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        通用搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_metadata: 元数据过滤器

        Returns:
            检索结果列表
        """
        print(f"\n🔍 搜索查询: '{query}'")
        print(f"  返回数量: {top_k}")

        try:
            # 向量化查询
            query_embedding = self.embedder.embed_query(query)

            # 构建查询参数
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k * 2, 20),  # 获取更多用于过滤
                "include": ["documents", "metadatas", "distances"]
            }

            # 添加过滤器
            if filter_metadata:
                query_params["where"] = filter_metadata
                print(f"  过滤条件: {filter_metadata}")

            # 执行查询
            results = self.collection.query(**query_params)

            # 处理结果
            processed_results = self._process_results(results, query, top_k)

            print(f"✅ 找到 {len(processed_results)} 个相关结果")

            return processed_results

        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def search_by_zone(self, zone: str, query: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        按展区搜索

        Args:
            zone: 展区名称（如"艺术与科技展区130509T-X"）
            query: 附加查询文本
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        # 构建完整查询
        full_query = f"{zone}展区 {query}".strip()

        # 设置过滤器
        filter_metadata = {"zone": {"$eq": zone}}

        return self.search(full_query, top_k, filter_metadata)

    def search_by_category(self, category: str, query: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        full_query = f"{category} {query}".strip()

        # ChromaDB不支持$contains，使用其他方法
        # 方案A：直接搜索，不设过滤（让向量相似度自己匹配）
        return self.search(full_query, top_k)

        # 或者方案B：如果需要严格过滤，可以在获取结果后再过滤
        # results = self.search(full_query, top_k * 2)  # 获取更多结果
        # filtered = [r for r in results if category in r.get("category", "") or
        #             category in r.get("sub_category", "")]
        # return filtered[:top_k]

    def search_by_author(self, author: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query = f"作者{author}的作品"

        # 直接搜索，不在数据库层过滤
        return self.search(query, top_k)

    # 在 retriever.py 中修改
    def search_by_technique(self, technique: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        按技术特点搜索（新格式）

        Args:
            technique: 技术关键词（如"RFID"、"3D建模"、"磁悬浮"）
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        # 直接搜索，因为技术特点现在在详细描述中
        query = f"{technique}技术"

        return self.search(query, top_k)

    def search_by_item_name(self, item_name: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        按作品名称搜索

        Args:
            item_name: 作品名称
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        query = f"作品{item_name}"

        filter_metadata = {
            "item_name": {"$contains": item_name}
        }

        return self.search(query, top_k, filter_metadata)

    def get_similar_items(self, item_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        获取相似作品

        Args:
            item_name: 作品名称
            top_k: 返回结果数量

        Returns:
            相似作品列表
        """
        # 首先找到该作品
        target_results = self.search_by_item_name(item_name, top_k=1)

        if not target_results:
            print(f"⚠️  未找到作品: {item_name}")
            return []

        # 使用该作品的文档进行相似度搜索
        target_doc = target_results[0].get("content", "")

        return self.search(target_doc, top_k)

    def get_collection_statistics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        获取集合统计信息

        Args:
            force_refresh: 强制刷新缓存

        Returns:
            统计信息字典
        """
        # 检查缓存
        if (not force_refresh and self._stats_cache and self._stats_time and
                (datetime.now() - self._stats_time).seconds < 300):  # 5分钟缓存
            return self._stats_cache

        try:
            # 获取一些样本文档进行分析
            sample_results = self.collection.query(
                query_embeddings=[[0] * 384],  # 虚拟查询
                n_results=min(100, self.collection.count()),
                include=["metadatas"]
            )

            stats = {
                "collection_name": self.collection.name,
                "total_documents": self.collection.count(),
                "last_updated": datetime.now().isoformat(),
                "document_types": {},
                "categories": {},
                "zones": set(),
                "authors": set(),
                "sample_size": len(sample_results.get("metadatas", [[]])[0])
            }

            # 分析元数据
            if sample_results.get("metadatas"):
                for metadata_list in sample_results["metadatas"]:
                    for metadata in metadata_list:
                        # 文档类型统计
                        doc_type = metadata.get("type", "unknown")
                        stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1

                        # 类别统计
                        category = metadata.get("category", "")
                        if category:
                            stats["categories"][category] = stats["categories"].get(category, 0) + 1

                        # 展区统计
                        zone = metadata.get("zone", "")
                        if zone:
                            stats["zones"].add(zone)

                        # 作者统计
                        authors = metadata.get("authors", "")
                        if authors:
                            stats["authors"].add(authors)

            # 转换集合为列表
            stats["zones"] = list(stats["zones"])
            stats["authors"] = list(stats["authors"])

            # 缓存结果
            self._stats_cache = stats
            self._stats_time = datetime.now()

            return stats

        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {
                "collection_name": self.collection.name,
                "total_documents": self.collection.count(),
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }

    def _process_results(self, results: Dict[str, Any], query: str,
                         top_k: int) -> List[Dict[str, Any]]:
        """
        处理原始检索结果

        Args:
            results: 原始检索结果
            query: 原始查询
            top_k: 目标结果数量

        Returns:
            处理后的结果列表
        """
        processed_results = []

        if not results or not results.get("documents") or not results["documents"][0]:
            return processed_results

        # 提取数据
        documents = results["documents"][0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # 处理每个结果
        for i in range(min(len(documents), len(metadatas), len(distances))):
            doc_content = documents[i]
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 0.0

            # 计算相似度分数（余弦距离转相似度）
            similarity_score = 1.0 - distance if distance <= 1.0 else 0.0

            # 构建结果对象
            result = {
                "rank": i + 1,
                "content": doc_content,
                "metadata": metadata,
                "similarity": similarity_score,
                "distance": distance,
                "relevance": self._calculate_relevance(doc_content, query, similarity_score),
                "type": metadata.get("type", "unknown"),
                "item_name": metadata.get("item_name", "未知作品"),
                "category": metadata.get("category", "未知"),
                "zone": metadata.get("zone", "未知")
            }

            processed_results.append(result)

        # 按相关性排序并截取
        processed_results.sort(key=lambda x: x["relevance"], reverse=True)
        processed_results = processed_results[:top_k]

        # 为结果添加解释
        for i, result in enumerate(processed_results):
            result["explanation"] = self._generate_explanation(result, query, i + 1)

        return processed_results

    def _calculate_relevance(self, content: str, query: str,
                             similarity: float) -> float:
        """
        计算综合相关性分数

        Args:
            content: 文档内容
            query: 查询文本
            similarity: 向量相似度

        Returns:
            综合相关性分数（0-1）
        """
        # 基础相似度分数（50%权重）
        base_score = similarity * 0.5

        # 关键词匹配分数（30%权重）
        query_keywords = set(query.lower().split())
        content_lower = content.lower()

        keyword_score = 0.0
        for keyword in query_keywords:
            if len(keyword) > 2 and keyword in content_lower:
                keyword_score += 1.0

        if query_keywords:
            keyword_score = min(keyword_score / len(query_keywords) * 0.3, 0.3)

        # 文档类型分数（20%权重）
        # 优先显示基本信息文档
        type_score = 0.2  # 默认分数

        # 组合分数
        total_score = base_score + keyword_score + type_score

        return min(total_score, 1.0)

    def _generate_explanation(self, result: Dict[str, Any],
                              query: str, rank: int) -> str:
        """
        生成结果解释

        Args:
            result: 结果对象
            query: 查询文本
            rank: 排名

        Returns:
            解释文本
        """
        item_name = result.get("item_name", "作品")
        category = result.get("category", "")
        similarity = result.get("similarity", 0.0)

        explanations = []

        # 添加排名信息
        explanations.append(f"排名第{rank}位")

        # 添加相似度信息
        if similarity > 0.8:
            explanations.append("高度相关")
        elif similarity > 0.6:
            explanations.append("比较相关")
        elif similarity > 0.4:
            explanations.append("一般相关")
        else:
            explanations.append("弱相关")

        # 添加类别信息
        if category:
            explanations.append(f"属于{category}类别")

        # 添加匹配说明
        query_lower = query.lower()
        content_lower = result.get("content", "").lower()

        if any(keyword in content_lower for keyword in query_lower.split() if len(keyword) > 2):
            explanations.append("包含查询关键词")

        return "，".join(explanations)

    def format_results_for_display(self, results: List[Dict[str, Any]],
                                   query: str = "") -> str:
        """
        格式化结果用于显示

        Args:
            results: 结果列表
            query: 原始查询

        Returns:
            格式化后的字符串
        """
        if not results:
            return f"未找到与'{query}'相关的结果。"

        output = []
        output.append(f"📊 搜索结果（查询：'{query}'）")
        output.append("=" * 60)

        for i, result in enumerate(results):
            output.append(f"\n[{i + 1}] {result.get('item_name', '未知作品')}")
            output.append(f"   相似度: {result.get('similarity', 0):.3f} - {result.get('explanation', '')}")
            output.append(f"   类别: {result.get('category', '未知')}")
            output.append(f"   展区: {result.get('zone', '未知')}")

            # 显示内容摘要
            content = result.get("content", "")
            if len(content) > 150:
                content = content[:150] + "..."
            output.append(f"   内容: {content}")

        output.append(f"\n总计找到 {len(results)} 个相关结果")

        return "\n".join(output)

    def test_retrieval(self):
        """
        测试检索功能
        """
        print("🧪 检索功能测试")
        print("=" * 60)

        # 测试用例
        test_cases = [
            ("通用搜索", "search", "互动装置"),
            ("按展区搜索", "search_by_zone", "艺术与科技展区130509T-X", "数字文娱"),
            ("按类别搜索", "search_by_category", "数字文娱设计"),
            ("按作者搜索", "search_by_author", "郭海媚"),
            ("按技术搜索", "search_by_technique", "RFID"),
        ]

        for test_name, method, *args in test_cases:
            print(f"\n🔍 {test_name}: {args}")

            try:
                if method == "search":
                    results = self.search(args[0], top_k=3)
                elif method == "search_by_zone":
                    results = self.search_by_zone(args[0], args[1] if len(args) > 1 else "", top_k=2)
                elif method == "search_by_category":
                    results = self.search_by_category(args[0], top_k=2)
                elif method == "search_by_author":
                    results = self.search_by_author(args[0], top_k=2)
                elif method == "search_by_technique":
                    results = self.search_by_technique(args[0], top_k=2)
                else:
                    results = []

                if results:
                    print(f"✅ 找到 {len(results)} 个结果")
                    for i, result in enumerate(results[:2]):  # 显示前2个
                        print(f"  [{i + 1}] {result.get('item_name', '未知')}")
                        print(f"      相似度: {result.get('similarity', 0):.3f}")
                else:
                    print("  ⚠️  未找到结果")

            except Exception as e:
                print(f"  ❌ 测试失败: {e}")

        # 显示统计信息
        print(f"\n{'=' * 60}")
        print("📊 数据库统计信息:")
        stats = self.get_collection_statistics()

        print(f"集合名称: {stats.get('collection_name', '未知')}")
        print(f"总文档数: {stats.get('total_documents', 0)}")

        if 'document_types' in stats:
            print(f"\n文档类型分布:")
            for doc_type, count in stats['document_types'].items():
                print(f"  {doc_type}: {count}")

        if 'categories' in stats:
            print(f"\n作品类别分布:")
            for category, count in stats['categories'].items():
                print(f"  {category}: {count}")


# 添加在 retriever.py 文件的合适位置（可以在 ExhibitionRetriever 类后面）

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
        # 复用现有的 ExhibitionRetriever
        self.exhibition_retriever = ExhibitionRetriever("exhibition_docs")

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

__all__ = ['ExhibitionRetriever', 'Retriever']

def main():
    """
    主函数：命令行接口
    """
    import argparse

    parser = argparse.ArgumentParser(description="展览数据检索系统")
    parser.add_argument("--query", type=str, help="搜索查询")
    parser.add_argument("--zone", type=str, help="按展区搜索")
    parser.add_argument("--category", type=str, help="按类别搜索")
    parser.add_argument("--author", type=str, help="按作者搜索")
    parser.add_argument("--technique", type=str, help="按技术搜索")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")

    args = parser.parse_args()

    try:
        # 初始化检索器
        retriever = ExhibitionRetriever()

        # 执行相应操作
        if args.test:
            retriever.test_retrieval()

        elif args.stats:
            stats = retriever.get_collection_statistics()
            print("📊 数据库统计信息:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        elif args.query:
            results = retriever.search(args.query, args.top_k)
            print(retriever.format_results_for_display(results, args.query))

        elif args.zone:
            query = args.query if args.query else ""
            results = retriever.search_by_zone(args.zone, query, args.top_k)
            print(retriever.format_results_for_display(results, f"{args.zone} {query}"))

        elif args.category:
            results = retriever.search_by_category(args.category, args.top_k)
            print(retriever.format_results_for_display(results, args.category))

        elif args.author:
            results = retriever.search_by_author(args.author, args.top_k)
            print(retriever.format_results_for_display(results, f"作者{args.author}"))

        elif args.technique:
            results = retriever.search_by_technique(args.technique, args.top_k)
            print(retriever.format_results_for_display(results, f"{args.technique}技术"))

        else:
            # 交互模式
            print("🎨 展览数据检索系统（交互模式）")
            print("=" * 60)

            while True:
                print(f"\n当前集合: {retriever.collection.name} ({retriever.collection.count()}文档)")
                print("可用命令:")
                print("  search <查询>      - 通用搜索")
                print("  zone <展区> <查询> - 按展区搜索")
                print("  category <类别>    - 按类别搜索")
                print("  author <作者>      - 按作者搜索")
                print("  technique <技术>   - 按技术搜索")
                print("  stats             - 显示统计")
                print("  test              - 运行测试")
                print("  exit              - 退出")

                command = input("\n请输入命令: ").strip()

                if command.lower() in ['exit', 'quit', 'q']:
                    print("👋 再见！")
                    break

                elif command.lower() == 'stats':
                    stats = retriever.get_collection_statistics()
                    print("📊 统计信息:")
                    for key, value in stats.items():
                        if isinstance(value, (dict, list, set)):
                            print(f"  {key}:")
                            if isinstance(value, dict):
                                for k, v in value.items():
                                    print(f"    {k}: {v}")
                            else:
                                for item in list(value)[:10]:  # 只显示前10个
                                    print(f"    {item}")
                        else:
                            print(f"  {key}: {value}")

                elif command.lower() == 'test':
                    retriever.test_retrieval()

                elif command.startswith('search '):
                    query = command[7:].strip()
                    if query:
                        results = retriever.search(query, 5)
                        print(retriever.format_results_for_display(results, query))
                    else:
                        print("❌ 请输入查询内容")

                elif command.startswith('zone '):
                    parts = command[5:].strip().split(' ', 1)
                    if len(parts) >= 1:
                        zone = parts[0]
                        query = parts[1] if len(parts) > 1 else ""
                        results = retriever.search_by_zone(zone, query, 5)
                        print(retriever.format_results_for_display(results, f"{zone} {query}"))
                    else:
                        print("❌ 请输入展区名称")

                elif command.startswith('category '):
                    category = command[9:].strip()
                    if category:
                        results = retriever.search_by_category(category, 5)
                        print(retriever.format_results_for_display(results, category))
                    else:
                        print("❌ 请输入类别名称")

                elif command.startswith('author '):
                    author = command[7:].strip()
                    if author:
                        results = retriever.search_by_author(author, 5)
                        print(retriever.format_results_for_display(results, f"作者{author}"))
                    else:
                        print("❌ 请输入作者姓名")

                elif command.startswith('technique '):
                    technique = command[10:].strip()
                    if technique:
                        results = retriever.search_by_technique(technique, 5)
                        print(retriever.format_results_for_display(results, f"{technique}技术"))
                    else:
                        print("❌ 请输入技术关键词")

                else:
                    print("❌ 未知命令，请输入有效命令")

    except Exception as e:
        print(f"❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
