"""
展览数据RAG入库主程序
"""
import os
import sys
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple
import uuid
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 导入本地模块
# 在 ingest.py 中修改导入部分
try:
    from backend.rag.embeddings import get_embedding_manager
    # 使用修改后的新格式加载器
    from backend.rag.excel_loader import load_complex_exhibition_excel as load_exhibition_excel_files
    EXHIBITION_LOADER_AVAILABLE = True
    EMBEDDINGS_AVAILABLE = True  # 添加这行
except ImportError as e:
    print(f"⚠️  模块导入警告: {e}")
    EXHIBITION_LOADER_AVAILABLE = False
    EMBEDDINGS_AVAILABLE = False  # 添加这行

def load_documents(data_dir: str = "data/raw_docs") -> List[Document]:
    """
    加载所有文档

    Args:
        data_dir: 数据目录路径

    Returns:
        文档对象列表
    """
    documents = []

    print("=" * 70)
    print("🎨 艺术与科技展览数据RAG入库系统")
    print("=" * 70)
    print(f"📁 数据目录: {os.path.abspath(data_dir)}")

    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在")
        print(f"创建目录: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        print(f"✅ 目录已创建，请将Excel文件放入此目录")
        return documents

    # 检查目录是否为空
    all_files = os.listdir(data_dir)
    if not all_files:
        print(f"⚠️  数据目录为空")
        print(f"请将Excel文件放入: {data_dir}")
        print(f"文件要求:")
        print(f"  - 格式: .xlsx 或 .xls")
        print(f"  - 内容: 包含展区、作品名称、描述等信息")
        print(f"  - 建议: 包含多个sheet（工业设计、环境设计、艺术与科技）")
        return documents

    print(f"📋 发现 {len(all_files)} 个文件")

    # 1. 优先加载展览Excel数据
    if EXHIBITION_LOADER_AVAILABLE:
        print(f"\n📊 处理展览Excel数据...")
        exhibition_docs = load_exhibition_excel_files(data_dir)

        if exhibition_docs:
            documents.extend(exhibition_docs)
            print(f"✅ 展览数据: {len(exhibition_docs)} 个文档片段")
        else:
            print(f"⚠️  未找到展览数据，尝试其他格式...")
    else:
        print(f"❌ 展览加载器不可用，跳过Excel处理")

    # 2. 加载其他格式文档（如果需要）
    if not documents:
        print(f"\n📄 尝试加载其他格式文档...")
        other_docs = _load_other_formats(data_dir)

        if other_docs:
            documents.extend(other_docs)
            print(f"✅ 其他格式: {len(other_docs)} 个文档片段")
        else:
            print(f"⚠️  未找到其他格式文档")

    # 3. 如果没有数据，创建测试数据
    if not documents:
        print(f"\n🧪 创建测试数据...")
        test_docs = _create_test_documents()
        documents.extend(test_docs)
        print(f"✅ 测试数据: {len(test_docs)} 个文档片段")
        print(f"⚠️  注意: 这是测试数据，请放入真实的Excel文件")

    # 4. 创建系统摘要文档
    if documents:
        print(f"\n📝 创建系统摘要...")
        summary_docs = _create_system_summary(documents, data_dir)
        documents.extend(summary_docs)
        print(f"✅ 系统摘要: {len(summary_docs)} 个文档")

    print(f"\n📈 文档加载统计:")
    print(f"  总文档数: {len(documents)}")

    # 文档类型统计
    doc_types = {}
    categories = {}

    for doc in documents:
        doc_type = doc.metadata.get("type", "unknown")
        category = doc.metadata.get("category", "未知")

        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        categories[category] = categories.get(category, 0) + 1

    print(f"  文档类型: {doc_types}")
    print(f"  作品类别: {categories}")

    return documents

def _load_other_formats(data_dir: str) -> List[Document]:
    """
    加载其他格式文档（PDF、TXT、MD）

    Args:
        data_dir: 数据目录

    Returns:
        文档列表
    """
    documents = []

    try:
        from langchain_community.document_loaders import (
            DirectoryLoader, PyPDFLoader, TextLoader
        )

        # PDF文件
        pdf_pattern = os.path.join(data_dir, "**/*.pdf")
        if glob.glob(pdf_pattern, recursive=True):
            print(f"  📕 加载PDF文件...")
            pdf_loader = DirectoryLoader(
                data_dir,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            print(f"    ✅ PDF文档: {len(pdf_docs)} 个")

        # 文本文件
        txt_pattern = os.path.join(data_dir, "**/*.txt")
        if glob.glob(txt_pattern, recursive=True):
            print(f"  📝 加载文本文件...")
            txt_loader = DirectoryLoader(
                data_dir,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            print(f"    ✅ 文本文档: {len(txt_docs)} 个")

        # Markdown文件
        md_pattern = os.path.join(data_dir, "**/*.md")
        if glob.glob(md_pattern, recursive=True):
            print(f"  📋 加载Markdown文件...")
            md_loader = DirectoryLoader(
                data_dir,
                glob="**/*.md",
                loader_cls=TextLoader,
                show_progress=True
            )
            md_docs = md_loader.load()
            documents.extend(md_docs)
            print(f"    ✅ Markdown文档: {len(md_docs)} 个")

    except ImportError:
        print(f"  ⚠️  langchain_community未安装，跳过其他格式")
    except Exception as e:
        print(f"  ❌ 其他格式加载失败: {e}")

    return documents

def _create_test_documents() -> List[Document]:
    """
    创建测试文档

    Returns:
        测试文档列表
    """
    test_docs = [
        Document(
            page_content="""
【作品基本信息】

作品名称：Adaptive Helix 仿生蠕动机械设计
展区位置：艺术与科技展区130509T-X - X01
作品类别：艺术与科技 / 展示艺术与技术
呈现形式：文字 + 图片

设计作者：郭海媚
指导老师：倪思慧
创作时间：2025年

【作品简介】
基于理查德・道金斯《自私的基因》理论的跨学科互动装置，通过仿生蠕动机械模拟蚯蚓在热带雨林、沙漠和雪地三种环境中的行为。
""",
            metadata={
                "type": "basic_info",
                "category": "艺术与科技",
                "item_name": "Adaptive Helix 仿生蠕动机械设计",
                "source": "test_data"
            }
        ),
        Document(
            page_content="""
【作品基本信息】

作品名称：译文交互界面设计
展区位置：艺术与科技展区130509T-X - X02
作品类别：艺术与科技 / 数字文娱设计
呈现形式：文字 + 图片

设计作者：钟燕营
指导老师：王心妍
创作时间：2025年

【作品简介】
以中国传统纹样作为科普内容的互动网站，包含七大功能模块，为用户提供传统纹样现代化应用的路径。
""",
            metadata={
                "type": "basic_info",
                "category": "艺术与科技",
                "item_name": "译文交互界面设计",
                "source": "test_data"
            }
        )
    ]

    return test_docs

def _create_system_summary(documents: List[Document], data_dir: str) -> List[Document]:
    """
    创建系统摘要文档

    Args:
        documents: 所有文档
        data_dir: 数据目录

    Returns:
        摘要文档列表
    """
    summary_docs = []

    try:
        # 1. 系统信息摘要
        total_docs = len(documents)

        # 统计信息
        doc_types = {}
        categories = {}
        sources = set()

        for doc in documents:
            doc_type = doc.metadata.get("type", "unknown")
            category = doc.metadata.get("category", "未知")
            source = doc.metadata.get("source", "未知")

            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            categories[category] = categories.get(category, 0) + 1
            sources.add(os.path.basename(str(source)))

        # 系统摘要文档
        sys_content = f"""
【系统信息摘要】

数据入库时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据目录：{os.path.abspath(data_dir)}

【数据统计】
文档总数：{total_docs} 个片段
数据源文件：{len(sources)} 个文件

【文档类型分布】
"""

        for doc_type, count in doc_types.items():
            percentage = (count / total_docs * 100) if total_docs > 0 else 0
            sys_content += f"- {doc_type}: {count}个 ({percentage:.1f}%)\n"

        sys_content += f"""
【作品类别分布】
"""

        for category, count in categories.items():
            if category != "未知":
                percentage = (count / total_docs * 100) if total_docs > 0 else 0
                sys_content += f"- {category}: {count}个 ({percentage:.1f}%)\n"

        sys_content += f"""
【系统说明】
此数据库包含艺术与科技展览作品的详细信息，支持按展区、类别、作者、技术等维度检索。
"""

        sys_doc = Document(
            page_content=sys_content.strip(),
            metadata={
                "source": "ingest.py",
                "type": "system_summary",
                "created_at": datetime.now().isoformat(),
                "total_documents": total_docs,
                "data_dir": data_dir
            }
        )
        summary_docs.append(sys_doc)

        # 2. 使用指南文档
        guide_content = f"""
【展览数据库使用指南】

📌 查询示例：
1. 按展区查询：搜索"艺术与科技展区130509T-X"
2. 按类别查询：搜索"数字文娱设计"或"艺术与科技"
3. 按作者查询：搜索"郭海媚"或"设计作者"
4. 按技术查询：搜索"RFID"、"互动装置"、"3D建模"
5. 按作品查询：搜索"Adaptive Helix"或"译文交互"

📌 检索技巧：
- 使用具体关键词获得更精确结果
- 可以组合查询：如"艺术与科技 互动装置"
- 系统支持自然语言查询

📌 数据内容：
- 作品基本信息（名称、作者、时间等）
- 详细设计描述（理念、技术、效果等）
- 图片信息（文件路径、说明等）

📌 系统特性：
- 支持多sheet Excel文件处理
- 智能文档切分和向量化
- 基于语义的相似度检索
"""

        guide_doc = Document(
            page_content=guide_content.strip(),
            metadata={
                "source": "ingest.py",
                "type": "user_guide",
                "version": "1.0",
                "created_at": datetime.now().isoformat()
            }
        )
        summary_docs.append(guide_doc)

    except Exception as e:
        print(f"  ❌ 创建摘要失败: {e}")

    return summary_docs

def split_documents(documents: List[Document]) -> List[Document]:
    """
    智能文档切分

    Args:
        documents: 输入文档列表

    Returns:
        切分后的文档列表
    """
    if not documents:
        return []

    print(f"\n✂️  文档智能切分")
    print(f"输入文档: {len(documents)} 个")

    chunks = []

    # 分离不同类型的文档
    basic_info_docs = []
    detailed_info_docs = []
    image_info_docs = []
    concept_docs = []
    summary_docs = []
    other_docs = []

    for doc in documents:
        doc_type = doc.metadata.get("type", "unknown")

        if doc_type == "basic_info":
            basic_info_docs.append(doc)
        elif doc_type == "detailed_info":
            detailed_info_docs.append(doc)
        elif doc_type == "image_info":
            image_info_docs.append(doc)
        elif doc_type == "design_concept":
            concept_docs.append(doc)
        elif doc_type in ["system_summary", "user_guide"]:
            summary_docs.append(doc)
        else:
            other_docs.append(doc)

    print(f"📋 文档类型统计:")
    print(f"  基本信息: {len(basic_info_docs)}")
    print(f"  详细描述: {len(detailed_info_docs)}")
    print(f"  图片信息: {len(image_info_docs)}")
    print(f"  设计理念: {len(concept_docs)}")
    print(f"  系统摘要: {len(summary_docs)}")
    print(f"  其他文档: {len(other_docs)}")

    # 1. 基本信息文档：不切分（通常较短）
    chunks.extend(basic_info_docs)
    print(f"✅ 基本信息文档: 保持原样")

    # 2. 图片信息文档：不切分
    chunks.extend(image_info_docs)
    print(f"✅ 图片信息文档: 保持原样")

    # 3. 详细描述文档：需要切分（可能较长）
    if detailed_info_docs:
        detailed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )
        detailed_chunks = detailed_splitter.split_documents(detailed_info_docs)
        chunks.extend(detailed_chunks)
        print(f"✅ 详细描述文档: {len(detailed_info_docs)} → {len(detailed_chunks)}")

    # 4. 设计理念文档：适当切分
    if concept_docs:
        concept_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n\n", "\n", "。", "；", " ", ""],
            length_function=len,
        )
        concept_chunks = concept_splitter.split_documents(concept_docs)
        chunks.extend(concept_chunks)
        print(f"✅ 设计理念文档: {len(concept_docs)} → {len(concept_chunks)}")

    # 5. 摘要文档：适当切分
    if summary_docs:
        summary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=90,
            separators=["\n\n", "\n", "。", "；", " ", ""],
            length_function=len,
        )
        summary_chunks = summary_splitter.split_documents(summary_docs)
        chunks.extend(summary_chunks)
        print(f"✅ 系统摘要文档: {len(summary_docs)} → {len(summary_chunks)}")

    # 6. 其他文档：默认切分
    if other_docs:
        default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""],
            length_function=len,
        )
        other_chunks = default_splitter.split_documents(other_docs)
        chunks.extend(other_chunks)
        print(f"✅ 其他文档: {len(other_docs)} → {len(other_chunks)}")

    print(f"📈 切分完成: {len(documents)} → {len(chunks)} 个片段")

    return chunks

def build_vector_database(data_dir: str = "data/raw_docs") -> Tuple[bool, Dict[str, Any]]:
    """
    构建向量数据库

    Args:
        data_dir: 数据目录

    Returns:
        (成功标志, 统计信息)
    """
    print("\n" + "=" * 70)
    print("🏗️  开始构建向量数据库")
    print("=" * 70)

    stats = {
        "status": "started",
        "start_time": datetime.now().isoformat(),
        "data_dir": data_dir
    }

    try:
        # 检查依赖
        if not EMBEDDINGS_AVAILABLE:
            print("❌ 嵌入模块不可用")
            stats["status"] = "error"
            stats["error"] = "Embeddings module not available"
            return False, stats

        # 1. 加载文档
        print(f"\n1️⃣ 加载文档...")
        documents = load_documents(data_dir)

        if not documents:
            print("❌ 没有加载到文档")
            stats["status"] = "error"
            stats["error"] = "No documents loaded"
            return False, stats

        stats["loaded_documents"] = len(documents)

        # 2. 文档切分
        print(f"\n2️⃣ 文档切分...")
        chunks = split_documents(documents)

        if not chunks:
            print("❌ 文档切分失败")
            stats["status"] = "error"
            stats["error"] = "Document splitting failed"
            return False, stats

        stats["chunks_after_splitting"] = len(chunks)

        # 3. 准备数据...
        print(f"\n3️⃣ 准备数据...")
        texts = [chunk.page_content for chunk in chunks]
        metadatas = []

        def sanitize_metadata(metadata):
            """清理元数据，确保所有值都是ChromaDB支持的类型"""
            sanitized = {}
            for key, value in metadata.items():
                if value is None:
                    sanitized[key] = None
                elif isinstance(value, (str, int, float, bool)):
                    sanitized[key] = value
                elif isinstance(value, list):
                    # 列表转换为逗号分隔的字符串
                    sanitized[key] = ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    # 字典转换为JSON字符串
                    import json
                    sanitized[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, set):
                    # 集合转换为逗号分隔的字符串
                    sanitized[key] = ", ".join(str(item) for item in value)
                else:
                    # 其他类型转换为字符串
                    sanitized[key] = str(value)
            return sanitized

        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata.copy()

            # 清理元数据
            metadata = sanitize_metadata(metadata)

            metadata["chunk_id"] = i
            metadata["chunk_length"] = len(chunk.page_content)
            metadata["ingest_time"] = datetime.now().isoformat()
            metadatas.append(metadata)

        print(f"  文本数量: {len(texts)}")
        print(f"  平均长度: {sum(len(t) for t in texts) // len(texts) if texts else 0} 字符")

        stats["texts_prepared"] = len(texts)
        stats["avg_text_length"] = sum(len(t) for t in texts) // len(texts) if texts else 0

        # 4. 向量化
        print(f"\n4️⃣ 文本向量化...")
        embedder = get_embedding_manager()
        embeddings = embedder.embed_texts(texts)

        if not embeddings:
            print("❌ 向量化失败")
            stats["status"] = "error"
            stats["error"] = "Text embedding failed"
            return False, stats

        stats["embeddings_created"] = len(embeddings)
        stats["embedding_dimension"] = len(embeddings[0]) if embeddings else 0

        # 5. 存入向量数据库
        print(f"\n5️⃣ 保存到向量数据库...")
        collection = embedder.get_or_create_collection()

        # 生成唯一ID
        ids = [f"doc_{uuid.uuid4().hex[:12]}" for _ in range(len(texts))]

        # 批量添加（避免内存问题）
        batch_size = 100
        total_added = 0

        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))

            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]

            collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

            total_added += len(batch_texts)
            progress = (total_added / len(texts)) * 100
            print(f"  进度: {total_added}/{len(texts)} ({progress:.1f}%)")

        # 6. 收集统计信息
        collection_count = collection.count()

        # 文档类型统计
        doc_types = {}
        categories = {}

        for metadata in metadatas:
            doc_type = metadata.get("type", "unknown")
            category = metadata.get("category", "未知")

            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            if category != "未知":
                categories[category] = categories.get(category, 0) + 1

        # 更新统计信息
        stats.update({
            "status": "success",
            "end_time": datetime.now().isoformat(),
            "collection_name": collection.name,
            "collection_count": collection_count,
            "document_types": doc_types,
            "categories": categories,
            "total_processing_time": (
                datetime.now() - datetime.fromisoformat(stats["start_time"])
            ).total_seconds()
        })

        # 7. 显示结果
        print(f"\n" + "=" * 70)
        print("🎉 向量数据库构建成功！")
        print("=" * 70)

        print(f"\n📊 数据库统计:")
        print(f"  集合名称: {collection.name}")
        print(f"  存储位置: data/chroma_db/")
        print(f"  总文档数: {collection_count}")
        print(f"  向量维度: {len(embeddings[0]) if embeddings else 0}")

        print(f"\n📋 文档类型分布:")
        for doc_type, count in doc_types.items():
            percentage = (count / collection_count * 100) if collection_count > 0 else 0
            print(f"  {doc_type}: {count} ({percentage:.1f}%)")

        print(f"\n🎨 作品类别分布:")
        for category, count in categories.items():
            if category != "未知":
                percentage = (count / collection_count * 100) if collection_count > 0 else 0
                print(f"  {category}: {count} ({percentage:.1f}%)")

        print(f"\n⏱️  处理时间: {stats['total_processing_time']:.2f} 秒")

        print(f"\n💡 使用说明:")
        print(f"  1. 测试检索: python backend/rag/retriever.py")
        print(f"  2. 启动API: uvicorn backend.app:app --host 0.0.0.0 --port 8000")
        print(f"  3. 访问Web: http://localhost:8000")

        return True, stats

    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()

        stats.update({
            "status": "error",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "total_processing_time": (
                datetime.now() - datetime.fromisoformat(stats["start_time"])
            ).total_seconds()
        })

        return False, stats

def clear_vector_database(collection_name: str = "exhibition_docs") -> bool:
    """
    清空向量数据库

    Args:
        collection_name: 集合名称

    Returns:
        是否成功
    """
    print(f"\n⚠️  警告：即将清空向量数据库！")
    print(f"集合名称: {collection_name}")

    confirm = input("请输入 'DELETE' 确认操作: ")

    if confirm != "DELETE":
        print("操作已取消")
        return False

    try:
        if not EMBEDDINGS_AVAILABLE:
            print("❌ 嵌入模块不可用")
            return False

        embedder = get_embedding_manager()
        collection = embedder.get_or_create_collection(collection_name)

        # 获取当前文档数
        current_count = collection.count()

        # 删除所有文档
        collection.delete(where={})

        print(f"✅ 向量数据库已清空")
        print(f"  删除文档数: {current_count}")
        print(f"  当前文档数: {collection.count()}")

        return True

    except Exception as e:
        print(f"❌ 清空失败: {e}")
        return False

def get_database_info() -> Dict[str, Any]:
    """
    获取数据库信息

    Returns:
        数据库信息字典
    """
    try:
        if not EMBEDDINGS_AVAILABLE:
            return {"status": "error", "error": "Embeddings module not available"}

        embedder = get_embedding_manager()
        info = embedder.get_collection_info()

        return info

    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    """
    命令行接口
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="艺术与科技展览数据RAG入库系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python ingest.py --build                    # 构建向量数据库
  python ingest.py --clear                    # 清空数据库
  python ingest.py --info                     # 查看数据库信息
  python ingest.py --data-dir /path/to/data   # 指定数据目录
        """
    )

    parser.add_argument("--build", action="store_true", help="构建向量数据库")
    parser.add_argument("--clear", action="store_true", help="清空向量数据库")
    parser.add_argument("--info", action="store_true", help="查看数据库信息")
    parser.add_argument("--data-dir", type=str, default="data/raw_docs",
                       help="数据目录路径 (默认: data/raw_docs)")

    args = parser.parse_args()

    # 执行相应操作
    if args.clear:
        success = clear_vector_database()
        sys.exit(0 if success else 1)

    elif args.info:
        info = get_database_info()
        print("📊 数据库信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        sys.exit(0)

    elif args.build:
        success, stats = build_vector_database(args.data_dir)
        sys.exit(0 if success else 1)

    else:
        # 默认显示帮助
        parser.print_help()
        print(f"\n🎨 请使用以上参数运行本程序")
        sys.exit(1)
