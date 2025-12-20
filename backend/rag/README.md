# 📚 **给A同学的完整接口文档**

## 🎯 **项目概述**

我已经完成了**RAG检索系统**的开发，为你（A同学）的多模态模型提供**艺术作品知识检索**功能。你可以通过简单的接口调用，获取82个艺术作品的结构化信息。

## 📁 **项目结构**

```
text
vl-rag-system/
├── backend/
│   ├── rag/
│   │   ├── ingest.py          # 知识库构建脚本
│   │   └── retriever.py       # ✅ 核心检索接口（你需要调用这个）
│   ├── llm/                   # A同学的目录
│   │   └── qwen_vl.py         # A同学的模型推理
│   ├── app.py                 # C同学的FastAPI服务
│   └── requirements.txt       # 后端依赖文件
├── data/
│   ├── raw_docs/
│   │   └── 艺术与科技展览数据.xlsx
│   └── chroma_db_local_model/
├── models/
│   └── bge-small-zh-v1.5/
├── frontend/                  # C同学的目录
│   └── index.html
```

## 🔧 **接口说明**

### **核心接口类：MuseumRetriever**

**文件位置**：`backend/rag/retriever.py`

**功能**：
- 加载本地BGE中文嵌入模型
- 加载ChromaDB向量数据库
- 支持语义搜索82个艺术作品
- 返回结构化知识文本

## 🚀 **三种调用方式**

### **方式1：命令行调用（最简单）**
```bash
# 进入项目目录，激活虚拟环境后
python backend/rag/retriever.py --query "你的问题"

# 示例：
python backend/rag/retriever.py --query "永栖所的设计作者是谁"
python backend/rag/retriever.py --query "磁悬浮技术" --top_k 5
python backend/rag/retriever.py --query "传统文化" --simple
```

### **方式2：Python模块导入调用**
```python
import sys
from pathlib import Path

# 1. 添加项目路径
project_root = Path(__file__).parent.parent  # 假设你的qwen_vl.py在backend/llm/
sys.path.append(str(project_root))

# 2. 导入我的检索器
from rag.retriever import MuseumRetriever

# 3. 初始化
retriever = MuseumRetriever()

# 4. 检索知识
knowledge = retriever.retrieve("你的问题", top_k=3)
```

### **方式3：命令行调用（在你的Python代码中）**
```python
import subprocess

def get_knowledge_from_rag(query: str, top_k: int = 3) -> str:
    """调用我的RAG系统获取知识"""
    result = subprocess.run(
        ['python', 'backend/rag/retriever.py', '--query', query, '--top_k', str(top_k)],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    return result.stdout

# 使用
knowledge = get_knowledge_from_rag("这是什么作品？")
```

## 📋 **API详细说明**

### **1. `retrieve(query: str, top_k: int = 3) -> str`**

**功能**：核心检索接口，返回格式化的知识文本

**参数**：
- `query`: 查询问题，支持自然语言
- `top_k`: 返回前几个相关作品，默认3个

**返回值格式**：
```text
============================================================
作品名称：《作品1名称》
设计作者：作者1
指导老师：指导老师1
类别标签：类别1
呈现形式：形式1
作品描述：描述内容...
设计动机：动机内容...
灵感来源：灵感内容...
技术特点：技术内容...
所属展区：展区1

============================================================
作品名称：《作品2名称》
设计作者：作者2
...
============================================================
```

**示例**：
```python
knowledge = retriever.retrieve("这是什么作品？")
print(knowledge[:500])  # 查看前500字符
```

### **2. `search(query: str, top_k: int = 5, show_full: bool = True)`**

**功能**：带格式的交互式搜索（用于调试）

**参数**：
- `query`: 查询问题
- `top_k`: 返回结果数量
- `show_full`: 是否显示完整内容

### **3. `get_stats() -> dict`**

**功能**：获取知识库统计信息

**返回示例**：
```python
{
    "total_documents": 82,
    "embedding_model": "bge-small-zh-v1.5",
    "status": "ready",
    "database_path": "data/chroma_db_local_model"
}
```

## 🎨 **知识库内容**

### **包含的82个作品类型**：
1. **工业设计类**（36个）- 交通工具、产品设计、交互设计等
2. **环境设计类**（16个）- 可持续设计、空间设计、景观设计等  
3. **艺术与科技类**（30个）- 数字文娱、展示艺术、互动装置等

### **每个作品包含18个字段**：
- 作品名称、设计作者、指导老师、类别标签
- 呈现形式、作品描述、创作时间
- 设计动机、灵感来源、设计目的/意义
- 设计理念/风格、视觉形式语言、技术特点
- 预期效果、创作历程、面临的困难
- 所属展区

## 🧪 **测试查询示例**

```python
# 技术相关的查询
retriever.retrieve("磁悬浮技术", top_k=2)
retriever.retrieve("虚幻引擎5", top_k=2)
retriever.retrieve("RFID交互", top_k=2)

# 主题相关的查询
retriever.retrieve("传统文化 现代转化", top_k=2)
retriever.retrieve("环境保护 可持续发展", top_k=2)
retriever.retrieve("儿童心理成长", top_k=2)

# 人员相关的查询
retriever.retrieve("王心妍", top_k=3)
retriever.retrieve("温馨", top_k=3)
retriever.retrieve("林哲轩", top_k=3)

# 作品相关的查询
retriever.retrieve("未来出行概念汽车设计", top_k=1)
retriever.retrieve("哈尼印象", top_k=1)
retriever.retrieve("红色脉冲", top_k=1)
```

## 🔗 **与你的模型集成示例**

```python
class QwenVLInference:
    def __init__(self):
        # 初始化我的RAG检索器
        self.retriever = MuseumRetriever()
        print(f"✅ RAG知识库已加载: {self.retriever.get_stats()['total_documents']} 个作品")
    
    def identify_product(self, image_data, question: str):
        # 1. 调用我的RAG获取相关知识
        knowledge_text = self.retriever.retrieve(question, top_k=3)
        
        # 2. 构建多模态prompt
        prompt = f"""
        基于以下知识库内容回答问题：
        {knowledge_text}
        
        用户问题：{question}
        
        请结合图片内容，给出准确的回答。
        """
        
        # 3. 调用你的Qwen2-VL模型
        answer = self._call_your_model(image_data, prompt)
        
        # 4. 返回结果
        return {
            "success": True,
            "answer": answer,
            "context": knowledge_text[:200] + "...",  # 截取部分用于调试
            "confidence": "高",
            "error": None
        }
```

## ⚙️ **环境要求**

### **Python版本**：3.9+
### **操作系统**：Windows/Linux/macOS

### **依赖列表**（requirements.txt）：
```txt
# ============================================
# RAG系统完整依赖列表
# Python 3.9 兼容版本
# ============================================

# 核心数据处理
pandas==1.5.3
openpyxl==3.1.2
numpy==1.24.3

# 向量数据库
chromadb==0.4.22

# 文本嵌入模型
sentence-transformers==2.2.2
transformers==4.30.2
tokenizers==0.13.3

# ChromaDB依赖
httpx==0.24.1
pydantic==1.10.7
onnxruntime==1.14.1
posthog==2.4.0

# 工具和工具链
tqdm==4.65.0
typing-extensions==4.5.0
huggingface-hub==0.16.4
protobuf==3.20.3

```

### **安装命令**：
```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 安装依赖
pip install pandas==1.5.3 openpyxl==3.1.2 numpy==1.24.3
pip install sentence-transformers==2.2.2
pip install chromadb==0.4.22 --no-deps
pip install httpx==0.24.1 pydantic==1.10.7 onnxruntime==1.14.1

注：如果这个依赖安装不行，直接用这个命令：
# 虚拟环境中运行（一次性安装所有核心包）
pip install pandas openpyxl sentence-transformers chromadb
运行时间长（大概20-30分钟）

```

## 🚀 **快速开始指南**

### **步骤1：确保环境正确**
```bash
# 进入项目目录
cd D:\大三上学期\人形机器人项目实践\vl-rag-system

# 激活虚拟环境
.\venv\Scripts\activate

# 或直接使用虚拟环境的Python
.\venv\Scripts\python.exe backend/rag/retriever.py --query "测试"
```

### **步骤2：测试接口**
```python
# test_rag.py
import sys
from pathlib import Path

# 设置路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from rag.retriever import MuseumRetriever

def test():
    # 初始化
    retriever = MuseumRetriever()
    
    # 获取统计
    stats = retriever.get_stats()
    print(f"📊 知识库: {stats['total_documents']} 个作品")
    
    # 测试检索
    query = "永栖所的设计作者是谁"
    knowledge = retriever.retrieve(query, top_k=2)
    print(f"🔍 查询: {query}")
    print(f"📝 返回知识长度: {len(knowledge)} 字符")
    print(knowledge[:300])  # 查看前300字符
    
if __name__ == "__main__":
    test()
```

## 🆘 **故障排除**

### **问题1：导入失败**
```python
# 确保正确添加项目路径
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据你的文件位置调整
sys.path.append(str(project_root))
```

### **问题2：数据库不存在**
```
❌ 向量数据库不存在
💡 请先运行 'python backend/rag/ingest.py' 构建数据库
```
**解决**：运行 `python backend/rag/ingest.py` 构建数据库（已为你构建好）

### **问题3：模型加载失败**
```
❌ 模型不存在
```
**解决**：确保 `models/bge-small-zh-v1.5/` 目录存在且包含5个模型文件

## 📞 **支持信息**

1. ✅ `retrieve()` 接口已完全实现并测试通过
2. ✅ 支持中文自然语言查询
3. ✅ 返回结构化知识文本
4. ✅ 包含82个作品的完整信息
5. ✅ 语义搜索准确率高


## 🎯 **一句话总结**

**A同学，在你的 `qwen_vl.py` 中：**
```python
from rag.retriever import MuseumRetriever

retriever = MuseumRetriever()
knowledge = retriever.retrieve(你的问题, top_k=3)
# 然后结合图片和knowledge调用你的模型
```

**或者用命令行：**
```bash
python backend/rag/retriever.py --query "你的问题"
```

**RAG系统已就绪，随时为你提供知识支持！** 🚀

---

**文档版本**：v2.0  
**负责人**：B同学（RAG & 数据）