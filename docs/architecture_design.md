# VL-RAG-System 架构设计文档

本文档定义了“技心”多模态交互机器人的整体技术方案、模块规范及数据流转逻辑。

---

## 1. 核心架构图 (Core Architecture Diagrams)

### 1.1 系统全局架构
![系统架构图](架构图.png)

### 1.2 逻辑调用流程
![模块调用流程图](模块调用流程图.png)
![内部模块调用细节](内部模块调用细节图.png)

---

## 2. 架构分层 (Architectural Layers)

系统采用高度解耦的五层架构设计，通过同步响应链 (Sync Chain) 与异步内省链 (Async Reflection) 实现智慧闭环：

1.  **感官层 (Perception Layer)**: `ASR` (听觉) 与 `Vision` (视觉) 服务，负责原始环境信号的语义抽象。
2.  **认知层 (Cognitive Layer)**: `LLM` (语言大脑)、**`Memory System` (记忆系统)** 与 **`Retrieval Engine` (检索引擎)**。
3.  **代理层 (Agent Layer)**: **核心业务大脑**，执行认知循环并调用记忆系统获取与存储所有信息。
4.  **调度层 (Orchestration Layer)**: `System Orchestrator` (总控)，负责跨模块的状态机维护、路由管理及任务编排。
5.  **反馈层 (Execution Layer)**: `TTS`、`Body Controller` (肢体控制) 与前端展示组件，完成交互闭环。

---

## 3. 详细模块设计 (Detailed Module Design)

---

### 3.1 系统总控与 API 层 (Orchestration & API)
- **文件**: `local_model_processor.py`, `api/routers/`
- **职责**: 负责 WebSocket 连接维护、分发请求至 Agent 以及执行同步/异步双环路由。
- **关键接口**:
    - `dispatch(query: str, image: bytes) -> Stream`: 分发原始输入并返回流式响应。
    - `identify_intent(query: str) -> Intent`: (核心细节) 语义意图识别，决定后续路由。
    - `handle_async_reflection(chat_history: list) -> None`: 触发后台内省任务。

### 3.2 认知引擎模块 (Cognitive Engines)
#### 3.2.1 检索引擎 (Retrieval Engine)
- **职责**: **“底层技术插件”**。专门负责向量数据库的原子搜索、重排序与语义对齐。
- **定位**: **检索为记忆服务**。它不直接对接代理层，仅作为记忆系统的基础设施。
- **各引擎组件 (Engine Components)**:
    - **`StaticRAGProvider`**: 展品专业库向量匹配。
    - **`InsightArchiveProvider`**: 同步读取见解库向量。
    - **`UserGroupProvider`**: 匹配特定群体的配置信息。
- **核心接口 (针对记忆层内部)**:
    - **`vector_search(query: str, collection: str)`**: 原子级向量召回。
    - **`fuse_knowledge(chunks: list) -> str`**: 结果融合与语义压制。
    - `rebuild_index()`: 索引维护。

- **接口定义示例 (Python)**:
```python
class MemoryProvider:
    """记忆提供者基类"""
    async def search(self, query: RetrievalQuery) -> List[KnowledgeChunk]:
        raise NotImplementedError

class StaticRAGProvider(MemoryProvider):
    """静态展品知识检索 (ChromaDB Collection A)"""
    async def search(self, query: RetrievalQuery) -> List[KnowledgeChunk]:
        # 实现专业展品背景的向量检索
        pass

class InsightArchiveProvider(MemoryProvider):
    """历史对话见解检索 (ChromaDB Collection B)"""
    async def search(self, query: RetrievalQuery) -> List[KnowledgeChunk]:
        # 实现对该用户过往“觉察/见解”的语义检索
        pass

class UserGroupProvider(MemoryProvider):
    """用户群体偏好检索 (PostgreSQL/JSON)"""
    async def search(self, query: RetrievalQuery) -> List[KnowledgeChunk]:
        # 实现对群体审美偏好、交流风格的提取与 Prompt 化转换
        pass

class RetrievalOrchestrator:
    """多路检索编排器 (指挥官)"""
    def __init__(self, providers: List[MemoryProvider]):
        self.providers = providers

    async def multi_path_retrieve(self, query: RetrievalQuery) -> List[KnowledgeChunk]:
        import asyncio
        # 1. 并发执行所有检索 (通过 asyncio 提升多库查询性能)
        tasks = [provider.search(query) for provider in self.providers]
        results = await asyncio.gather(*tasks)
        
        # 2. 扁平化结果列表并进行全局得分排序
        flat_results = [chunk for sublist in results for chunk in sublist]
        return sorted(flat_results, key=lambda x: x.score, reverse=True)
```

#### 3.2.2 记忆系统 (Memory System)
- **目录位置**: `memory/` (根文件夹)
- **职责**: 管理跨时空的对话上下文、长期洞察积淀与用户信息。
- **核心数据结构**:
```python
class UserGroupProfile(BaseModel):
    """用户群体画像模型 (User Group Profiles)"""
    group_id: str                 # 群体唯一标识 (如: "youth_tech", "elderly_family")
    category_name: str            # 类别名 (如: "科技青年", "亲子家庭")
    aesthetic_pref: str           # 审美偏好 (如: "前卫简约", "传统温馨")
    communication_pref: str       # 交流偏好 (如: "深度技术讨论", "浅显易懂介绍")
    typical_tags: List[str]       # 该群体典型标签: ["极客", "参数党"]
    response_style: Dict          # 响应风格定制: {"speed": "fast", "detail_level": "high"}

class InsightEntry(BaseModel):
    """对话见解模型 (Insight Archive)"""
    insight_id: str
    topic: str                    # 提取的主题: "设备操作疑问"
    content: str                  # 深度摘要: "用户曾表达过对 XX 展品手势交互的困惑"
    key_entities: List[str]       # 关联实体: ["双子机器人", "空间交互"]
    timestamp: datetime
    embedding: List[float]        # 见解内容的语义向量
```

**核心子模块与接口 (记忆 Hub 接口层)**:
- **定位**: **记忆对外提供服务**。作为代理层获取与存储信息的唯一官方入口。
1.  **记忆获取接口 (Read)**:
    - **`recall(query: str, user_id: str) -> List`**: (业务感知) 内部调用检索引擎进行多路回想。
    - **`get_group_config(group_id: str)`**: (配置读取) 获取该群体的交互风格与配置。
    - **`fetch_session(session_id: str)`**: (上下文调取) 调取本轮对话历史。
2.  **记忆持久化接口 (Write)**:
    - **`commit_insight(entry: InsightEntry)`**: (写接口) 异步写入新产生的交互见解。
    - **`save_group_profile(profile: UserGroupProfile)`**: (写接口) 定义或更新群体交互策略。
    - **`match_group(user_features: dict)`**: 基于特征匹配 ID。
3.  **核心文件**: `memory/static_rag.py`, `memory/insight_archive.py`, `memory/user_group_profiles.py`

#### 3.2.3 共鸣引擎 (Resonance Engine)
- **文件**: `services/resonance_engine.py`
- **职责**: 实现“技心”人设的人格化算法，调节回应的情感质感与美学比重。
- **核心接口**: `calculate_vibe(text_input)`, `apply_persona_filter(raw_response)`

### 3.3 代理层 (Agent Layer)
代理层作为整个系统的业务大脑，根据交互场景选择最合适的智能体模块进行响应。其内部遵循 **Perceive-Retrieve-Plan-Execute-Context-Reflect** 的六步认知循环框架。

- **核心数据结构**:
    - **`PerceptionResult`**: `{intent, entities, scene_description, objects_detected}`
    - **`PlanningResult`**: `{selected_agent, internal_thought, reasoning_chain}`

> [!NOTE]
> 详细的认知循环定义与各个阶段的逻辑细节请参考：[认知引擎设计文档 (Cognition Engine)](cognition_engine.md)

#### 3.3.1 展品介绍代理 (Exhibit Intro Agent)
- **职责**: 结合 RAG 检索结果与实况视觉特征，提供专业化、权威的展品背景讲解。
- **关键逻辑**: 事实对齐 (Fact Alignment) + 展示互动引导。

#### 3.3.2 深度聊天代理 (Deep Chat Agent)
- **职责**: 利用记忆系统及复杂的推理链条，与用户进行深度、跨轮次的语义探讨。
- **关键逻辑**: 多轮上下文感知 (Context-Awareness) + 知识图谱扩展。

#### 3.3.3 闲聊代理 (Small Talk Agent)
- **职责**: 负责身份认同、日常寒暄及情感抚慰。
- **关键逻辑**: 人格化 Prompt 注入 + 情感共鸣算法输出。

### 3.4 感知与执行模块 (Perception & Execution)
#### 3.4.1 视觉系统 (Vision System)
- **文件**: `services/vision_service.py`
- **职责**: 负责实时摄像头流的捕捉、快照分析及展品特征提取。
- **核心接口**: `capture_snapshot()`, `get_latest_frame()`

#### 3.4.2 听觉系统 (Hearing System)
- **文件**: `services/asr_service.py`
- **职责**: 负责音频降噪、语义断句及文字转化 (ASR)。
- **核心接口**: `start_listening()`, `stop_listening()`

#### 3.4.3 语言系统 (Language System)
- **文件**: `services/llm_service.py`
- **职责**: 负责大模型的底层调用、流式输出管理及代理提示词注入。
- **核心接口**:
    - `generate_stream(prompts, history)`: 发起流式回复。
    - `on_generation_complete(history)`: (钩子函数) 触发异步内省逻辑。
    - `generate_sync(prompts)`: 发起同步调用。

### 3.5 执行控制层 (Execution Layer)
#### 3.5.1 行为控制器 (Behavior Controller)
- **职责**: 将代理决策转化为底层的硬件执行指令。
- **数据结构**:
    - **`AgentAction`**: `{reply_text, motion_id, tts_config: dict, expression_id, light_effect}`
- **核心接口**: `sync_execute(action: AgentAction) -> None`

---

## 4. 模块交互与数据流 (Interaction & Data Flow)

### 4.1 交互时序图
![时序图](时序图.png)

### 4.2 数据流描述
1.  **语音唤醒**: 听觉系统解析语音 ⮕ 转化为文本 ⮕ 发布到 ROS 话题。
2.  **视觉捕捉**: 总控监听话题 ⮕ 触发视觉系统拍摄最新帧图像。
3.  **认知分析**: 认知层提取展品专业背景并召回历史相关记忆。
4.  **代理决策**: 代理层根据当前 Context (展品、历史、闲聊) 选取合适的 Agent (介绍/深聊/闲聊) 生成原始文本。
5.  **智慧生成**: 语言系统整合 Agent 策略、人设与历史 ⮕ 产生流式响应内容。
6.  **反馈输出**: 响应内容推送到前端显示，并触发 TTS 进行语音播报。

---

## 5. 目录结构 (Directory Structure)

```text
vl-rag-system/
├── agents/                  # 🤖 代理层 (业务思维与策略路由)
│   ├── base_agent.py        # 🆔 代理通用基类
│   ├── intro_agent.py       # 🏺 展品讲解专有代理
│   ├── chat_agent.py        # 💬 深度聊天专有代理
│   └── smalltalk_agent.py   # 🌸 闲聊与情感专有代理
├── api/                     # 🌐 Web 接口层 (FastAPI 分层实现)
│   ├── routers/             # 🚦 路由定义 (URL 路径)
│   │   ├── chat_router.py
│   │   └── system_router.py
│   └── controllers/         # 🛡️ 控制逻辑 (参数校验与服务调用)
│       ├── chat_controller.py
│       └── system_controller.py
├── services/                # 🧱 核心服务层 (业务逻辑与节点封装)
│   ├── agent_manager.py     # 🧭 代理路由与场景分发
│   ├── llm_service.py       # 🧠 语言大脑与核心生成
│   ├── tts_service.py       # 🔊 语音合成输出
│   ├── asr_service.py       # 🎙️ 听觉识别服务
│   └── vision_service.py    # 📸 视觉捕捉服务
├── memory/                  # 🧠 记忆系统 (根目录级核心模块)
│   ├── static_rag.py        # 📚 静态检索与常识库
│   ├── insight_archive.py   # 📁 对话洞察与交互档案
│   └── user_group_profiles.py # 👥 (新) 用户群体画像与类别管理中心
├── local_model_processor.py # 🤖 系统总控调度器 (Orchestrator)
├── main.py                  # 🏁 Web 服务入口与启动配置
├── config.py                # ⚙️ 全局配置中心
├── rag/                     # 📚 RAG 检索逻辑与知识管理
├── prompts/                 # 📝 提示词模板 (包含各 Agent 人设协议)
├── docs/                    # 📂 架构图、时序图与设计文档
├── data/                    # 💾 数据库与临时缓存
└── models/                  # 🤖 本地模型存放 (Embedding 等)
```

---

## 6. 技术选型 (Technology Stack)

| 核心维度 | 技术选型 | 作用说明 |
| :--- | :--- | :--- |
| **机器人框架** | ROS 2 Humble | 组件化异步通信与硬件节点管理 |
| **后端框架** | Python + FastAPI | 高性能、异步化的业务逻辑支撑 |
| **多模态核心** | DeepSeek / Qwen | 核心的语义理解、推理与视觉对齐 |
| **推理引擎** | Ollama / API | 驱动大语言模型的高效运行 |
| **向量数据库** | ChromaDB | 实时的展品专业知识向量检索 |
| **嵌入模型** | BGE-Small-ZH | 本地化的中文语义向量化 |
| **前端展现** | HTML + Vue 3 | 现代、组件化的交互式仪表盘与视觉反馈 |
