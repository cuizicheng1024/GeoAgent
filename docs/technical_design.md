# 地理探究伴学智能体技术方案

## 1. 项目状态

地理探究伴学智能体 GeoMentor 面向中国地质大学（北京）地理信息科学相关课程，为学生提供 GIS 实验问答、研究问题建模、探究过程评价和通用 GIS 问答。当前应用已部署到 Render，公网地址为 <https://geoagent-i808.onrender.com>。本轮功能基线对应应用 commit `760babc`，文档更新会形成后续独立提交。

当前产品采用单一 Chatbot 入口。学生不需要选择模式，系统先识别当前问题的意图，再把请求转给相应能力。界面保留中国地质大学校徽、统一聊天区和底部联系方式。校徽以 Base64 形式内嵌，避免 Render 工作目录和 Gradio 静态文件路由变化导致图片无法加载。

Render 运行时固定为 Python 3.11.9。免费实例休眠后的冷启动通常需要约 30—50 秒，应用恢复后首次教材问答还会触发 RAG 索引懒加载。

## 2. 当前架构

系统采用单体 Python 应用，主入口为 `app.py`。Gradio 负责 Web 界面和流式消息展示，业务层负责意图分类与能力路由，检索层负责教材解析与 TF-IDF 召回，LLM 层通过 OpenAI 兼容接口调用 DeepSeek。

```text
学生输入
   │
   ▼
单一 Gradio Chatbot
   │
   ▼
LLM 意图分类
   ├── gis_tutor  ── 教材 RAG + DeepSeek
   ├── modeling   ── 六真框架 Prompt + DeepSeek
   ├── evaluation ── 五维评价 Prompt + DeepSeek
   └── general    ── GIS 专家 System Prompt + DeepSeek
   │
   ▼
stream=True 增量生成
   │
   ▼
Gradio yield 流式展示
```

当前不采用多 Agent 后台调度。现阶段的四类能力共享同一个模型调用层和会话入口，使用轻量路由即可满足课堂演示需求。只有在后续引入复杂工具执行、长任务编排或隔离式数据处理时，才有必要评估多 Agent 架构。

## 3. 前端与交互

页面使用单一 Chatbot，而不是多个 Tab。首次打开时，助手会用一句简短欢迎语说明可咨询的范围。学生直接输入问题，后台自动判断是教材操作问答、研究问题建模、过程评价还是通用 GIS 问答。

回答通过 Gradio 生成器持续 `yield`。DeepSeek 请求设置 `stream=True`，服务端收到增量 token 后立即更新最后一条助手消息。线上测试中首批内容约 6.6 秒出现，完整回答约 15.6 秒结束，期间记录到 622 次增量更新。总生成时间与非流式模式接近，但用户不再需要等待完整回答后才能看到内容。

校徽文件仍保存在 `static/cug_logo.png`，应用启动时读取图片并转换为 Base64 Data URL。这样不依赖 `/file/static/...` 路由，也不要求 Render 提供额外静态文件服务。

## 4. 能力路由

路由层先使用一次轻量 LLM 调用，只传当前用户消息，不带完整聊天历史。分类温度为 0，最大输出为 10 token。合法分类包括 `gis_tutor`、`modeling`、`evaluation` 和 `general`。如果分类接口异常或输出不在枚举内，系统使用本地规则做最小兜底，避免所有问题都落入同一固定回答。

| 分类 | 典型问题 | 后续能力 |
| --- | --- | --- |
| `gis_tutor` | 操作步骤、参数、报错、软件使用、模型和教材内容 | 教材检索后生成回答，并附教材依据 |
| `modeling` | 研究想法、选题、是否可探究 | 使用“六真”框架收窄问题并给出改写建议 |
| `evaluation` | 探究过程评价、评分和改进建议 | 按五个维度分别评分，总分 25 分 |
| `general` | 其他 GIS、遥感、空间分析和地理问题 | 使用 GIS 专家 System Prompt 直接回答 |

## 5. RAG 知识库

### 5.1 数据来源与解析

RAG 的主要知识来源是 `虚拟地理环境实践教程 0922版本.docx`。教材与应用代码一同部署，但生成后的 `index/kb.pkl` 不提交到 GitHub。

为降低内存占用，文档解析不再使用 `python-docx` 一次性打开整个 Office 包，而是通过 `zipfile` 打开 DOCX，再使用 `ElementTree.iterparse` 流式读取 `word/document.xml`。图片和其他二进制资源不会进入索引构建过程。

### 5.2 懒加载与缓存

应用启动时不会读取教材或构建矩阵，以保证 Render 尽快绑定端口。首次收到教材问答后，`ensure_kb()` 在进程锁保护下加载 `index/kb.pkl`；如果缓存不存在，则解析教材、切片、构建 TF-IDF 索引并写入缓存。同一进程后续请求直接复用内存索引和磁盘缓存。

```python
KB_AVAILABLE = False
KB_LOCK = threading.Lock()

def ensure_kb():
    if KB_AVAILABLE:
        return True
    with KB_LOCK:
        if not KB_AVAILABLE:
            KB.build(force=False)
    return KB_AVAILABLE
```

### 5.3 当前检索方案

当前检索使用 sklearn TF-IDF，`max_features=5000`，矩阵使用 `float32`。教材先按标题和正文关系切片，再对标题与正文拼接文本进行向量化。查询与 chunk 计算余弦相似度，默认返回 Top-K 片段，并把标题、正文和相似度注入问答 Prompt。

索引文件路径为 `index/kb.pkl`。缓存内容包括 chunk、TF-IDF 矩阵和 vectorizer。缓存不进入公开仓库，因为它包含教材派生文本并且可以在运行环境重新生成。

### 5.4 已知问题

当前 RAG 的稳定性仍低于预期。线上测试“土地利用方向有什么可以使用的模型？”时，没有稳定召回教材中的 PLUS、LEAS 和 CARS 章节；空间插值问题甚至出现 SWMM 章节以 `0.000` 相似度进入上下文的情况。这说明当前缺少低相关度过滤，中文词语切分和标题权重也不够合理。

下一阶段需要先增加最低相似度阈值，禁止 `0.000` 或明显低相关度片段进入 Prompt；随后增加适合中文的分词、关键词扩展和章节标题加权。中期可以将 TF-IDF 替换为中文效果更好的 embedding 与向量检索，并保留关键词检索作为混合召回的一部分。

## 6. LLM 与流式生成

当前主模型为 DeepSeek `deepseek-v4-flash`，接口地址为 `https://api.deepseek.com`，使用 OpenAI 兼容客户端。Key 通过 `DEEPSEEK_API_KEY` 环境变量读取，模型名可以通过 `DEEPSEEK_MODEL` 覆盖。

```python
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
stream = client.chat.completions.create(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    messages=messages,
    stream=True,
)
```

意图分类仍使用短响应的非流式调用，正式回答使用流式调用。若 DeepSeek 不可用且配置了 `OPENAI_API_KEY`，模型调用层可以切换到 OpenAI；如果两类 Key 都不可用，系统返回与问题类型相关的本地建议，而不是固定一句话。

## 7. System Prompt

System Prompt 已改为亲切、自然的学长或课程助教风格。助手需要先回答学生最关心的问题，再给步骤和原因；避免使用“需要向你说明”“请务必”等公文或训诫式表达。教材未覆盖时，助手会自然说明“这部分是通用做法，教材里没有展开”，并继续给可执行建议，而不是简单终止回答。

遇到报错时，助手按由易到难的顺序给出排查路径。遇到暂时不能确认的信息时，助手会说明还缺什么信息，以及学生下一步可以检查什么，不会只让学生自行确认。

## 8. 部署配置

Render 公网地址为 <https://geoagent-i808.onrender.com>。Python 版本通过 `.python-version` 和 `runtime.txt` 固定为 3.11.9，避免 Python 3.14 与旧版音频依赖链的兼容问题。Gradio 固定为 4.44.1，同时固定 FastAPI、Pydantic、HTTPX 和 Hugging Face Hub 的兼容版本。

Render 启动时读取平台注入的 `PORT`，并监听所有网卡。

```python
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
```

免费实例休眠后的冷启动约为 30—50 秒。冷启动完成后，普通 DeepSeek 回答的服务端生成时间大致在 9—13 秒；启用流式输出后，用户通常可以更早看到首批文字。

## 9. 待优化路线

| 优先级 | 事项 | 目标 |
| --- | --- | --- |
| P0 | 增加 RAG 最低相关度阈值 | 低相关度和 `0.000` 片段不进入 Prompt |
| P0 | 改进中文检索 | 支持中文分词、同义词扩展和标题加权 |
| P1 | 混合召回 | 结合关键词检索与 embedding 语义检索 |
| P1 | 更换向量方案 | 使用中文效果更好的 embedding 和向量库 |
| P1 | 引用质量 | 输出稳定的章、节和教材定位信息 |
| P2 | 数据体检恢复 | 在内存预算允许后恢复 Shapefile 数据体检与可视化 |
| P2 | 教师侧能力 | 增加作业记录、班级共性问题和形成性评价视图 |

## 10. 目录结构

```text
GeoAgent/
├── app.py
├── requirements.txt
├── .python-version
├── runtime.txt
├── README.md
├── 虚拟地理环境实践教程 0922版本.docx
├── static/
│   └── cug_logo.png
├── index/
│   └── kb.pkl              # 运行时生成，不提交
└── docs/
    ├── technical_design.md
    └── test_records.md
```

## 11. 结论

GeoMentor 已完成从多 Tab 演示页到单 Chatbot 入口的重构，并跑通 Render 公网部署、DeepSeek V4 Flash、LLM 能力路由、教材索引懒加载和流式输出。当前工程边界已经清晰，下一阶段最重要的工作不是继续堆叠功能，而是修复 RAG 中文召回和低相关度过滤，使教材问答真正稳定命中正确章节。
