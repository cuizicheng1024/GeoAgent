---
title: GeoMentor
emoji: 🌏
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# 地理探究伴学智能体 MVP

这是一个面向《中国地质大学（北京）虚拟地理环境实践课程》的 Web Chatbot 演示版。当前版本优先把效果跑通，不做复杂后台、不做多 Agent 调度。

## 功能范围

当前 MVP 包含三个 Tab：GIS 实验助教、问题建模助手和过程评价。

GIS 实验助教会解析工作区中的 `虚拟地理环境实践教程 0922版本.docx`，自动切分教材文本并建立本地向量索引。学生提问时，系统先检索教材相关片段，再调用 GPT-4o 生成操作步骤回答，并展示命中的教材章节。

问题建模助手会基于“六真”课程框架，即真场景、真数据、真处理、真模拟、真应用、真评价，判断学生提出的问题是否适合做地理探究，并给出收窄和改写建议。

过程评价会按五个维度评分：问题提出、数据准备、方法选择、结论表达、反思改进。每项 0-5 分，合计 25 分。当前评分细则是 MVP 临时版本，后续可按张春晓老师要求调整。

## 启动方式

先进入目录并安装依赖：

```bash
cd geo_agent
python -m pip install -r requirements.txt
```

设置 OpenAI Key：

```bash
export OPENAI_API_KEY="你的 OpenAI API Key"
```

启动 Web 界面：

```bash
python app.py
```

默认访问地址是：

```text
http://127.0.0.1:7860
```

如果端口被占用，可以指定端口：

```bash
PORT=7861 python app.py
```

## 文件依赖

程序默认从 `geo_agent/` 的上一级目录读取两份课程材料：

```text
虚拟地理环境实践教程 0922版本.docx
“六真”虚拟地理环境实践智慧课程建设 张春晓 河海大学.pptx
```

当前代码实际用于 RAG 的主来源是 Word 教材。PPT 中的“六真”框架已整理进问题建模助手提示词，后续如果需要，也可以把 PPT 文本一并纳入知识库。

## Embedding 与索引说明

默认逻辑是：如果检测到 `OPENAI_API_KEY`，优先使用 OpenAI `text-embedding-3-small` 生成教材向量，并使用 FAISS 做相似度检索；如果没有 OpenAI Key，则尝试使用本地 `sentence-transformers` 模型 `paraphrase-multilingual-MiniLM-L12-v2`；如果本地模型不可用，会降级使用 TF-IDF 检索，保证演示不被 embedding 模型下载卡死。

如果希望强制使用 OpenAI embedding，可以这样启动：

```bash
EMBEDDING_BACKEND=openai python app.py
```

如需重建教材索引，可以在界面点击“重建教材索引”，或者删除 `geo_agent/index/kb.pkl` 后重新启动。

## 演示建议

建议现场演示三类问题：

GIS 实验助教可以问：“PLUS 模型里如何提取 2010 到 2020 年土地扩张？”或“SWAT 建模前需要准备哪些数据？”

问题建模助手可以输入：“北京城市扩张对生态环境有什么影响？”系统应该提示这个问题过大，并建议补充空间范围、时间范围、数据和评价指标。

过程评价可以粘贴一段学生项目描述，让系统按五维度给出分项分数、反馈和改进建议。

## 启动后界面说明

界面顶部会显示知识库状态，包括教材文件名、切分片段数和当前 embedding 后端。下面有三个 Tab，分别对应 GIS 实验助教、问题建模助手和过程评价。每个 Tab 都只有输入框和执行按钮，适合给委托方快速演示“学生提问—智能体反馈”的基本闭环。

## 当前边界

这是 P0 演示版。它不会保存学生历史记录，不做登录权限，不做作业管理后台，也不会解析教材图片。Word 中的公式、图片和复杂表格会尽量跳过或保留可读文本。后续如果进入正式版，需要补充课程管理、学生记录、教师可配置评分细则、教材章节导航、答案引用定位和人工复核机制。

## DeepSeek 配置

优先使用 DeepSeek：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
python app.py
```

调用地址为 `https://api.deepseek.com`，默认模型为 `deepseek-v4-flash`，也可通过 `DEEPSEEK_MODEL` 覆盖。如果没有配置 DeepSeek Key，但配置了 `OPENAI_API_KEY`，服务会自动改用 OpenAI。

## 构建本地教材索引

仓库不包含 `index/kb.pkl`，也不包含课程教材。请先准备教材 Word 文件，再执行：

```bash
python build_index.py --textbook "/path/to/虚拟地理环境实践教程 0922版本.docx" --backend tfidf
```

索引会生成到 `index/kb.pkl`。该文件已被 `.gitignore` 排除，不应提交到公开仓库。
