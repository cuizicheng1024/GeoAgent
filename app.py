import os
import re
import json
import pickle
import threading
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 兼容部分企业代理环境：老版本 httpx 在解析 no_proxy 里的 IPv6/CIDR 项时可能报 Invalid port。
# 清理 IPv6 项不影响本地 127.0.0.1 访问，也能避免 Gradio import 阶段被代理环境卡住。
for _proxy_key in ("no_proxy", "NO_PROXY"):
    if _proxy_key in os.environ:
        os.environ[_proxy_key] = ",".join([x for x in os.environ[_proxy_key].split(",") if "::" not in x.strip()])

import gradio as gr
import numpy as np
import requests
from openai import OpenAI

faiss = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
INDEX_DIR = APP_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

TEXTBOOK_PATH = Path(os.getenv("TEXTBOOK_PATH", str(APP_DIR / "虚拟地理环境实践教程 0922版本.docx")))
if not TEXTBOOK_PATH.exists():
    TEXTBOOK_PATH = ROOT_DIR / "虚拟地理环境实践教程 0922版本.docx"
PPT_PATH = Path(os.getenv("PPT_PATH", str(APP_DIR / "“六真”虚拟地理环境实践智慧课程建设 张春晓 河海大学.pptx")))
if not PPT_PATH.exists():
    PPT_PATH = ROOT_DIR / "“六真”虚拟地理环境实践智慧课程建设 张春晓 河海大学.pptx"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "tfidf").lower()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "5"))

SYSTEM_STYLE = """你是"地理探究伴学智能体"，是 GIS 领域的专家，熟知虚拟地理环境理论与实践、ArcGIS/QGIS/ENVI 等主流 GIS 软件的操作与工作流、地理建模与空间分析方法（含遥感、水文、城市扩张、土地利用、生态模拟等）、常见数据格式与坐标系规范，以及学生在 GIS 实践中遇到的典型认知障碍和操作报错。你具有丰富的实践经验，服务对象是中国地质大学（北京）地理信息科学相关课程学生。

回答要求：
1. 用中文，表达清楚，先给结论，再给步骤或建议。
2. 如果依据教材回答，必须优先使用检索到的教材片段，不要编造教材中没有的操作；如教材未涵盖，可结合 GIS 专业知识补充，并明确说明来源。
3. 遇到不确定信息，要明确说"不确定"，并建议学生回到教材对应章节或向老师确认。
4. 保持助教风格：具体、可执行、不过度替学生完成思考；遇到报错或卡点时，给出排查思路而不仅仅是答案。"""

SIX_REAL_FRAMEWORK = """“六真”课程框架：真场景、真数据、真处理、真模拟、真应用、真评价。
课程目标是以城市扩张、森林火灾、地表水文、城市内涝等真实场景为任务导向，让学生综合运用 GIS、遥感、地理建模与模拟分析能力，完成从问题提出、数据处理、模型模拟到结果解释和评价反思的完整探究过程。"""


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heading_text(text: str, style_name: str = "") -> bool:
    if not text:
        return False
    if style_name.startswith("Heading") or style_name.startswith("标题"):
        return True
    return bool(re.match(r"^(第[一二三四五六七八九十0-9]+[章节篇]|\d+(\.\d+){0,3}\s+|[一二三四五六七八九十]+、)", text))


def extract_docx_blocks(path: Path) -> List[Dict]:
    """流式读取 document.xml，跳过图片和其他二进制资源。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到教材文件：{path}")

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    blocks = []
    current_heading = "教材正文"
    buffer = []

    def flush():
        nonlocal buffer
        content = clean_text("\n".join(buffer))
        if content:
            blocks.append({"heading": current_heading, "text": content, "source": path.name})
        buffer = []

    with zipfile.ZipFile(path) as archive:
        with archive.open("word/document.xml") as xml_file:
            for event, elem in ET.iterparse(xml_file, events=("end",)):
                if elem.tag != ns + "p":
                    continue
                text = clean_text("".join(node.text or "" for node in elem.iter(ns + "t")))
                style_node = elem.find(f"{ns}pPr/{ns}pStyle")
                style_name = style_node.get(ns + "val", "") if style_node is not None else ""
                if text:
                    if is_heading_text(text, style_name):
                        flush()
                        current_heading = text[:120]
                    else:
                        buffer.append(text)
                elem.clear()
    flush()
    return blocks


def split_chunks(blocks: List[Dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    chunks = []
    for block in blocks:
        text = block["text"]
        heading = block["heading"]
        if len(text) <= chunk_size:
            chunks.append({**block, "chunk": text})
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end]
            chunks.append({"heading": heading, "text": text, "chunk": piece, "source": block["source"]})
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


class EmbeddingEngine:
    def __init__(self):
        self.backend = None
        self.model = None
        self.vectorizer = None
        self.openai_client = None

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if TfidfVectorizer is None:
            raise RuntimeError("缺少 scikit-learn，无法构建 TF-IDF 索引。")
        self.backend = "tfidf"
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), dtype=np.float32)
        matrix = self.vectorizer.fit_transform(texts)
        return matrix.toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.backend == "tfidf":
            return self.vectorizer.transform(texts).astype("float32").toarray()
        raise RuntimeError("EmbeddingEngine 尚未初始化。")


class KnowledgeBase:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.engine = EmbeddingEngine()

    def build(self, force: bool = False):
        cache_path = INDEX_DIR / "kb.pkl"
        if cache_path.exists() and not force:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            self.chunks = payload["chunks"]
            self.embeddings = payload["embeddings"]
            self.engine = payload["engine"]
            self._build_index()
            return

        blocks = extract_docx_blocks(TEXTBOOK_PATH)
        self.chunks = split_chunks(blocks)
        texts = [f"{c['heading']}\n{c['chunk']}" for c in self.chunks]
        self.embeddings = self.engine.fit_transform(texts)
        self._build_index()
        with cache_path.open("wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings, "engine": self.engine}, f)

    def _build_index(self):
        if self.embeddings is None:
            raise RuntimeError("索引构建失败：embeddings 为空。")
        emb = np.asarray(self.embeddings, dtype="float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        self.embeddings = emb
        if faiss is not None:
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
        else:
            self.index = None

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        if not query.strip():
            return []
        q = self.engine.transform([query])
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        if self.index is not None:
            scores, ids = self.index.search(q.astype("float32"), top_k)
            pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        else:
            sims = (self.embeddings @ q[0]).reshape(-1)
            ids = np.argsort(-sims)[:top_k]
            pairs = [(int(i), float(sims[i])) for i in ids]
        results = []
        for idx, score in pairs:
            item = dict(self.chunks[idx])
            item["score"] = score
            results.append(item)
        return results


KB = KnowledgeBase()
KB_AVAILABLE = False
KB_ERROR = ""
KB_LOCK = threading.Lock()


def ensure_kb() -> bool:
    """首次需要教材检索时才加载或构建索引，避免阻塞 Web 服务启动。"""
    global KB_AVAILABLE, KB_ERROR
    if KB_AVAILABLE:
        return True
    with KB_LOCK:
        if KB_AVAILABLE:
            return True
        try:
            KB.build(force=False)
            KB_AVAILABLE = True
            KB_ERROR = ""
        except Exception as error:
            KB_ERROR = str(error)
            print(f"知识库加载失败，将使用纯 LLM 模式：{KB_ERROR[:180]}")
    return KB_AVAILABLE


def get_aime_llm_headers() -> Dict[str, str]:
    token = os.getenv("AIME_USER_CLOUD_JWT") or os.getenv("IRIS_USER_CLOUD_JWT")
    if not token:
        return {}
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Assistant-Id": os.getenv("AIME_ASSISTANT_ID", ""),
        "X-Assistant-LanId": os.getenv("AIME_ASSISTANT_LAN_ID", ""),
        "X-Aime-Model-Resource": os.getenv("AIME_MODEL_RESOURCE", ""),
    }


def call_llm(messages: List[Dict], temperature: float = 0.2, max_tokens: int = 2048) -> str:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    try:
        if deepseek_key:
            client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"), messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
        if openai_key:
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"), messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
    except Exception as error:
        print(f"LLM 调用失败：{str(error)[:180]}")
    return ""


def format_context(results: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[片段{i}] 来源：{r['source']}｜章节：{r['heading']}｜相关度：{r['score']:.3f}\n{r['chunk']}")
    return "\n\n".join(lines)


def gis_tutor(question: str) -> Tuple[str, str]:
    results = KB.search(question, TOP_K) if ensure_kb() else []
    context = format_context(results)
    if results:
        prompt = f"请基于以下教材片段回答学生的 GIS/虚拟地理环境实验操作问题。\n\n教材片段：\n{context}\n\n学生问题：{question}\n\n请输出：直接回答、分步操作、注意事项、对应章节。"
    else:
        prompt = f"当前未加载教材知识库，请结合 GIS 专业知识回答学生问题，并明确说明这是通用 GIS 建议，不是教材原文。\n\n学生问题：{question}\n\n请输出：直接回答、分步操作、注意事项。"
    messages = [
        {"role": "system", "content": SYSTEM_STYLE},
        {"role": "user", "content": prompt},
    ]
    answer = call_llm(messages)
    if not answer:
        answer = "当前大模型服务暂不可用。请稍后重试，或补充所用 GIS 软件、数据格式和具体报错信息。"
    citations = "\n\n".join([f"{i}. {r['heading']}（{r['source']}，相关度 {r['score']:.3f}）" for i, r in enumerate(results, 1)])
    if not results:
        citations = "教材知识库尚未构建，当前回答基于通用 GIS 知识。"
    return answer, citations


def research_question_helper(question: str) -> str:
    context = SIX_REAL_FRAMEWORK
    messages = [
        {"role": "system", "content": SYSTEM_STYLE},
        {"role": "user", "content": f"你是地理探究课程的问题建模助手。请判断学生问题是否适合做课程探究，并给出修改建议。\n\n课程框架：{context}\n\n学生研究问题：{question}\n\n请按以下结构输出：\n1. 是否可以探究：可以 / 需要收窄 / 暂不适合\n2. 判断理由：从真场景、真数据、真处理、真模拟、真应用、真评价角度简要判断\n3. 主要问题：指出研究对象、空间范围、时间范围、数据、方法或评价指标中的缺口\n4. 改写建议：给出 2-3 个更适合作为课程项目的问题表述\n5. 下一步行动：列出学生下一步要补充的信息。"},
    ]
    return call_llm(messages)


def process_evaluator(process_text: str) -> str:
    rubric = """
评分维度，每项 0-5 分，合计 25 分：
- 问题提出：研究问题是否清晰、可探究，是否有明确空间/时间/对象边界。
- 数据准备：数据类型、来源、尺度、格式、预处理思路是否合理。
- 方法选择：模型、GIS 操作或模拟方法是否与问题匹配，步骤是否可执行。
- 结论表达：是否基于结果给出清楚解释，是否避免超过数据支持范围。
- 反思改进：是否说明局限、误差来源、改进方向或进一步验证方法。
"""
    messages = [
        {"role": "system", "content": SYSTEM_STYLE},
        {"role": "user", "content": f"你是地理探究课程过程评价助教。请按评分细则评价学生提交的探究过程。\n\n{rubric}\n\n学生提交：\n{process_text}\n\n请输出一个 Markdown 表格，列为：维度、得分、反馈、改进建议。表格后给出总分、总体评价和优先改进项。评分要克制，不能全给高分。"},
    ]
    return call_llm(messages)


def rebuild_index() -> str:
    KB.build(force=True)
    return f"索引已重建。当前切分片段数：{len(KB.chunks)}；Embedding 后端：{KB.engine.backend}。"


def build_ui():
    css = """
    .gradio-container {max-width: 980px !important; margin: 0 auto !important; background: #f7fbfa;}
    .gm-header {display: flex; align-items: center; gap: 20px; padding: 24px 8px 16px;}
    .gm-logo {width: 72px; height: 72px; object-fit: contain;}
    .gm-title-group {display: flex; flex-direction: column; gap: 2px;}
    .gm-main-title {color: #1a5fa8; font-size: 26px; font-weight: 700; margin: 0;}
    .gm-sub-title {color: #5a7d9a; font-size: 14px; font-weight: 500; margin: 0;}
    .gm-card {background: #ffffff; border: 1px solid #d0dae2; border-radius: 18px; padding: 18px; box-shadow: 0 10px 24px rgba(26, 95, 168, 0.04);}
    .gm-footer {text-align: center; color: #a0aec0; font-size: 13px; padding: 24px 0 12px; margin-top: 10px;}
    button.primary {background: linear-gradient(135deg, #1a5fa8, #168aad) !important; border: none !important;}
    textarea, input {border-radius: 12px !important;}
    .tabs {border-radius: 16px !important;}
    """
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    with gr.Blocks(theme=theme, css=css, title="地理探究伴学智能体") as demo:
        gr.HTML("""
        <div class="gm-header">
          <img src="file/static/cug_logo.png" class="gm-logo" alt="CUG Logo">
          <div class="gm-title-group">
            <h1 class="gm-main-title">地理探究伴学智能体</h1>
            <p class="gm-sub-title">GeoMentor · 中国地质大学</p>
          </div>
        </div>
        """)
        with gr.Group(elem_classes=["gm-card"]):
            with gr.Tab("GIS 实验助教"):
                q = gr.Textbox(label="学生问题", placeholder="", lines=3)
                ask_btn = gr.Button("生成回答", variant="primary")
                ans = gr.Markdown(label="回答")
                with gr.Accordion("教材依据", open=False):
                    cite = gr.Textbox(label="命中的教材章节", lines=5)
                ask_btn.click(gis_tutor, inputs=q, outputs=[ans, cite])

            with gr.Tab("问题建模"):
                rq = gr.Textbox(label="研究问题", placeholder="", lines=3)
                rq_btn = gr.Button("给出建议", variant="primary")
                rq_out = gr.Markdown(label="建议")
                rq_btn.click(research_question_helper, inputs=rq, outputs=rq_out)

            with gr.Tab("过程评价"):
                proc = gr.Textbox(label="探究过程", placeholder="", lines=6)
                ev_btn = gr.Button("生成评价", variant="primary")
                ev_out = gr.Markdown(label="评价结果")
                ev_btn.click(process_evaluator, inputs=proc, outputs=ev_out)
        gr.HTML("<div class='gm-footer'>中国地质大学（北京）zcx@cugb.edu.cn</div>")
    return demo


if __name__ == "__main__":
    demo = build_ui()
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
