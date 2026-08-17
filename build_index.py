import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="构建 GeoMentor 教材检索索引")
    parser.add_argument("--textbook", required=True, help="教材 docx 路径")
    parser.add_argument("--backend", default="tfidf", choices=["tfidf", "sentence-transformers", "openai"])
    args = parser.parse_args()

    textbook = Path(args.textbook).resolve()
    if not textbook.exists():
        raise FileNotFoundError(f"未找到教材：{textbook}")

    os.environ["TEXTBOOK_PATH"] = str(textbook)
    os.environ["EMBEDDING_BACKEND"] = args.backend
    os.environ["REBUILD_INDEX"] = "1"

    import app

    print(f"索引构建完成：{len(app.KB.chunks)} 个片段")
    print(f"索引位置：{app.INDEX_DIR / 'kb.pkl'}")
    print(f"Embedding 后端：{app.KB.engine.backend}")


if __name__ == "__main__":
    main()
