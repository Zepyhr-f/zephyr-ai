import argparse

import httpx


DEFAULT_MESSAGE = "Zephyr AI 使用什么向量数据库？"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the RAG chat API.")
    parser.add_argument("--url", default="http://localhost:8000/api/v1/rag/chat")
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--conversation-id")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def answer_preview(answer: str, limit: int = 160) -> str:
    normalized = answer.replace("\n", " ").strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def build_payload(args: argparse.Namespace) -> dict:
    payload = {
        "message": args.message,
        "top_k": args.top_k,
    }
    if args.conversation_id:
        payload["conversation_id"] = args.conversation_id
    return payload


def print_summary(status_code: int, data: dict) -> None:
    sources = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(sources, list):
        sources = []

    print(f"status_code={status_code}")
    print(f"conversation_id={data.get('conversation_id') if isinstance(data, dict) else None}")
    print(f"answer_preview={answer_preview(data.get('answer', '')) if isinstance(data, dict) else ''}")
    print(f"sources_count={len(sources)}")

    if sources and isinstance(sources[0], dict):
        source = sources[0]
        print(f"top_source_title={source.get('title')}")
        print(f"top_source_file={source.get('file_path')}")
        print(f"top_source_header={source.get('header_path')}")
        print(f"top_source_score={source.get('score')}")


def main() -> None:
    args = parse_args()
    response = httpx.post(args.url, json=build_payload(args), timeout=60.0)
    try:
        data = response.json()
    except ValueError:
        data = {}
    print_summary(response.status_code, data)


if __name__ == "__main__":
    main()
