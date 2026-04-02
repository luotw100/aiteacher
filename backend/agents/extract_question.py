import sys
import os
import mimetypes
from pathlib import Path
from typing import TypedDict, Dict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
import base64

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

def encode_image_to_base64(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        raise ValueError(f"Unsupported image type for {image_path}: {mime_type}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

load_dotenv(ENV_PATH)

MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MODEL_NAME = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")
MAX_TOKENS = int(os.getenv("QUESTION_AGENT_MAX_TOKENS", "102400"))

class ExtractExamQuestionState(TypedDict):
    user_uploaded_images_path: list[Path]
    extracted_exam_question_bboxes: list[Dict]


def create_llm() -> ChatAnthropic:
    client_kwargs = {
        "model": MODEL_NAME,
        "base_url": MINIMAX_BASE_URL,
        "api_key": MINIMAX_API_KEY,
        "temperature": 1.0,
        "max_tokens": MAX_TOKENS,
        "max_retries": 0,
    }

    return ChatAnthropic(**client_kwargs)

def extract_exam_question_bbox(state: ExtractExamQuestionState) -> dict:
    llm = create_llm()

    request_content = []
    for image_path in state["user_uploaded_images_path"]:
        encoded_image = encode_image_to_base64(str(image_path))
        request_content.append({"type": "image_url", "image_url": {"url": encoded_image}})
    request_content.append({"type": "text", "text": "描述一下这些图片中的内容。"})

    response = llm.invoke([
        HumanMessage(content=request_content)
    ])

    final_answer = ""
    if isinstance(response.content, str):
        final_answer = response.content
    elif isinstance(response.content, list):
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                final_answer += block.get("text", "")
    print(f"{final_answer}")
    return {"extracted_exam_question_bboxes": [{"description": final_answer}]}

def build_agent():
    graph = StateGraph(ExtractExamQuestionState)
    graph.add_node("extract_exam_question_bbox", extract_exam_question_bbox)
    graph.add_edge(START, "extract_exam_question_bbox")
    graph.add_edge("extract_exam_question_bbox", END)
    return graph.compile()


agent = build_agent()


def main() -> None:
    final_state = agent.invoke({"user_uploaded_images_path": [Path(path) for path in sys.argv[1:]]})
    print(final_state)

if __name__ == "__main__":
    main()
