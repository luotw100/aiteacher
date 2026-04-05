import sys
import os
import base64
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

load_dotenv(ENV_PATH)

ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")
ALIYUN_BASE_URL = os.getenv("ALIYUN_BASE_URL")
ALIYUN_MODEL_NAME = os.getenv("ALIYUN_MODEL_NAME")

class ImagesUnderstandState(TypedDict):
    user_uploaded_images_path: list[Path]
    desc: str

def encode_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_exam_question_bbox(state: ImagesUnderstandState) -> dict:
    prompt = "描述一下这些图片中的内容。"

    content = [{"type": "text", "text": prompt}]

    for index, image_path in enumerate(state["user_uploaded_images_path"], start=1):
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        base64_image = encode_image_base64(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    llm = ChatOpenAI(
        api_key=ALIYUN_API_KEY,
        base_url=ALIYUN_BASE_URL,
        model=ALIYUN_MODEL_NAME,
        max_tokens=2048,
    )

    message = HumanMessage(content=content)
    response = llm.invoke([message])
    combined_desc = response.content

    print("Combined Description:\n", combined_desc)
    return {"desc": combined_desc}

def build_agent():
    graph = StateGraph(ImagesUnderstandState)
    graph.add_node("extract_exam_question_bbox", extract_exam_question_bbox)
    graph.add_edge(START, "extract_exam_question_bbox")
    graph.add_edge("extract_exam_question_bbox", END)
    return graph.compile()

agent = build_agent()

def main() -> None:
    final_state = agent.invoke({"user_uploaded_images_path": [Path(path) for path in sys.argv[1:]]})

if __name__ == "__main__":
    main()
