import sys
import os
import asyncio
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

BACKEND_DIR = ENV_PATH.parent
LOCAL_MCP_COMMAND = BACKEND_DIR / ".venv" / "bin" / "minimax-coding-plan-mcp"
MINIMAX_API_HOST = "https://api.minimaxi.com"

load_dotenv(ENV_PATH)

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")

class ImagesUnderstandState(TypedDict):
    user_uploaded_images_path: list[Path]
    desc: str

def create_mcp_server_params() -> StdioServerParameters:
    if not MINIMAX_API_KEY:
        raise RuntimeError("MINIMAX_API_KEY is not configured")

    return StdioServerParameters(
        command=str(LOCAL_MCP_COMMAND),
        env={
            "MINIMAX_API_KEY": MINIMAX_API_KEY,
            "MINIMAX_API_HOST": MINIMAX_API_HOST,
        },
    )

def extract_text_from_tool_result(result: CallToolResult) -> str:
    if getattr(result, "isError", False) or result.isError:
        raise RuntimeError(f"understand_image failed: {result}")

    text_parts = [block.text for block in result.content if isinstance(block, TextContent)]
    if text_parts:
        return "\n".join(text_parts)

    structured_content = getattr(result, "structuredContent", {}) or {}
    if isinstance(structured_content, dict):
        text = structured_content.get("text")
        if isinstance(text, str) and text:
            return text

    raise RuntimeError(f"understand_image returned no text content: {result}")

async def understand_images(image_paths: list[Path], prompt: str) -> str:
    server_params = create_mcp_server_params()
    descriptions: list[str] = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for index, image_path in enumerate(image_paths, start=1):
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                result = await session.call_tool(
                    "understand_image",
                    arguments={
                        "prompt": prompt,
                        "image_source": str(image_path.resolve()),
                    },
                )
                descriptions.append(f"Image {index}:\n{extract_text_from_tool_result(result)}")

    return "\n\n".join(descriptions)

def extract_exam_question_bbox(state: ImagesUnderstandState) -> dict:
    prompt = "描述一下这些图片中的内容。"

    combined_desc = asyncio.run(
        understand_images(state["user_uploaded_images_path"], prompt)
    )

    print(combined_desc)
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
    print(final_state)

if __name__ == "__main__":
    main()
