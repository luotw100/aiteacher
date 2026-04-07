import sys
import os
import json
import base64
import io
import binascii
from PIL import Image
from pathlib import Path
from typing import Any, TypedDict, Dict, List

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

load_dotenv(ENV_PATH)

ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")
ALIYUN_BASE_URL = os.getenv("ALIYUN_BASE_URL")
ALIYUN_MODEL_NAME = os.getenv("ALIYUN_MODEL_NAME")

JPEG_DATA_URI_PREFIX = "data:image/jpeg;base64,"
VALID_ROTATIONS = {0, 90, 180, 270}

class ImagesUnderstandState(TypedDict):
    user_uploaded_images: List[str] # base64 encoded images from user upload
    correct_order_result: Dict # the parsed JSON result from the vision model about image validity, order and rotation
    ordered_images: List[str] # base64 encoded images after rotation correction

CORRECT_ORDER_PROMPT = """
你是一个试卷图像预处理引擎。用户上传了若干张试卷的照片.
第一步:请确保这是合法的试卷照片上传,你需要检查图片的质量和内容是否满足以下要求:
1. 图片必须清晰可见,文字必须清晰可读。
2. 图片必须包含完整的试卷内容,不能有大面积的遮挡或残缺或反光或暗光。
3. 图片必须是正确的试卷照片,不能包含其他无关内容。

第二步:如果合格,请根据图片页码或上下文给出正确的阅读顺序,并判断每张图需要顺时针旋转多少度(0, 90, 180, 270)才能使文字正向朝上.

严格只输出以下格式的JSON,不要有任何其他解释文字,你的输出第一个字符必须是"{"，最后一个字符必须是"}":
{
  "is_valid": true or false,
  "reject_reason": "如果不合格的原因,例如第x张图片模糊,第x张图片非法试卷照片等",
  "sorted_pages": [{"original_index": 0, "rotation": 90}]
}
"""


def ensure_model_config() -> None:
    missing_env_vars = [
        name
        for name, value in {
            "ALIYUN_API_KEY": ALIYUN_API_KEY,
            "ALIYUN_BASE_URL": ALIYUN_BASE_URL,
            "ALIYUN_MODEL_NAME": ALIYUN_MODEL_NAME,
        }.items()
        if not value
    ]
    if missing_env_vars:
        missing = ", ".join(missing_env_vars)
        raise RuntimeError(f"缺少环境变量配置: {missing}")


def encode_jpeg_file_to_data_uri(image_path: Path) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    with Image.open(image_path) as img:
        image_format = img.format
        img.verify()

    if image_format != "JPEG":
        raise ValueError(f"只支持 JPEG 图片输入，当前文件不是 JPEG: {image_path}")

    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"{JPEG_DATA_URI_PREFIX}{encoded}"


def decode_jpeg_data_uri(image_data_uri: str, image_index: int) -> bytes:
    if not image_data_uri.startswith(JPEG_DATA_URI_PREFIX):
        raise ValueError(
            f"第 {image_index} 张图片的 Base64 字符串格式不正确，必须以 '{JPEG_DATA_URI_PREFIX}' 开头"
        )

    try:
        image_bytes = base64.b64decode(
            image_data_uri.removeprefix(JPEG_DATA_URI_PREFIX),
            validate=True,
        )
    except binascii.Error as exc:
        raise ValueError(f"第 {image_index} 张图片不是合法的 Base64 JPEG 数据: {exc}") from exc

    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise ValueError(f"第 {image_index} 张图片虽然带有 JPEG 前缀，但实际内容不是 JPEG")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.format != "JPEG":
                raise ValueError(f"第 {image_index} 张图片实际格式不是 JPEG")
            img.verify()
    except Exception as exc:
        raise ValueError(f"第 {image_index} 张图片无法被解析为有效 JPEG: {exc}") from exc

    return image_bytes


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        merged_text = "".join(text_parts).strip()
        if merged_text:
            return merged_text

    raise ValueError("视觉模型未返回可解析的文本 JSON")


def normalize_json_text(raw_text: str) -> str:
    stripped_text = raw_text.strip()

    if stripped_text.startswith("```"):
        lines = stripped_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped_text = "\n".join(lines).strip()
        if stripped_text.lower().startswith("json"):
            stripped_text = stripped_text[4:].lstrip()

    json_start = stripped_text.find("{")
    json_end = stripped_text.rfind("}")
    if json_start != -1 and json_end != -1 and json_start < json_end:
        return stripped_text[json_start : json_end + 1]

    return stripped_text


def normalize_order_result(parsed_order_result: Dict[str, Any], total_images: int) -> Dict[str, Any]:
    is_valid = parsed_order_result.get("is_valid")
    if not isinstance(is_valid, bool):
        raise ValueError("模型输出中的 is_valid 必须是布尔值")

    reject_reason = parsed_order_result.get("reject_reason", "")
    if not isinstance(reject_reason, str):
        raise ValueError("模型输出中的 reject_reason 必须是字符串")

    sorted_pages = parsed_order_result.get("sorted_pages", [])
    if not isinstance(sorted_pages, list):
        raise ValueError("模型输出中的 sorted_pages 必须是数组")

    normalized_pages = []
    seen_indexes = set()
    for page_index, page in enumerate(sorted_pages):
        if not isinstance(page, dict):
            raise ValueError(f"sorted_pages[{page_index}] 必须是对象")

        original_index = page.get("original_index")
        rotation = page.get("rotation", 0)
        if not isinstance(original_index, int) or not 0 <= original_index < total_images:
            raise ValueError(f"sorted_pages[{page_index}].original_index 超出范围")
        if original_index in seen_indexes:
            raise ValueError(f"sorted_pages[{page_index}] 出现重复 original_index: {original_index}")
        if rotation not in VALID_ROTATIONS:
            raise ValueError(f"sorted_pages[{page_index}].rotation 必须是 0/90/180/270")

        seen_indexes.add(original_index)
        normalized_pages.append({"original_index": original_index, "rotation": rotation})

    if is_valid and len(normalized_pages) != total_images:
        raise ValueError("模型输出的 sorted_pages 数量必须与输入图片数量一致")

    return {
        "is_valid": is_valid,
        "reject_reason": reject_reason,
        "sorted_pages": normalized_pages,
    }

def correct_order(state: ImagesUnderstandState) -> Dict:
    multimodal_messages = [{"type": "text", "text": CORRECT_ORDER_PROMPT}]

    image_inputs = state.get("user_uploaded_images", [])
    if not image_inputs:
        raise ValueError("user_uploaded_images is empty")

    image_bytes_list = []
    for idx, b64_str in enumerate(image_inputs):
        image_bytes_list.append(decode_jpeg_data_uri(b64_str, idx))

        # 为了让视觉模型明确感知原始索引，打上文本标签
        multimodal_messages.append({"type": "text", "text": f"--- 这张图片的 original_index 是: {idx} ---"})
        multimodal_messages.append({
            "type": "image_url",
            "image_url": {
                "url": b64_str
            }
        })

    ensure_model_config()
    vision_model = ChatOpenAI(
        api_key=ALIYUN_API_KEY,
        base_url=ALIYUN_BASE_URL,
        model=ALIYUN_MODEL_NAME,
        max_tokens=2048,
        temperature=0.1,  # 降低温度使 JSON 输出更稳定
    )

    human_req = HumanMessage(content=multimodal_messages)
    vision_response = vision_model.invoke([human_req])
    vision_response_text = normalize_json_text(extract_text_content(vision_response.content))

    # 将模型输出的 JSON 字符串解析为 Python 字典
    try:
        parsed_order_result = json.loads(vision_response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析模型输出为 JSON: {e}\n模型原始输出: {vision_response_text}") from e

    parsed_order_result = normalize_order_result(parsed_order_result, len(image_inputs))

    # 基于解析结果加载并旋转图像转为 Base64 存于内存
    processed_b64_list = []
    if parsed_order_result.get("is_valid"):
        for page in parsed_order_result.get("sorted_pages", []):
            orig_index = page["original_index"]
            rotation = page["rotation"]
            image_bytes = image_bytes_list[orig_index]

            with Image.open(io.BytesIO(image_bytes)) as img:
                # 提示词为“顺时针旋转多少度使文字正向”，而 Pillow rotate 是逆时针，因此转 -rotation 度（expand=True 保证宽高互换后不裁剪）
                rotated_img = img.rotate(-rotation, expand=True)

                if rotated_img.mode in ("RGBA", "P"):
                    rotated_img = rotated_img.convert("RGB")

                buffered = io.BytesIO()
                rotated_img.save(buffered, format="JPEG", quality=90)
                b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                processed_b64_list.append(f"{JPEG_DATA_URI_PREFIX}{b64_str}")

    return {
        "correct_order_result": parsed_order_result,
        "ordered_images": processed_b64_list
    }

def build_agent():
    graph = StateGraph(ImagesUnderstandState)
    graph.add_node("correct_order", correct_order)
    graph.add_edge(START, "correct_order")
    graph.add_edge("correct_order", END)
    return graph.compile()

workflow_agent = build_agent()

def main() -> None:
    img_paths = [Path(path) for path in sys.argv[1:]]
    if not img_paths:
        print("用法: python extract_question.py <图片路径1> <图片路径2> ...")
        sys.exit(1)

    try:
        user_uploaded_images = [encode_jpeg_file_to_data_uri(path) for path in img_paths]
        print(f"正在处理 {len(img_paths)} 张图片，请稍候...")
        final_output = workflow_agent.invoke({"user_uploaded_images": user_uploaded_images})
    except Exception as exc:
        print(f"处理失败: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n大模型识别及排序结果：")
    print(json.dumps(final_output.get("correct_order_result", {}), indent=2, ensure_ascii=False))

    processed_b64 = final_output.get("ordered_images", [])
    print(f"\n已成功生成并在内存中存入了 {len(processed_b64)} 张经过排序和旋转修正的 Base64 格式图片。")

if __name__ == "__main__":
    main()
