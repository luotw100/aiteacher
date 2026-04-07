import sys
import os
import json
import base64
import mimetypes
import io
from PIL import Image
from pathlib import Path
from typing import TypedDict, Optional, Dict, List

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

def correct_order(state: ImagesUnderstandState) -> Dict:
    multimodal_messages = [{"type": "text", "text": CORRECT_ORDER_PROMPT}]

    image_inputs = state.get("user_uploaded_images", [])
    if not image_inputs:
        raise ValueError("user_uploaded_images is empty")

    for idx, b64_str in enumerate(image_inputs):
        if not b64_str.startswith("data:image/jpeg;base64,"):
            raise ValueError(f"第 {idx} 张图片的 Base64 字符串格式不正确，必须以 'data:image/jpeg;base64,' 开头")

        # 为了让视觉模型明确感知原始索引，打上文本标签
        multimodal_messages.append({"type": "text", "text": f"--- 这张图片的 original_index 是: {idx} ---"})
        multimodal_messages.append({
            "type": "image_url",
            "image_url": {
                "url": b64_str
            }
        })

    vision_model = ChatOpenAI(
        api_key=ALIYUN_API_KEY,
        base_url=ALIYUN_BASE_URL,
        model=ALIYUN_MODEL_NAME,
        max_tokens=2048,
        temperature=0.1,  # 降低温度使 JSON 输出更稳定
    )

    human_req = HumanMessage(content=multimodal_messages)
    vision_response = vision_model.invoke([human_req])
    vision_response = vision_response.content.strip()

    # 将模型输出的 JSON 字符串解析为 Python 字典
    try:
        parsed_order_result = json.loads(vision_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析模型输出为 JSON: {e}\n模型原始输出: {vision_response}")

    # 基于解析结果加载并旋转图像转为 Base64 存于内存
    processed_b64_list = []
    if parsed_order_result.get("is_valid"):
        for page in parsed_order_result.get("sorted_pages", []):
            try:
                orig_index = page.get("original_index")
                rotation = page.get("rotation", 0)

                orig_b64_str = image_inputs[orig_index]

                # 如果有 data: URI 前缀，则去掉以便进行base64解码给Pillow
                if orig_b64_str.startswith("data:image/"):
                    b64_data_only = orig_b64_str.split(",", 1)[-1]
                else:
                    b64_data_only = orig_b64_str

                image_bytes = base64.b64decode(b64_data_only)

                with Image.open(io.BytesIO(image_bytes)) as img:
                    # 提示词为“顺时针旋转多少度使文字正向”，而 Pillow rotate 是逆时针，因此转 -rotation 度（expand=True 保证宽高互换后不裁剪）
                    rotated_img = img.rotate(-rotation, expand=True)

                    if rotated_img.mode in ("RGBA", "P"):
                        rotated_img = rotated_img.convert("RGB")

                    buffered = io.BytesIO()
                    rotated_img.save(buffered, format="JPEG", quality=90)
                    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    processed_b64_list.append(f"data:image/jpeg;base64,{b64_str}")
            except Exception as e:
                print(f"处理带有 original_index: {orig_index} 的图片时发生异常: {e}")

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

    print(f"正在处理 {len(img_paths)} 张图片，请稍候...")
    final_output = workflow_agent.invoke({"user_uploaded_images_path": img_paths})

    print("\n大模型识别及排序结果：")
    print(json.dumps(final_output.get("correct_order_result", {}), indent=2, ensure_ascii=False))

    processed_b64 = final_output.get("processed_images_base64", [])
    print(f"\n已成功生成并在内存中存入了 {len(processed_b64)} 张经过排序和旋转修正的 Base64 格式图片。")

if __name__ == "__main__":
    main()
