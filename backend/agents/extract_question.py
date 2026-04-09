import argparse
import sys
import os
import json
import base64
import io
import binascii
import math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Any, TypedDict, Dict, List, Tuple

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rapidocr_onnxruntime import RapidOCR

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

load_dotenv(ENV_PATH)

ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")
ALIYUN_BASE_URL = os.getenv("ALIYUN_BASE_URL")
ALIYUN_MODEL_NAME = os.getenv("ALIYUN_MODEL_NAME")

JPEG_DATA_URI_PREFIX = "data:image/jpeg;base64,"
VALID_ROTATIONS = {0, 90, 180, 270}
OVERLAY_COLOR_PALETTE = [
    (230, 57, 70),
    (29, 78, 216),
    (46, 125, 50),
    (245, 124, 0),
    (126, 87, 194),
]

class ImagesUnderstandState(TypedDict):
    user_uploaded_images: List[str] # base64 encoded images from user upload
    correct_order_result: Dict[str, Any] # the parsed JSON result from the vision model about image validity, order and rotation
    ordered_images: List[str] # base64 encoded images after rotation correction
    ocr_results: List[Dict[str, Any]] # OCR results for each ordered page 
    question_extraction_result: Dict[str, Any] # the parsed JSON result for retained and dropped questions
    extracted_questions: List[Dict[str, Any]] # complete big questions with N boxes in absolute coordinates

CORRECT_ORDER_PROMPT = """
你是一个试卷图像预处理引擎。用户上传了若干张试卷的照片.
注意：单张图片里可能会露出相邻页面的一小部分残片。你只能根据主体页面来判断合法性、顺序和旋转角度，不要被边缘露出的相邻页残片干扰。主体页面通常是画面中占比最大的那一页，并且其页码在页面底部角落能清晰看到。
第一步:请确保这是合法的试卷照片上传,你需要检查图片的质量和内容是否满足以下要求:
1. 图片必须清晰可见,文字必须清晰可读。
2. 图片必须包含完整的试卷内容,不能有大面积的遮挡或残缺或反光或暗光。
3. 图片必须是正确的试卷照片,不能包含其他无关内容。

第二步:如果合格,请根据图片页码或上下文给出正确的阅读顺序,并判断每张图需要顺时针旋转多少度(0, 90, 180, 270)才能使文字正向朝上.
如果主体页面中的中文正文已经是正常竖直阅读方向，rotation 必须为 0。

严格只输出以下格式的JSON,不要有任何其他解释文字,你的输出第一个字符必须是"{"，最后一个字符必须是"}":
{
  "is_valid": true or false,
  "reject_reason": "如果不合格的原因,例如第x张图片模糊,第x张图片非法试卷照片等",
  "sorted_pages": [{"original_index": 0, "rotation": 90}]
}
"""

QUESTION_EXTRACTION_PROMPT = """
你是一个试卷题目定位引擎。
输入是若干张已经完成排序和方向矫正的试卷页面图片，它们已经是最终阅读顺序。每张图都会明确标出自己的 page_index。
此外，我还附带了每张图片的 OCR (光学字符识别) 结果及其在原图上的“绝对像素坐标”。

注意：由于是翻开的书本拍照，单张图片里可能会露出相邻页面的一小部分残片。你只能分析当前图片中的主体页面，不要分析边缘露出的相邻页残片。主体页面通常是画面中占比最大的那一页，并且其页码在页面底部角落能清晰看到。
注意：页面内部的真实阅读顺序要以题号推进关系和内容续接关系为准，不要机械按全局坐标从上到下排序。

你的任务是利用我附带的 OCR 结果和图片视觉信息，输出“完整大题”的大包围框坐标列表。规则如下：
1. 按大题识别，不按小题识别。比如 17.(1)(2) 只算一道题。
    - question_label 必须优先使用卷面上该大题明确可见的题号或稳定编号。
    - 如果题目本体完整，但没有明确可见的正式题号，优先使用题目开头紧邻题干的括号来源文字、年份地区信息、或第一段短语作为 question_label。
2. 一道大题可以对应 N 个方框：跨页或跨栏的题目应拆分多个 boxes。
    - 如果某道大题被一分为二，请合并为同一个 question，提供两个 boxes。
3. 只保留内容完整的大题。如果有题目明显在当前页没完全显示且无法从其它页接续，必须丢弃并在 dropped_questions 中说明。
4. 返回的 box 必须使用原图的绝对像素坐标。
    - 你应“严格参考属于同一题的 OCR 文本框”将它们的坐标外包围框计算出来（取最小的 x_min, y_min，最大的 x_max, y_max），并作为包围该部分内容的最终 box 返回。
    - 这个外包围框必须覆盖这道题的所有文本内容。
    - 输出时必须直接返回整数绝对像素坐标。
    - x1, y1 是左上角，x2, y2 是右下角，必须满足 0 <= x1 < x2，0 <= y1 < y2。
5. box 要尽量紧贴该大题在当前页面上的全部可见内容外接矩形。
    - 不要为了求稳而给出过大的宽松框。不要把大块空白边距、相邻题目一并框进去。

严格只输出以下 JSON 结构，不要有任何其他解释文字，你的输出第一个字符必须是 "{"，最后一个字符必须是 "}"：
{
    "questions": [
        {
            "question_label": "17",
            "boxes": [
                {"page_index": 0, "x1": 120, "y1": 180, "x2": 930, "y2": 820},
                {"page_index": 1, "x1": 60, "y1": 0, "x2": 940, "y2": 260}
            ]
        }
    ],
    "dropped_questions": [
        {"question_label": "18", "reason": "第1页底部只有半道题，缺少下一页内容"}
    ]
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


def create_vision_model(max_tokens: int) -> ChatOpenAI:
    ensure_model_config()
    return ChatOpenAI(
        api_key=ALIYUN_API_KEY,
        base_url=ALIYUN_BASE_URL,
        model=ALIYUN_MODEL_NAME,
        max_tokens=max_tokens,
        temperature=0.0,
    )


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


def decode_jpeg_data_uri_to_image(image_data_uri: str, image_index: int) -> Image.Image:
    image_bytes = decode_jpeg_data_uri(image_data_uri, image_index)
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.convert("RGBA")


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


def invoke_vision_json(
    prompt: str,
    image_inputs: List[str],
    index_label: str,
    response_label: str,
    max_tokens: int,
    per_image_texts: List[str] = None
) -> Dict[str, Any]:
    if not image_inputs:
        raise ValueError(f"{response_label}的图片输入为空")

    multimodal_messages = [{"type": "text", "text": prompt}]
    for idx, b64_str in enumerate(image_inputs):
        text_content = f"--- 这张图片的 {index_label} 是: {idx} ---\n"
        if per_image_texts and idx < len(per_image_texts):
            text_content += f"它的 OCR 绝对像素坐标和文本数据为（格式: [x1, y1, x2, y2], 文本内容）：\n{per_image_texts[idx]}\n"

        multimodal_messages.append({"type": "text", "text": text_content})
        multimodal_messages.append({
            "type": "image_url",
            "image_url": {"url": b64_str},
        })

    vision_model = create_vision_model(max_tokens=max_tokens)
    vision_response = vision_model.invoke([HumanMessage(content=multimodal_messages)])
    vision_response_text = normalize_json_text(extract_text_content(vision_response.content))

    try:
        parsed_result = json.loads(vision_response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"无法解析{response_label}为 JSON: {exc}\n模型原始输出: {vision_response_text}"
        ) from exc

    if not isinstance(parsed_result, dict):
        raise ValueError(f"{response_label}必须是 JSON 对象")

    return parsed_result


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


def normalize_question_label(raw_label: Any, fallback_prefix: str, index: int) -> str:
    if raw_label is None:
        return f"{fallback_prefix}_{index + 1}"

    if not isinstance(raw_label, str):
        raw_label = str(raw_label)

    normalized_label = raw_label.strip()
    if not normalized_label:
        return f"{fallback_prefix}_{index + 1}"

    return normalized_label


def normalize_coordinate_value(
    raw_value: Any,
    field_name: str,
    question_index: int,
    box_index: int,
) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(
            f"questions[{question_index}].boxes[{box_index}].{field_name} 必须是数字"
        )

    normalized_value = int(round(raw_value))
    return max(0, normalized_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="排序试卷图片并提取题目坐标")
    parser.add_argument("images", nargs="+", help="输入的 JPEG 图片路径")
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="若提供则把题目框叠加后的页面图导出到该目录",
    )
    return parser.parse_args()


def normalize_to_pixel_coordinates(box: Dict[str, int], width: int, height: int) -> Tuple[int, int, int, int]:
    left = max(0, min(width - 1, box.get("x1", 0)))
    top = max(0, min(height - 1, box.get("y1", 0)))
    right = max(0, min(width - 1, box.get("x2", width - 1)))
    bottom = max(0, min(height - 1, box.get("y2", height - 1)))

    if right <= left:
        right = min(width - 1, left + 1)
    if bottom <= top:
        bottom = min(height - 1, top + 1)

    return left, top, right, bottom


def load_overlay_font(font_size: int) -> ImageFont.ImageFont:
    candidate_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for candidate_path in candidate_paths:
        if Path(candidate_path).is_file():
            try:
                return ImageFont.truetype(candidate_path, font_size)
            except OSError:
                continue

    return ImageFont.load_default()


def build_overlay_output_name(
    page_index: int,
    img_paths: List[Path],
    correct_order_result: Dict[str, Any],
) -> str:
    sorted_pages = correct_order_result.get("sorted_pages", [])
    if page_index < len(sorted_pages):
        original_index = sorted_pages[page_index].get("original_index")
        if isinstance(original_index, int) and 0 <= original_index < len(img_paths):
            stem = img_paths[original_index].stem
            return f"page_{page_index + 1:02d}_{stem}_overlay.jpg"

    return f"page_{page_index + 1:02d}_overlay.jpg"


def export_question_overlays(
    output_dir: Path,
    ordered_images: List[str],
    extracted_questions: List[Dict[str, Any]],
    img_paths: List[Path],
    correct_order_result: Dict[str, Any],
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_page_images = [
        decode_jpeg_data_uri_to_image(image_data_uri, page_index)
        for page_index, image_data_uri in enumerate(ordered_images)
    ]
    page_overlays = [Image.new("RGBA", image.size, (0, 0, 0, 0)) for image in ordered_page_images]
    border_drawers = [ImageDraw.Draw(overlay, "RGBA") for overlay in page_overlays]

    base_width = ordered_page_images[0].width if ordered_page_images else 0
    font = load_overlay_font(max(18, base_width // 48 if base_width else 18))
    border_width = max(5, base_width // 220 if base_width else 5)
    text_padding = max(8, border_width + 2)

    for question_index, question in enumerate(extracted_questions):
        color = OVERLAY_COLOR_PALETTE[question_index % len(OVERLAY_COLOR_PALETTE)]
        fill_color = (*color, 72)
        border_color = (*color, 255)
        text_fill = (255, 255, 255, 255)
        text_background = (*color, 230)

        boxes = question.get("boxes", [])
        total_boxes = len(boxes)
        question_label = str(question.get("question_label", question.get("question_id", question_index + 1)))
        for box_index, box in enumerate(boxes):
            page_index = box["page_index"]
            image = ordered_page_images[page_index]
            drawer = border_drawers[page_index]
            left, top, right, bottom = normalize_to_pixel_coordinates(box, image.width, image.height)

            drawer.rectangle(
                [(left, top), (right, bottom)],
                fill=fill_color,
                outline=border_color,
                width=border_width,
            )

            label_text = f"题目:{question_label}\n{box_index + 1}/{total_boxes}"
            text_bbox = drawer.multiline_textbbox((0, 0), label_text, font=font, spacing=4)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_left = left + border_width
            text_top = max(0, top - text_height - text_padding * 2)
            if text_top == 0 and top + text_height + text_padding * 2 < image.height:
                text_top = top + border_width

            text_right = min(image.width - 1, text_left + text_width + text_padding * 2)
            text_bottom = min(image.height - 1, text_top + text_height + text_padding * 2)

            if text_right - text_left < text_width + text_padding * 2:
                text_left = max(0, image.width - (text_width + text_padding * 2))
                text_right = min(image.width - 1, text_left + text_width + text_padding * 2)

            drawer.rectangle(
                [(text_left, text_top), (text_right, text_bottom)],
                fill=text_background,
            )
            drawer.multiline_text(
                (text_left + text_padding, text_top + text_padding),
                label_text,
                font=font,
                fill=text_fill,
                spacing=4,
            )

    saved_files = []
    for page_index, image in enumerate(ordered_page_images):
        composited_image = Image.alpha_composite(image, page_overlays[page_index]).convert("RGB")
        output_path = output_dir / build_overlay_output_name(page_index, img_paths, correct_order_result)
        composited_image.save(output_path, format="JPEG", quality=95)
        saved_files.append(output_path)

    return saved_files



def export_ocr_overlays(
    output_dir: Path,
    ordered_images: List[str],
    ocr_results: List[Dict[str, Any]],
    img_paths: List[Path],
    correct_order_result: Dict[str, Any],
) -> List[Path]:
    import re
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_page_images = [
        decode_jpeg_data_uri_to_image(image_data_uri, page_index)
        for page_index, image_data_uri in enumerate(ordered_images)
    ]
    page_overlays = [Image.new("RGBA", image.size, (0, 0, 0, 0)) for image in ordered_page_images]
    border_drawers = [ImageDraw.Draw(overlay, "RGBA") for overlay in page_overlays]

    base_width = ordered_page_images[0].width if ordered_page_images else 0
    font = load_overlay_font(max(14, base_width // 60 if base_width else 14))
    border_width = max(2, base_width // 400 if base_width else 2)

    for ocr_res in ocr_results:
        page_index = ocr_res.get("page_index", 0)
        drawer = border_drawers[page_index]
        image = ordered_page_images[page_index]
        
        for line in ocr_res.get("ocr_lines", []):
            match = re.match(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],(.*)", line)
            if match:
                x1, y1, x2, y2 = map(int, match.groups()[:4])
                text = match.group(5).strip()
                
                drawer.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline=(0, 255, 0, 200),
                    width=border_width,
                )
                
                # Draw text loosely
                text_bbox = drawer.textbbox((0, 0), text, font=font)
                text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                drawer.rectangle(
                    [(x1, max(0, y1 - text_h - 4)), (x1 + text_w, max(0, y1))],
                    fill=(0, 255, 0, 180)
                )
                drawer.text((x1, max(0, y1 - text_h - 4)), text, font=font, fill=(0, 0, 0, 255))

    saved_files = []
    for page_index, image in enumerate(ordered_page_images):
        composited_image = Image.alpha_composite(image, page_overlays[page_index]).convert("RGB")
        original_name = build_overlay_output_name(page_index, img_paths, correct_order_result)
        ocr_name = original_name.replace("_overlay.jpg", "_OCR.jpg")
        output_path = output_dir / ocr_name
        composited_image.save(output_path, format="JPEG", quality=95)
        saved_files.append(output_path)

    return saved_files


def normalize_question_box(
    raw_box: Any,
    total_pages: int,
    question_index: int,
    box_index: int,
) -> Dict[str, int]:
    if not isinstance(raw_box, dict):
        raise ValueError(f"questions[{question_index}].boxes[{box_index}] 必须是对象")

    page_index = raw_box.get("page_index")
    if isinstance(page_index, bool) or not isinstance(page_index, int):
        raise ValueError(f"questions[{question_index}].boxes[{box_index}].page_index 必须是整数")
    if not 0 <= page_index < total_pages:
        raise ValueError(f"questions[{question_index}].boxes[{box_index}].page_index 超出范围")

    x1 = normalize_coordinate_value(raw_box.get("x1"), "x1", question_index, box_index)
    y1 = normalize_coordinate_value(raw_box.get("y1"), "y1", question_index, box_index)
    x2 = normalize_coordinate_value(raw_box.get("x2"), "x2", question_index, box_index)
    y2 = normalize_coordinate_value(raw_box.get("y2"), "y2", question_index, box_index)

    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"questions[{question_index}].boxes[{box_index}] 的坐标必须满足 x1 < x2 且 y1 < y2"
        )

    return {
        "page_index": page_index,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def normalize_dropped_questions(raw_dropped_questions: Any) -> List[Dict[str, str]]:
    if raw_dropped_questions is None:
        return []

    if not isinstance(raw_dropped_questions, list):
        raise ValueError("模型输出中的 dropped_questions 必须是数组")

    normalized_dropped_questions = []
    for dropped_index, raw_dropped_question in enumerate(raw_dropped_questions):
        if isinstance(raw_dropped_question, str):
            normalized_dropped_questions.append({
                "question_label": f"unknown_{dropped_index + 1}",
                "reason": raw_dropped_question.strip() or "模型判定为内容不全",
            })
            continue

        if not isinstance(raw_dropped_question, dict):
            raise ValueError(f"dropped_questions[{dropped_index}] 必须是对象或字符串")

        question_label = normalize_question_label(
            raw_dropped_question.get("question_label"),
            "unknown",
            dropped_index,
        )
        reason = raw_dropped_question.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "模型判定为内容不全"

        normalized_dropped_questions.append({
            "question_label": question_label,
            "reason": reason.strip(),
        })

    return normalized_dropped_questions


def normalize_question_extraction_result(
    parsed_question_result: Dict[str, Any],
    total_pages: int,
) -> Dict[str, Any]:
    raw_questions = parsed_question_result.get("questions", [])
    if not isinstance(raw_questions, list):
        raise ValueError("模型输出中的 questions 必须是数组")

    normalized_questions = []
    seen_question_signatures = set()
    for question_index, raw_question in enumerate(raw_questions):
        if not isinstance(raw_question, dict):
            raise ValueError(f"questions[{question_index}] 必须是对象")

        raw_boxes = raw_question.get("boxes", [])
        if not isinstance(raw_boxes, list) or not raw_boxes:
            raise ValueError(f"questions[{question_index}].boxes 必须是非空数组")

        normalized_boxes = []
        seen_boxes = set()
        for box_index, raw_box in enumerate(raw_boxes):
            normalized_box = normalize_question_box(raw_box, total_pages, question_index, box_index)
            box_signature = (
                normalized_box["page_index"],
                normalized_box["x1"],
                normalized_box["y1"],
                normalized_box["x2"],
                normalized_box["y2"],
            )
            if box_signature in seen_boxes:
                raise ValueError(f"questions[{question_index}] 出现重复 box")
            seen_boxes.add(box_signature)
            normalized_boxes.append(normalized_box)

        question_label = normalize_question_label(
            raw_question.get("question_label"),
            "question",
            question_index,
        )
        page_indexes = list(dict.fromkeys(box["page_index"] for box in normalized_boxes))

        question_signature = (
            question_label,
            tuple(
                (
                    box["page_index"],
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                )
                for box in normalized_boxes
            ),
        )
        if question_signature in seen_question_signatures:
            raise ValueError("模型输出中出现重复 question")
        seen_question_signatures.add(question_signature)

        normalized_questions.append({
            "question_label": question_label,
            "page_indexes": page_indexes,
            "boxes": normalized_boxes,
        })

    extracted_questions = []
    for question_index, normalized_question in enumerate(normalized_questions):
        extracted_questions.append({
            "question_id": f"question_{question_index + 1}",
            **normalized_question,
        })

    dropped_questions = normalize_dropped_questions(parsed_question_result.get("dropped_questions", []))
    total_boxes = sum(len(question["boxes"]) for question in extracted_questions)

    return {
        "extracted_questions": extracted_questions,
        "question_extraction_result": {
            "coordinate_system": "absolute_pixels",
            "ordered_page_count": total_pages,
            "question_count": len(extracted_questions),
            "box_count": total_boxes,
            "dropped_questions": dropped_questions,
        },
    }


def build_empty_question_result(ordered_page_count: int, skipped_reason: str = "") -> Dict[str, Any]:
    question_extraction_result: Dict[str, Any] = {
        "coordinate_system": "absolute_pixels",
        "ordered_page_count": ordered_page_count,
        "question_count": 0,
        "box_count": 0,
        "dropped_questions": [],
    }
    if skipped_reason:
        question_extraction_result["skipped_reason"] = skipped_reason

    return {
        "extracted_questions": [],
        "question_extraction_result": question_extraction_result,
    }

def correct_order(state: ImagesUnderstandState) -> Dict:
    image_inputs = state.get("user_uploaded_images", [])
    if not image_inputs:
        raise ValueError("user_uploaded_images is empty")

    image_bytes_list = []
    for idx, b64_str in enumerate(image_inputs):
        image_bytes_list.append(decode_jpeg_data_uri(b64_str, idx))
    parsed_order_result = invoke_vision_json(
        prompt=CORRECT_ORDER_PROMPT,
        image_inputs=image_inputs,
        index_label="original_index",
        response_label="排序结果",
        max_tokens=2048,
    )
    parsed_order_result = normalize_order_result(parsed_order_result, len(image_inputs))

    processed_b64_list = []
    if parsed_order_result.get("is_valid"):
        for page in parsed_order_result.get("sorted_pages", []):
            orig_index = page["original_index"]
            rotation = page["rotation"]
            image_bytes = image_bytes_list[orig_index]

            with Image.open(io.BytesIO(image_bytes)) as img:
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

def perform_ocr(state: ImagesUnderstandState) -> Dict:
    ordered_images = state.get("ordered_images", [])
    if not ordered_images:
        return {"ocr_results": []}

    ocr = RapidOCR() 
    
    ocr_results = []
    print(f"\n[OCR] 正在对 {len(ordered_images)} 张图片进行 OCR 文字提取...")
    for idx, ordered_image in enumerate(ordered_images):
        img_bytes = decode_jpeg_data_uri(ordered_image, idx)
        
        result, elapse = ocr(img_bytes)
        page_texts = []
        if result:
            for box, text, score in result:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                page_texts.append(f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], {text}")
                
        ocr_results.append({"page_index": idx, "ocr_lines": page_texts})

    return {"ocr_results": ocr_results}

def extract_complete_questions(state: ImagesUnderstandState) -> Dict:
    correct_order_result = state.get("correct_order_result", {})
    ordered_images = state.get("ordered_images", [])
    ocr_results = state.get("ocr_results", [])

    if not correct_order_result.get("is_valid"):
        return build_empty_question_result(
            ordered_page_count=len(ordered_images),
            skipped_reason="correct_order_result.is_valid 为 false，跳过题目提取",
        )

    if not ordered_images:
        raise ValueError("ordered_images is empty")

    for page_index, ordered_image in enumerate(ordered_images):
        decode_jpeg_data_uri(ordered_image, page_index)

    per_image_texts = []
    for ocr_res in ocr_results:
        # 裁剪一定长度，避免极端情况超过长上下文
        per_image_texts.append("\n".join(ocr_res.get("ocr_lines", []))[:15000])

    parsed_question_result = invoke_vision_json(
        prompt=QUESTION_EXTRACTION_PROMPT,
        image_inputs=ordered_images,
        index_label="page_index",
        response_label="题目坐标结果",
        max_tokens=4096,
        per_image_texts=per_image_texts
    )

    return normalize_question_extraction_result(parsed_question_result, len(ordered_images))

def build_agent():
    graph = StateGraph(ImagesUnderstandState)
    graph.add_node("correct_order", correct_order)
    graph.add_node("perform_ocr", perform_ocr)
    graph.add_node("extract_complete_questions", extract_complete_questions)
    
    graph.add_edge(START, "correct_order")
    graph.add_edge("correct_order", "perform_ocr")
    graph.add_edge("perform_ocr", "extract_complete_questions")
    graph.add_edge("extract_complete_questions", END)
    return graph.compile()

workflow_agent = build_agent()

def main() -> None:
    args = parse_args()
    img_paths = [Path(path).expanduser() for path in args.images]
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None

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

    print("\n题目坐标提取结果：")
    print(json.dumps(final_output.get("question_extraction_result", {}), indent=2, ensure_ascii=False))

    print("\n完整题目坐标列表：")
    print(json.dumps(final_output.get("extracted_questions", []), indent=2, ensure_ascii=False))

    if output_dir is not None:
        ocr_paths = export_ocr_overlays(
            output_dir=output_dir,
            ordered_images=final_output.get("ordered_images", []),
            ocr_results=final_output.get("ocr_results", []),
            img_paths=img_paths,
            correct_order_result=final_output.get("correct_order_result", {}),
        )
        print("\n已导出OCR探测叠加图：")
        for p in ocr_paths:
            print(str(p))
            
        overlay_paths = export_question_overlays(
            output_dir=output_dir,
            ordered_images=final_output.get("ordered_images", []),
            extracted_questions=final_output.get("extracted_questions", []),
            img_paths=img_paths,
            correct_order_result=final_output.get("correct_order_result", {}),
        )
        print("\n已导出题目框叠加图：")
        for overlay_path in overlay_paths:
            print(str(overlay_path))

if __name__ == "__main__":
    main()
