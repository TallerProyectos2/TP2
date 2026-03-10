from __future__ import annotations

from pathlib import Path
import json
import os
import socket
from typing import Any
from urllib.parse import urlparse

import cv2
from inference_sdk import InferenceHTTPClient

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = Path(os.getenv("TP2_TEST_IMAGE", BASE_DIR / "test.jpg")).expanduser().resolve()
OUTPUT_IMAGE_PATH = Path(
    os.getenv("TP2_OUTPUT_IMAGE", BASE_DIR / f"{IMAGE_PATH.stem}_pred{IMAGE_PATH.suffix}")
).expanduser().resolve()

MODE = os.getenv("TP2_INFERENCE_MODE", "local").strip().lower()
LOCAL_API_URL = os.getenv("ROBOFLOW_LOCAL_API_URL", "http://127.0.0.1:9001").strip()
CLOUD_WORKFLOW_API_URL = os.getenv(
    "ROBOFLOW_CLOUD_WORKFLOW_API_URL", "https://serverless.roboflow.com"
).strip()
CLOUD_MODEL_API_URL = os.getenv(
    "ROBOFLOW_CLOUD_MODEL_API_URL", "https://detect.roboflow.com"
).strip()
API_KEY = os.getenv("ROBOFLOW_API_KEY", "0tD81VP5ij7wTYq6W7yA").strip()

WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "1-v8mk1").strip()
WORKFLOW = os.getenv("ROBOFLOW_WORKFLOW", "custom-workflow-2").strip()
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "").strip()

TARGET = os.getenv("TP2_INFERENCE_TARGET", "").strip().lower()
if TARGET not in {"workflow", "model"}:
    TARGET = "model" if MODEL_ID else "workflow"


def extract_predictions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        merged: list[dict[str, Any]] = []
        for item in payload:
            merged.extend(extract_predictions(item))
        return merged

    if not isinstance(payload, dict):
        return []

    top_predictions = payload.get("predictions")
    if isinstance(top_predictions, list):
        return top_predictions

    if isinstance(top_predictions, dict):
        nested_predictions = top_predictions.get("predictions")
        if isinstance(nested_predictions, list):
            return nested_predictions

    return []


def draw_predictions(image_path: Path, output_path: Path, predictions: list[dict[str, Any]]):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    img_h, img_w = image.shape[:2]

    for prediction in predictions:
        x = prediction.get("x")
        y = prediction.get("y")
        w = prediction.get("width")
        h = prediction.get("height")
        if None in (x, y, w, h):
            continue

        x1 = max(0, int(round(x - w / 2)))
        y1 = max(0, int(round(y - h / 2)))
        x2 = min(img_w - 1, int(round(x + w / 2)))
        y2 = min(img_h - 1, int(round(y + h / 2)))

        label = str(prediction.get("class", "unknown"))
        confidence = prediction.get("confidence")
        text = f"{label} {confidence:.2f}" if isinstance(confidence, (float, int)) else label

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        text_y1 = max(0, y1 - text_h - baseline - 6)
        text_y2 = max(text_h + baseline + 6, y1)
        text_x2 = min(img_w - 1, x1 + text_w + 8)

        cv2.rectangle(image, (x1, text_y1), (text_x2, text_y2), (0, 220, 0), -1)
        cv2.putText(
            image,
            text,
            (x1 + 4, text_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"No se pudo escribir la imagen de salida: {output_path}")


def local_endpoint_reachable(api_url: str, timeout_sec: float = 2.0) -> bool:
    parsed = urlparse(api_url)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def select_api_url(mode: str, target: str) -> str:
    if mode == "local":
        return LOCAL_API_URL
    if target == "workflow":
        return CLOUD_WORKFLOW_API_URL
    return CLOUD_MODEL_API_URL


def run_inference() -> dict[str, Any]:
    mode = MODE
    if mode not in {"local", "cloud"}:
        raise ValueError("TP2_INFERENCE_MODE debe ser local o cloud.")

    api_url = select_api_url(mode, TARGET)

    if mode == "local" and not local_endpoint_reachable(api_url):
        raise ConnectionError(
            f"No hay servicio de inferencia local accesible en {api_url}. "
            "Arranca tu Roboflow Inference server en el EPC o cambia TP2_INFERENCE_MODE=cloud."
        )

    client = InferenceHTTPClient(api_url=api_url, api_key=API_KEY)

    if TARGET == "workflow":
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW,
            images={"image": str(IMAGE_PATH)},
            use_cache=True,
        )
    elif TARGET == "model":
        if not MODEL_ID:
            raise ValueError(
                "Para TP2_INFERENCE_TARGET=model debes definir ROBOFLOW_MODEL_ID, "
                "por ejemplo tu-proyecto/1."
            )
        result = client.infer(str(IMAGE_PATH), model_id=MODEL_ID)
    else:
        raise ValueError("TP2_INFERENCE_TARGET debe ser workflow o model.")

    return {
        "mode": mode,
        "target": TARGET,
        "api_url": api_url,
        "result": result,
    }


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"No existe la imagen de entrada: {IMAGE_PATH}")

    payload = run_inference()
    result = payload["result"]
    predictions = extract_predictions(result)
    draw_predictions(IMAGE_PATH, OUTPUT_IMAGE_PATH, predictions)

    print(
        json.dumps(
            {
                "mode": payload["mode"],
                "target": payload["target"],
                "api_url": payload["api_url"],
                "input_image": str(IMAGE_PATH),
                "output_image": str(OUTPUT_IMAGE_PATH),
                "detections": len(predictions),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
