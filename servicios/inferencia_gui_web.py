from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import socket
import traceback
from typing import Any
from urllib.parse import urlparse

import cv2
import gradio as gr
from inference_sdk import InferenceHTTPClient


DEFAULT_MODE = os.getenv("TP2_INFERENCE_MODE", "local")
DEFAULT_TARGET = os.getenv("TP2_INFERENCE_TARGET", "workflow")
DEFAULT_LOCAL_API_URL = os.getenv("ROBOFLOW_LOCAL_API_URL", "http://127.0.0.1:9001")
DEFAULT_CLOUD_WORKFLOW_API_URL = os.getenv(
    "ROBOFLOW_CLOUD_WORKFLOW_API_URL", "https://serverless.roboflow.com"
)
DEFAULT_CLOUD_MODEL_API_URL = os.getenv(
    "ROBOFLOW_CLOUD_MODEL_API_URL", "https://detect.roboflow.com"
)
DEFAULT_API_KEY = os.getenv("ROBOFLOW_API_KEY", "0tD81VP5ij7wTYq6W7yA")
DEFAULT_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "1-v8mk1")
DEFAULT_WORKFLOW = os.getenv("ROBOFLOW_WORKFLOW", "custom-workflow-2")
DEFAULT_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "")
DEFAULT_OUTPUT_DIR = os.getenv("TP2_OUTPUT_DIR", str(Path.cwd() / "outputs"))


def normalize_input_files(files: Any) -> list[Path]:
    if not files:
        return []

    normalized: list[Path] = []
    for item in files:
        if isinstance(item, str):
            normalized.append(Path(item))
            continue

        if isinstance(item, dict):
            candidate = item.get("path") or item.get("name")
            if candidate:
                normalized.append(Path(candidate))
            continue

        name = getattr(item, "name", None)
        if name:
            normalized.append(Path(name))

    return normalized


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


def infer_one_image(
    client: InferenceHTTPClient,
    image_path: Path,
    target: str,
    workspace: str,
    workflow: str,
    model_id: str,
):
    if target == "workflow":
        return client.run_workflow(
            workspace_name=workspace,
            workflow_id=workflow,
            images={"image": str(image_path)},
            use_cache=True,
        )

    if target == "model":
        if not model_id:
            raise ValueError("En modo model debes indicar Model ID (ej: proyecto/1).")
        return client.infer(str(image_path), model_id=model_id)

    raise ValueError("target debe ser workflow o model.")


def select_api_url(
    mode: str,
    target: str,
    local_api_url: str,
    cloud_workflow_api_url: str,
    cloud_model_api_url: str,
) -> str:
    if mode == "local":
        return local_api_url.strip()
    if target == "workflow":
        return cloud_workflow_api_url.strip()
    return cloud_model_api_url.strip()


def run_batch(
    files: Any,
    mode: str,
    target: str,
    local_api_url: str,
    cloud_workflow_api_url: str,
    cloud_model_api_url: str,
    api_key: str,
    workspace: str,
    workflow: str,
    model_id: str,
    output_dir: str,
):
    image_paths = normalize_input_files(files)
    if not image_paths:
        return [], "No se seleccionaron archivos.", ""

    mode = (mode or "local").strip().lower()
    target = (target or "workflow").strip().lower()
    if mode not in {"local", "cloud"}:
        return [], "ERROR: mode debe ser local o cloud.", ""
    if target not in {"workflow", "model"}:
        return [], "ERROR: target debe ser workflow o model.", ""

    api_url = select_api_url(
        mode=mode,
        target=target,
        local_api_url=local_api_url,
        cloud_workflow_api_url=cloud_workflow_api_url,
        cloud_model_api_url=cloud_model_api_url,
    )

    if mode == "local" and not local_endpoint_reachable(api_url):
        return (
            [],
            f"ERROR: no hay endpoint local accesible en {api_url}. Inicia Roboflow Inference en el EPC o cambia a cloud.",
            "",
        )

    if not api_key.strip():
        return [], "ERROR: API Key vacia.", ""

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key.strip())
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gallery_items: list[tuple[str, str]] = []
    log_lines: list[str] = [
        f"Modo={mode} | Target={target} | API={api_url}",
        f"Total imagenes={len(image_paths)}",
    ]
    all_results: dict[str, Any] = {}

    for image_path in image_paths:
        try:
            result = infer_one_image(
                client=client,
                image_path=image_path,
                target=target,
                workspace=workspace.strip(),
                workflow=workflow.strip(),
                model_id=model_id.strip(),
            )
            predictions = extract_predictions(result)

            output_path = out_dir / f"{image_path.stem}_pred{image_path.suffix}"
            draw_predictions(image_path, output_path, predictions)

            gallery_items.append((str(output_path), image_path.name))
            all_results[image_path.name] = result
            log_lines.append(
                f"OK | {image_path.name} | detecciones={len(predictions)} | salida={output_path}"
            )
        except Exception as exc:
            log_lines.append(f"ERROR | {image_path.name} | {exc}")
            log_lines.append(traceback.format_exc().strip())

    return gallery_items, "\n".join(log_lines), json.dumps(all_results, indent=2, ensure_ascii=False)


def build_ui():
    with gr.Blocks(title="TP2 Inference GUI") as demo:
        gr.Markdown(
            "# TP2 Inference GUI\n"
            "Selecciona una o varias imagenes, ejecuta inferencia y visualiza resultados anotados.\n"
            "Puedes alternar entre inferencia local en EPC y cloud Roboflow."
        )

        with gr.Row():
            mode = gr.Radio(
                choices=["local", "cloud"],
                value=DEFAULT_MODE if DEFAULT_MODE in {"local", "cloud"} else "local",
                label="Modo de inferencia",
            )
            target = gr.Radio(
                choices=["workflow", "model"],
                value=DEFAULT_TARGET if DEFAULT_TARGET in {"workflow", "model"} else "workflow",
                label="Tipo de inferencia",
            )

        with gr.Row():
            local_api_url = gr.Textbox(label="Local API URL (EPC)", value=DEFAULT_LOCAL_API_URL)
            cloud_workflow_api_url = gr.Textbox(
                label="Cloud Workflow API URL", value=DEFAULT_CLOUD_WORKFLOW_API_URL
            )

        cloud_model_api_url = gr.Textbox(
            label="Cloud Model API URL", value=DEFAULT_CLOUD_MODEL_API_URL
        )
        api_key = gr.Textbox(label="API Key", value=DEFAULT_API_KEY, type="password")

        with gr.Row():
            workspace = gr.Textbox(label="Workspace (workflow)", value=DEFAULT_WORKSPACE)
            workflow = gr.Textbox(label="Workflow ID", value=DEFAULT_WORKFLOW)

        model_id = gr.Textbox(
            label="Model ID (model mode, ej: proyecto/1)", value=DEFAULT_MODEL_ID
        )

        output_dir = gr.Textbox(label="Directorio de salida", value=DEFAULT_OUTPUT_DIR)
        file_input = gr.File(label="Imagenes", file_count="multiple", file_types=["image"])

        run_btn = gr.Button("Ejecutar inferencia")

        gallery = gr.Gallery(label="Imagenes anotadas", columns=3, height="auto")
        log_output = gr.Textbox(label="Log", lines=12)
        json_output = gr.Code(label="JSON de predicciones", language="json")

        run_btn.click(
            fn=run_batch,
            inputs=[
                file_input,
                mode,
                target,
                local_api_url,
                cloud_workflow_api_url,
                cloud_model_api_url,
                api_key,
                workspace,
                workflow,
                model_id,
                output_dir,
            ],
            outputs=[gallery, log_output, json_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="TP2 GUI web de inferencia por lotes")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
