from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import traceback
from typing import Any

import gradio as gr

from roboflow_runtime import (
    DEFAULT_API_KEY,
    DEFAULT_CLOUD_MODEL_API_URL,
    DEFAULT_CLOUD_WORKFLOW_API_URL,
    DEFAULT_LOCAL_API_URL,
    DEFAULT_MODE,
    DEFAULT_MODEL_ID,
    DEFAULT_TARGET,
    DEFAULT_WORKFLOW,
    DEFAULT_WORKSPACE,
    InferenceConfig,
    create_client,
    draw_predictions,
    extract_predictions,
    infer_one_image,
    local_endpoint_reachable,
    resolve_target,
)


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

    config = InferenceConfig(
        mode=mode,
        target=resolve_target(target, model_id.strip()),
        local_api_url=local_api_url.strip(),
        cloud_workflow_api_url=cloud_workflow_api_url.strip(),
        cloud_model_api_url=cloud_model_api_url.strip(),
        api_key=api_key.strip(),
        workspace=workspace.strip(),
        workflow=workflow.strip(),
        model_id=model_id.strip(),
    )

    try:
        config.validate()
    except Exception as exc:
        return [], f"ERROR: {exc}", ""

    if config.mode == "local" and not local_endpoint_reachable(config.api_url):
        return (
            [],
            f"ERROR: no hay endpoint local accesible en {config.api_url}. "
            "Inicia Roboflow Inference en el EPC, en la Jetson, o cambia a cloud.",
            "",
        )

    if not config.api_key:
        return [], "ERROR: API Key vacia.", ""

    client = create_client(config)
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gallery_items: list[tuple[str, str]] = []
    log_lines: list[str] = [
        f"Modo={config.mode} | Target={config.target} | API={config.api_url}",
        f"Total imagenes={len(image_paths)}",
    ]
    all_results: dict[str, Any] = {}

    for image_path in image_paths:
        try:
            result = infer_one_image(client=client, image_path=image_path, config=config)
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
            "Puedes alternar entre inferencia local en EPC o Jetson y cloud Roboflow."
        )

        with gr.Row():
            mode = gr.Radio(
                choices=["local", "cloud"],
                value=DEFAULT_MODE if DEFAULT_MODE in {"local", "cloud"} else "local",
                label="Modo de inferencia",
            )
            target = gr.Radio(
                choices=["workflow", "model"],
                value=resolve_target(DEFAULT_TARGET, DEFAULT_MODEL_ID),
                label="Tipo de inferencia",
            )

        with gr.Row():
            local_api_url = gr.Textbox(label="Local API URL (EPC o Jetson)", value=DEFAULT_LOCAL_API_URL)
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
