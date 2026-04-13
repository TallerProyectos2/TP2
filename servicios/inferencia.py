from __future__ import annotations

from pathlib import Path
import json
import os

from roboflow_runtime import (
    InferenceConfig,
    create_client,
    draw_predictions,
    extract_predictions,
    infer_one_image,
    local_endpoint_reachable,
)


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = Path(os.getenv("TP2_TEST_IMAGE", BASE_DIR / "test.jpg")).expanduser().resolve()
OUTPUT_IMAGE_PATH = Path(
    os.getenv("TP2_OUTPUT_IMAGE", BASE_DIR / f"{IMAGE_PATH.stem}_pred{IMAGE_PATH.suffix}")
).expanduser().resolve()
CONFIG = InferenceConfig.from_env()


def run_inference() -> dict:
    CONFIG.validate()
    if CONFIG.mode == "local" and not local_endpoint_reachable(CONFIG.api_url):
        raise ConnectionError(
            f"No hay servicio de inferencia local accesible en {CONFIG.api_url}. "
            "Arranca tu Roboflow Inference server o cambia TP2_INFERENCE_MODE=cloud."
        )

    client = create_client(CONFIG)
    result = infer_one_image(client, IMAGE_PATH, CONFIG)

    return {
        "mode": CONFIG.mode,
        "target": CONFIG.target,
        "api_url": CONFIG.api_url,
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
